"""Tests for batch clip selection.

Covers candidate reading from the reelsfy output artifacts, the disk-backed
SelectionStore, the one-call LLM ranking and its deterministic fallback, the
enqueue-then-mark ordering in select_and_enqueue, and the SelectionScheduler
trigger conditions (quiet period, batch cap, editing busy).
"""

import json
from pathlib import Path

import pytest

from social_media_publisher.clip_selection import (
    ClipCandidate,
    SelectionStore,
    SelectionScheduler,
    rank_candidates,
    read_candidate,
    select_and_enqueue,
)


CONTENT_JSON = {
    "metadata": {
        "title": "Ethereum Classic A Comunidade que Nunca Morreu",
        "description": "Um post sobre a comunidade ETC.",
        "tags": "ethereum classic, etc, comunidade",
        "source_link": "https://youtube.com/live/OV7XCxgeM3E",
    },
    "segments": [
        {
            "start_time": "00:00:15,958",
            "end_time": "00:01:14,959",
            "title": "A Comunidade Que Mantém o Ethereum Classic Vivo",
            "description": "Você já ouviu falar da comunidade mais raiz do crypto?",
            "tags": "#ethereumclassic #comunidade",
            "hook_start_time": "00:00:29,160",
            "hook_end_time": "00:00:36,240",
            "hook_text": "A comunidade é muito boa, a galera é boa",
            "duration": 59.001,
        }
    ],
}

AUDIO_SUMMARY = """Audio Analysis Summary
=====================

Segment 1:
Start: 64.519, End: 74.959
Energy: 0.50
Speech Rate: 0.50
Emotional Content: 0.50
Onset Strength: 0.50
Combined Score: 1.00
"""


def make_output_dir(tmp_path, name, content=None, audio=None):
    """Create a minimal reelsfy output directory for a clip."""
    d = tmp_path / name
    d.mkdir()
    if content is not None:
        (d / "content.txt").write_text(
            content if isinstance(content, str) else json.dumps(content)
        )
    if audio is not None:
        (d / "audio_analysis_summary.txt").write_text(audio)
    return d


# ---- read_candidate --------------------------------------------------

def test_read_candidate_parses_title_hook_and_tags(tmp_path):
    d = make_output_dir(tmp_path, "clip-a", content=CONTENT_JSON, audio=AUDIO_SUMMARY)

    c = read_candidate(str(d), "clip-a", "https://s", edited_at=1000.0)

    assert c.video_name == "clip-a"
    assert c.output_dir == str(d)
    assert c.source_link == "https://s"
    assert c.edited_at == 1000.0
    assert c.title == "A Comunidade Que Mantém o Ethereum Classic Vivo"
    assert c.hook_text == "A comunidade é muito boa, a galera é boa"
    assert c.hook_duration == pytest.approx(7.08, abs=0.01)
    assert c.duration == pytest.approx(59.001)
    assert c.metadata_error is None


def test_read_candidate_reads_audio_combined_score(tmp_path):
    d = make_output_dir(tmp_path, "clip-a", content=CONTENT_JSON, audio=AUDIO_SUMMARY)

    c = read_candidate(str(d), "clip-a", None)

    assert c.audio_score == pytest.approx(1.0)


def test_read_candidate_still_returns_candidate_when_content_missing(tmp_path):
    # A clip whose metadata cannot be read must never be silently dropped —
    # it is ranked at the bottom instead.
    d = make_output_dir(tmp_path, "clip-b")

    c = read_candidate(str(d), "clip-b", None)

    assert c.video_name == "clip-b"
    assert c.title == ""
    assert c.metadata_error


def test_read_candidate_still_returns_candidate_when_content_is_invalid_json(tmp_path):
    d = make_output_dir(tmp_path, "clip-c", content="{not json", audio=AUDIO_SUMMARY)

    c = read_candidate(str(d), "clip-c", None)

    assert c.video_name == "clip-c"
    assert c.metadata_error
    # The audio file was still readable, so that signal survives.
    assert c.audio_score == pytest.approx(1.0)


def test_read_candidate_without_audio_file_scores_zero(tmp_path):
    d = make_output_dir(tmp_path, "clip-d", content=CONTENT_JSON)

    c = read_candidate(str(d), "clip-d", None)

    assert c.audio_score == 0.0
    assert c.metadata_error is None


# ---- SelectionStore --------------------------------------------------

@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "selection_state.json"


@pytest.fixture
def store(store_path):
    return SelectionStore(str(store_path))


def test_store_add_and_read_pending(store):
    c = ClipCandidate("clip-a", "/o/a", "https://s", edited_at=5.0)
    store.add_candidate(c)

    pending = store.pending()
    assert [p.video_name for p in pending] == ["clip-a"]


def test_store_ignores_duplicate_candidate(store):
    store.add_candidate(ClipCandidate("clip-a", "/o/a", None, edited_at=5.0))
    store.add_candidate(ClipCandidate("clip-a", "/o/a", None, edited_at=6.0))

    assert len(store.pending()) == 1


def test_store_persists_pending_across_instances(store_path):
    first = SelectionStore(str(store_path))
    first.add_candidate(ClipCandidate("clip-a", "/o/a", "https://s", edited_at=5.0))

    second = SelectionStore(str(store_path))

    assert [p.video_name for p in second.pending()] == ["clip-a"]


def test_store_persists_decisions_across_instances(store_path):
    first = SelectionStore(str(store_path))
    first.add_candidate(ClipCandidate("clip-a", "/o/a", None, edited_at=5.0))
    first.mark_decided("clip-a", rank=1, score=90.0, selected=False, reason="loser")

    second = SelectionStore(str(store_path))

    assert second.is_decided("clip-a")
    assert second.pending() == []


def test_store_tolerates_corrupt_state_file(store_path):
    store_path.write_text("{ this is not json")

    s = SelectionStore(str(store_path))

    assert s.pending() == []


def test_store_stats_counts_pending_and_decided(store):
    store.add_candidate(ClipCandidate("clip-a", "/o/a", None, edited_at=5.0))
    store.add_candidate(ClipCandidate("clip-b", "/o/b", None, edited_at=6.0))
    store.mark_decided("clip-b", rank=1, score=80.0, selected=True, reason="top")

    stats = store.stats()

    assert stats["pending"] == 1
    assert stats["decided"] == 1
    assert stats["selected"] == 1


# ---- rank_candidates -------------------------------------------------


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._content)


class FakeClient:
    def __init__(self, content=None, error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = type("Chat", (), {"completions": self.completions})()


def make_candidates(n):
    """n candidates with distinct descending audio scores."""
    return [
        ClipCandidate(f"clip-{i}", f"/o/clip-{i}", None, edited_at=float(i),
                      title=f"Title {i}", audio_score=float(10 - i))
        for i in range(n)
    ]


def ranking_json(*names_and_scores):
    return json.dumps({
        "ranking": [
            {"video_name": n, "score": s, "reason": f"because {n}"}
            for n, s in names_and_scores
        ]
    })


def test_rank_candidates_returns_llm_order_with_scores():
    client = FakeClient(ranking_json(("clip-0", 91.0), ("clip-1", 42.0), ("clip-2", 7.0)))

    ranking = rank_candidates(make_candidates(3), limit=2, client=client)

    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]
    assert ranking.ranked[0].score == 91.0
    assert ranking.ranked[0].reason == "because clip-0"
    assert ranking.source == "llm"


def test_rank_candidates_selected_respects_limit():
    client = FakeClient(ranking_json(("clip-0", 91.0), ("clip-1", 42.0), ("clip-2", 7.0)))

    ranking = rank_candidates(make_candidates(3), limit=2, client=client)

    assert [r.candidate.video_name for r in ranking.selected] == ["clip-0", "clip-1"]
    # Rejected clips must survive so their decision can be recorded.
    assert [r.candidate.video_name for r in ranking.rejected] == ["clip-2"]


def test_rank_candidates_selects_all_when_fewer_than_limit():
    ranking = rank_candidates(make_candidates(2), limit=6, client=FakeClient(ranking_json(("clip-1", 9.0))))

    # The limit is a ceiling, never a reason to hold clips back.
    assert len(ranking.selected) == 2


def test_rank_candidates_falls_back_when_llm_raises():
    ranking = rank_candidates(make_candidates(3), limit=2, client=FakeClient(error=RuntimeError("boom")))

    # Fallback order is audio score descending, which here matches clip-0,1,2.
    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]
    assert ranking.source == "fallback"
    assert ranking.error


def test_rank_candidates_falls_back_when_llm_returns_garbage():
    ranking = rank_candidates(make_candidates(3), limit=2, client=FakeClient("I am not JSON at all"))

    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]
    assert ranking.source == "fallback"


def test_rank_candidates_falls_back_when_llm_returns_unknown_names_only():
    client = FakeClient(ranking_json(("clip-zzz", 91.0)))

    ranking = rank_candidates(make_candidates(3), limit=2, client=client)

    # Nothing usable came back, so every candidate is ranked by fallback.
    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]
    assert ranking.source == "fallback"


def test_rank_candidates_appends_omitted_candidates_without_duplicates():
    client = FakeClient(ranking_json(("clip-2", 50.0)))

    ranking = rank_candidates(make_candidates(3), limit=2, client=client)

    names = [r.candidate.video_name for r in ranking.ranked]
    assert names == ["clip-2", "clip-0", "clip-1"]
    assert len(names) == len(set(names))
    assert ranking.source == "llm-partial"


def test_rank_candidates_never_raises_when_no_client_is_available(monkeypatch):
    import social_media_publisher.clip_selection as mod

    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(mod, "_build_client", boom)

    ranking = rank_candidates(make_candidates(3), limit=2, client=None)

    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]
    assert ranking.source == "fallback"


def test_rank_candidates_retries_without_response_format_when_rejected():
    # Custom OpenAI-compatible endpoints may not support structured output;
    # the call is retried once without response_format.
    class RejectingThenOk:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise TypeError("unexpected keyword argument 'response_format'")
            return FakeResponse(ranking_json(("clip-0", 99.0), ("clip-1", 1.0), ("clip-2", 0.0)))

    completions = RejectingThenOk()
    client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()

    ranking = rank_candidates(make_candidates(3), limit=2, client=client)

    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert [r.candidate.video_name for r in ranking.ranked] == ["clip-0", "clip-1", "clip-2"]


def test_rank_candidates_of_empty_batch_is_empty():
    ranking = rank_candidates([], limit=6, client=FakeClient(ranking_json()))

    assert ranking.ranked == []
    assert ranking.selected == []
    assert ranking.rejected == []


# ---- select_and_enqueue ---------------------------------------------

@pytest.fixture
def publish_queue(tmp_path):
    from social_media_publisher.publish_queue import PublishQueue

    return PublishQueue(str(tmp_path / "publish_queue.json"))


def seed_pending(store, names):
    for name in names:
        store.add_candidate(
            ClipCandidate(name, f"/o/{name}", f"https://s/{name}", edited_at=1.0, audio_score=1.0)
        )


def test_select_and_enqueue_enqueues_only_the_top_n_in_rank_order(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1", "clip-2"])
    client = FakeClient(ranking_json(("clip-2", 99.0), ("clip-1", 50.0), ("clip-0", 10.0)))

    ranking = select_and_enqueue(store, publish_queue, limit=2, client=client)

    assert [e.video_name for e in publish_queue._entries] == ["clip-2", "clip-1"]
    assert [r.candidate.video_name for r in ranking.selected] == ["clip-2", "clip-1"]


def test_select_and_enqueue_passes_source_link_through(store, publish_queue):
    seed_pending(store, ["clip-0"])
    client = FakeClient(ranking_json(("clip-0", 99.0)))

    select_and_enqueue(store, publish_queue, limit=2, client=client)

    assert publish_queue._entries[0].source_link == "https://s/clip-0"


def test_select_and_enqueue_records_rejected_clips_as_not_selected(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1", "clip-2"])
    client = FakeClient(ranking_json(("clip-2", 99.0), ("clip-1", 50.0), ("clip-0", 10.0)))

    select_and_enqueue(store, publish_queue, limit=2, client=client)

    decisions = store.decisions()
    assert decisions["clip-2"]["selected"] is True
    assert decisions["clip-1"]["selected"] is True
    assert decisions["clip-0"]["selected"] is False
    # Nothing is dropped from the pending list without a decision.
    assert store.pending() == []


def test_select_and_enqueue_records_rank_position(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1", "clip-2"])
    client = FakeClient(ranking_json(("clip-2", 99.0), ("clip-1", 50.0), ("clip-0", 10.0)))

    select_and_enqueue(store, publish_queue, limit=2, client=client)

    decisions = store.decisions()
    assert decisions["clip-2"]["rank"] == 1
    assert decisions["clip-1"]["rank"] == 2
    assert decisions["clip-0"]["rank"] == 3


def test_select_and_enqueue_clears_pending_when_nothing_is_pending(store, publish_queue):
    result = select_and_enqueue(store, publish_queue, limit=6, client=FakeClient(ranking_json()))

    assert result is None
    assert publish_queue.stats()["total"] == 0


def test_select_and_enqueue_is_a_noop_when_re_run_on_the_same_batch(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1", "clip-2"])
    client = FakeClient(ranking_json(("clip-2", 99.0), ("clip-1", 50.0), ("clip-0", 10.0)))

    select_and_enqueue(store, publish_queue, limit=2, client=client)
    select_and_enqueue(store, publish_queue, limit=2, client=client)

    assert publish_queue.stats()["total"] == 2


def test_select_and_enqueue_skips_a_candidate_already_in_the_publish_queue(store, publish_queue):
    # A crash between enqueue and mark_decided leaves the clip pending AND
    # queued; it must never be published twice.
    seed_pending(store, ["clip-0", "clip-1"])
    publish_queue.enqueue("/o/clip-0", "clip-0", "https://s/clip-0")
    client = FakeClient(ranking_json(("clip-0", 99.0), ("clip-1", 50.0)))

    select_and_enqueue(store, publish_queue, limit=2, client=client)

    assert [e.video_name for e in publish_queue._entries] == ["clip-0", "clip-1"]
    assert publish_queue.stats()["total"] == 2


def test_select_and_enqueue_still_decides_when_the_llm_fails(store, publish_queue):
    # A provider outage degrades to the fallback order; it must still publish.
    seed_pending(store, ["clip-0", "clip-1"])
    client = FakeClient(error=RuntimeError("boom"))

    ranking = select_and_enqueue(store, publish_queue, limit=1, client=client)

    assert ranking.source == "fallback"
    assert [e.video_name for e in publish_queue._entries] == ["clip-0"]
    assert store.decisions()["clip-1"]["selected"] is False


def test_publish_queue_reports_already_queued_names(publish_queue):
    from social_media_publisher.publish_queue import (
        STATUS_PENDING, STATUS_PUBLISHING, STATUS_PUBLISHED,
    )

    a = publish_queue.enqueue("/o/a", "a")
    b = publish_queue.enqueue("/o/b", "b")
    c = publish_queue.enqueue("/o/c", "c")
    publish_queue.mark_published(b)

    queued = publish_queue.active_video_names()

    assert set(queued) == {"a", "c"}


# ---- SelectionScheduler ----------------------------------------------


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def build_scheduler(store, publish_queue, clock, **kwargs):
    opts = dict(
        store=store,
        publish_queue=publish_queue,
        limit=2,
        quiet_seconds=600.0,
        batch_cap=0,
        enabled=True,
        is_editing_busy=lambda: False,
        now_fn=clock,
        client=FakeClient(ranking_json(("clip-0", 99.0), ("clip-1", 50.0), ("clip-2", 10.0))),
        logger=lambda *a, **k: None,
    )
    opts.update(kwargs)
    sched = SelectionScheduler(**opts)
    # monitor.py calls on_edit_finished() after every edit; the tests seed the
    # same way so the quiet period starts at the clock's current time.
    sched.on_edit_finished()
    return sched


def test_scheduler_does_not_fire_when_nothing_is_pending(store, publish_queue):
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock)

    assert sched.maybe_select() is False
    assert publish_queue.stats()["total"] == 0


def test_scheduler_does_not_fire_before_the_quiet_period(store, publish_queue):
    seed_pending(store, ["clip-0"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock)

    clock.advance(599)
    assert sched.maybe_select() is False
    assert publish_queue.stats()["total"] == 0


def test_scheduler_fires_after_the_quiet_period(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock)

    clock.advance(600)
    assert sched.maybe_select() is True
    assert [e.video_name for e in publish_queue._entries] == ["clip-0", "clip-1"]


def test_scheduler_does_not_fire_while_editing_is_busy(store, publish_queue):
    seed_pending(store, ["clip-0"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock, is_editing_busy=lambda: True)

    clock.advance(10_000)
    assert sched.maybe_select() is False
    assert publish_queue.stats()["total"] == 0


def test_scheduler_fires_early_once_the_batch_cap_is_reached(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1", "clip-2"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock, batch_cap=3)

    # Quiet period has NOT elapsed; the cap is what allows an early firing.
    assert sched.maybe_select() is True
    assert [e.video_name for e in publish_queue._entries] == ["clip-0", "clip-1"]


def test_scheduler_batch_cap_does_not_fire_below_the_cap(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock, batch_cap=5)

    assert sched.maybe_select() is False


def test_scheduler_never_fires_when_disabled(store, publish_queue):
    seed_pending(store, ["clip-0"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock, enabled=False)

    clock.advance(10_000)
    assert sched.maybe_select() is False
    assert publish_queue.stats()["total"] == 0


def test_scheduler_can_fire_again_for_a_later_batch(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1"])
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock)

    clock.advance(600)
    assert sched.maybe_select() is True

    # A new clip edited after the first batch triggers a second selection.
    sched.on_edit_finished()
    seed_pending(store, ["clip-9"])
    assert sched.maybe_select() is False
    clock.advance(600)
    assert sched.maybe_select() is True
    assert [e.video_name for e in publish_queue._entries] == ["clip-0", "clip-1", "clip-9"]


def test_scheduler_records_last_edit_time_on_edit_finished(store, publish_queue):
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock)

    sched.on_edit_finished()

    assert sched.last_edit_finished() == clock.now


def test_scheduler_thread_starts_and_stops(store, publish_queue):
    clock = FakeClock()
    sched = build_scheduler(store, publish_queue, clock, poll_seconds=0.01)

    sched.start()
    sched.stop(timeout=2)

    assert sched._thread is not None
    assert not sched._thread.is_alive()


def test_scheduler_reports_ranking_to_on_decision(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1"])
    clock = FakeClock()
    seen = []
    sched = build_scheduler(
        store, publish_queue, clock, on_decision=lambda ranking: seen.append(ranking)
    )

    clock.advance(600)
    sched.maybe_select()

    assert len(seen) == 1
    assert [r.candidate.video_name for r in seen[0].selected] == ["clip-0", "clip-1"]

# ---- monitor.py wiring ------------------------------------------------

def build_handler(tmp_path, **kwargs):
    """A NewVideoHandler whose on-disk state lives in tmp_path, not the repo."""
    from monitor import NewVideoHandler

    opts = dict(
        selection_enabled=True,
        publish_queue_path=str(tmp_path / "publish_queue.json"),
        selection_state_path=str(tmp_path / "selection_state.json"),
    )
    opts.update(kwargs)
    return NewVideoHandler(**opts)


def make_edited_clip(tmp_path, name, content=CONTENT_JSON, audio=AUDIO_SUMMARY):
    """Create the artifacts _organize_output_files expects after a successful edit."""
    d = make_output_dir(tmp_path, name, content=content, audio=audio)
    (d / "final-abc_cropped000.mp4").write_bytes(b"video")
    return d


def test_handler_collects_a_candidate_instead_of_enqueuing(tmp_path):
    handler = build_handler(tmp_path)
    d = make_edited_clip(tmp_path, "clip-a")

    handler._organize_output_files(tmp_path, d, "clip-a")

    # Nothing may reach the publish queue until a ranking has run.
    assert handler.publish_queue.stats()["total"] == 0
    assert [c.video_name for c in handler.selection_store.pending()] == ["clip-a"]


def test_handler_still_enqueues_directly_when_selection_is_disabled(tmp_path):
    handler = build_handler(tmp_path, selection_enabled=False)
    d = make_edited_clip(tmp_path, "clip-a")

    handler._organize_output_files(tmp_path, d, "clip-a")

    assert handler.publish_queue.stats()["total"] == 1
    assert handler.publish_queue._entries[0].video_name == "clip-a"


def test_handler_counts_edited_total_when_collecting(tmp_path):
    handler = build_handler(tmp_path)
    d = make_edited_clip(tmp_path, "clip-a")

    handler._organize_output_files(tmp_path, d, "clip-a")

    assert handler.editing_stats()["edited_total"] == 1


def test_handler_records_last_edit_time_for_the_scheduler(tmp_path):
    handler = build_handler(tmp_path)
    d = make_edited_clip(tmp_path, "clip-a")

    handler._organize_output_files(tmp_path, d, "clip-a")

    assert handler.selection_scheduler.last_edit_finished() is not None


def test_handler_is_editing_busy_is_true_while_a_file_is_in_flight(tmp_path):
    handler = build_handler(tmp_path)

    assert handler.is_editing_busy() is False
    handler.processing_files.add("in-flight")
    assert handler.is_editing_busy() is True


def test_handler_reconciles_clips_that_are_already_in_the_publish_queue(tmp_path):
    # A restart must not re-select a clip that a previous run already queued.
    queue_path = tmp_path / "publish_queue.json"
    from social_media_publisher.publish_queue import PublishQueue

    existing = PublishQueue(str(queue_path))
    existing.enqueue("/o/clip-a", "clip-a", None)

    handler = build_handler(tmp_path)
    handler.selection_store.add_candidate(
        ClipCandidate("clip-a", "/o/clip-a", None, edited_at=1.0)
    )
    # Simulate the state a restart begins from.
    handler2 = build_handler(tmp_path)
    handler2.selection_store.add_candidate(
        ClipCandidate("clip-a", "/o/clip-a", None, edited_at=1.0)
    )

    handler2._reconcile_with_publish_queue()

    assert handler2.selection_store.is_decided("clip-a")
    assert handler2.selection_store.pending() == []


def test_handler_dashboard_reports_pending_candidates(tmp_path):
    handler = build_handler(tmp_path)
    handler.selection_store.add_candidate(
        ClipCandidate("clip-a", "/o/clip-a", None, edited_at=1.0)
    )

    rendered = handler.render_dashboard()

    assert "candidates" in rendered
    assert "1" in rendered


def test_handler_does_not_touch_the_repo_publish_queue_by_default(tmp_path, monkeypatch):
    # Guard: the handler must not silently write the live publish_queue.json.
    from monitor import NewVideoHandler

    monkeypatch.chdir(tmp_path)
    before = Path("publish_queue.json")
    assert not before.exists()

    handler = NewVideoHandler(
        publish_queue_path=str(tmp_path / "q.json"),
        selection_state_path=str(tmp_path / "s.json"),
    )
    handler.publish_queue.enqueue("/o/a", "a")

    assert not before.exists()


# ---- monitor CLI ------------------------------------------------------


def test_cli_accepts_selection_flags():
    from monitor import build_arg_parser

    args = build_arg_parser().parse_args([
        "--select-limit", "4", "--select-quiet-minutes", "45", "--select-batch-cap", "12",
    ])

    assert args.select_limit == 4
    assert args.select_quiet_minutes == 45
    assert args.select_batch_cap == 12


def test_cli_selection_flags_default_to_none_so_env_wins():
    from monitor import build_arg_parser

    args = build_arg_parser().parse_args([])

    assert args.select_limit is None
    assert args.select_quiet_minutes is None
    assert args.select_batch_cap is None
    assert args.no_selection is False


def test_cli_no_selection_disables_selection():
    from monitor import build_arg_parser

    assert build_arg_parser().parse_args(["--no-selection"]).no_selection is True


def test_cli_keeps_existing_publish_flags():
    from monitor import build_arg_parser

    args = build_arg_parser().parse_args(["--publish-delay", "30", "--no-dashboard", "--link", "https://s"])

    assert args.publish_delay == 30
    assert args.no_dashboard is True
    assert args.link == "https://s"


def test_publish_delay_is_not_overridden_by_selection_defaults():
    # --select-limit must not disturb the existing publish cadence flag.
    from monitor import build_arg_parser

    args = build_arg_parser().parse_args(["--publish-delay", "0", "--select-limit", "6"])

    assert args.publish_delay == 0


# ---- CLI (read-only by design) ---------------------------------------

def test_cli_ranks_existing_output_dirs_without_queueing(tmp_path, capsys):
    from social_media_publisher import clip_selection

    a = make_edited_clip(tmp_path, "clip-a")
    b = make_edited_clip(tmp_path, "clip-b")

    code = clip_selection.main(
        [str(a), str(b), "--limit", "1", "--fallback"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "clip-a" in out
    assert "clip-b" in out
    # The CLI must never touch the live queue or selection state.
    assert not (tmp_path / "publish_queue.json").exists()
    assert not (tmp_path / "selection_state.json").exists()


def test_cli_reports_missing_directories(tmp_path, capsys):
    from social_media_publisher import clip_selection

    code = clip_selection.main([str(tmp_path / "nope"), "--fallback"])

    assert code == 1
    assert "not a directory" in capsys.readouterr().out


def test_cli_marks_which_clips_would_be_selected(tmp_path, capsys):
    from social_media_publisher import clip_selection

    a = make_edited_clip(tmp_path, "clip-a")

    clip_selection.main([str(a), "--limit", "1", "--fallback"])

    out = capsys.readouterr().out
    assert "would publish" in out


def test_select_and_enqueue_warns_loudly_when_ranking_falls_back(store, publish_queue, caplog):
    import logging

    seed_pending(store, ["clip-0", "clip-1"])
    warnings = []
    client = FakeClient(error=RuntimeError("boom"))

    select_and_enqueue(
        store, publish_queue, limit=1, client=client,
        logger=lambda msg: warnings.append(msg),
    )

    # A degraded ranking must be visible, not silent: otherwise a monitor run
    # under the wrong interpreter just publishes in edit order forever. The
    # *reason* has to be surfaced, not merely the word "fallback".
    assert any("fallback" in w.lower() and "boom" in w for w in warnings)


def test_select_and_enqueue_does_not_warn_on_a_clean_llm_ranking(store, publish_queue):
    seed_pending(store, ["clip-0", "clip-1"])
    warnings = []
    client = FakeClient(ranking_json(("clip-0", 99.0), ("clip-1", 10.0)))

    select_and_enqueue(
        store, publish_queue, limit=1, client=client,
        logger=lambda msg: warnings.append(msg),
    )

    assert not any("fallback" in w.lower() for w in warnings)
    assert any("clip-0" in w for w in warnings)


def test_scheduler_primes_restored_batches_without_an_edit(tmp_path):
    # monitor.py restores pending candidates from a previous run; their edits are
    # long over, so the quiet period must be treated as already elapsed.
    from social_media_publisher.publish_queue import PublishQueue
    from social_media_publisher.clip_selection import SelectionScheduler, SelectionStore

    clock = FakeClock(now=10_000.0)
    store = SelectionStore(str(tmp_path / "s.json"))
    store.add_candidate(ClipCandidate("clip-0", "/o/clip-0", None, edited_at=1.0))
    queue = PublishQueue(str(tmp_path / "q.json"))

    sched = SelectionScheduler(
        store, queue, limit=1, quiet_seconds=600.0, now_fn=clock,
        client=FakeClient(ranking_json(("clip-0", 99.0))),
        logger=lambda *a, **k: None,
    )
    assert sched.maybe_select() is False  # no edit recorded yet

    sched.prime_as_quiet()
    assert sched.maybe_select() is True
    assert [e.video_name for e in queue._entries] == ["clip-0"]
