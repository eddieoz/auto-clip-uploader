"""Tests for the disk-backed PublishQueue and PublishScheduler.

These cover FIFO ordering, status transitions, retry caps, persistence/reload,
crash recovery of stale 'publishing' entries, last_publish_time continuity,
valid-JSON-after-save, stats() correctness, and a basic scheduler smoke test.
The scheduler's PostizPublisher is mocked so no network/heavy deps are needed.
"""

import json
import threading
import time

import pytest

from social_media_publisher.publish_queue import (
    PublishQueue,
    PublishScheduler,
    QueueEntry,
    STATUS_PENDING,
    STATUS_PUBLISHING,
    STATUS_PUBLISHED,
    STATUS_FAILED,
)


@pytest.fixture
def queue_path(tmp_path):
    return tmp_path / "publish_queue.json"


@pytest.fixture
def queue(queue_path):
    return PublishQueue(str(queue_path))


# ---- QueueEntry ------------------------------------------------------

def test_queue_entry_roundtrips_through_dict():
    entry = QueueEntry(id="x", output_dir="/o", video_name="v", source_link="https://s")
    d = entry.to_dict()
    restored = QueueEntry.from_dict(d)
    assert restored.id == entry.id
    assert restored.output_dir == entry.output_dir
    assert restored.source_link == entry.source_link
    assert restored.status == STATUS_PENDING


def test_queue_entry_from_dict_tolerates_missing_fields():
    # Older/smaller state files should not break loading.
    entry = QueueEntry.from_dict({"output_dir": "/o"})
    assert entry.output_dir == "/o"
    assert entry.video_name == "o"
    assert entry.status == STATUS_PENDING
    assert entry.attempts == 0


# ---- FIFO + transitions ---------------------------------------------

def test_next_pending_returns_fifo_order(queue):
    a = queue.enqueue("/a", "a")
    b = queue.enqueue("/b", "b")
    c = queue.enqueue("/c", "c")

    assert queue.next_pending().id == a.id
    queue.mark_publishing(a)
    assert queue.next_pending().id == b.id
    queue.mark_published(b)
    assert queue.next_pending().id == c.id


def test_next_pending_returns_none_when_drained(queue):
    assert queue.next_pending() is None
    entry = queue.enqueue("/a", "a")
    assert queue.next_pending().id == entry.id
    queue.mark_publishing(entry)
    queue.mark_published(entry)
    assert queue.next_pending() is None


def test_mark_published_sets_published_at_and_time(queue):
    entry = queue.enqueue("/a", "a")
    assert queue.get_last_publish_time() is None

    queue.mark_publishing(entry)
    queue.mark_published(entry)

    assert entry.status == STATUS_PUBLISHED
    assert entry.published_at is not None
    assert queue.get_last_publish_time() == entry.published_at


def test_mark_failed_requeues_until_retries_exhausted(queue):
    entry = queue.enqueue("/a", "a")

    # Attempts 1 and 2 -> still pending (retried).
    queue.mark_failed(entry, "boom", max_retries=3)
    assert entry.status == STATUS_PENDING
    assert entry.attempts == 1
    queue.mark_failed(entry, "boom", max_retries=3)
    assert entry.status == STATUS_PENDING
    assert entry.attempts == 2

    # 3rd attempt exhausts the cap -> failed (terminal).
    queue.mark_failed(entry, "boom", max_retries=3)
    assert entry.status == STATUS_FAILED
    assert entry.attempts == 3
    assert entry.last_error == "boom"


# ---- Persistence + crash recovery -----------------------------------

def test_state_persists_and_reloads(queue_path):
    q1 = PublishQueue(str(queue_path))
    q1.enqueue("/a", "a", source_link="https://s")
    q1.enqueue("/b", "b")

    # New instance reads the same on-disk state.
    q2 = PublishQueue(str(queue_path))
    first = q2.next_pending()
    assert first.video_name == "a"
    assert first.source_link == "https://s"


def test_stale_publishing_entries_recover_to_pending_on_load(queue_path):
    q1 = PublishQueue(str(queue_path))
    entry = q1.enqueue("/a", "a")
    q1.mark_publishing(entry)
    # Simulate a crash: entry is left "publishing" on disk.

    q2 = PublishQueue(str(queue_path))
    recovered = q2.next_pending()
    assert recovered is not None
    assert recovered.id == entry.id
    assert recovered.status == STATUS_PENDING


def test_last_publish_time_persists_across_instances(queue_path):
    q1 = PublishQueue(str(queue_path))
    entry = q1.enqueue("/a", "a")
    q1.mark_publishing(entry)
    q1.mark_published(entry)
    persisted = q1.get_last_publish_time()

    q2 = PublishQueue(str(queue_path))
    assert q2.get_last_publish_time() == persisted


def test_save_produces_valid_json(queue_path, queue):
    queue.enqueue("/a", "a")
    queue.enqueue("/b", "b")

    with open(queue_path) as f:
        data = json.load(f)

    assert "entries" in data
    assert "last_publish_time" in data
    assert len(data["entries"]) == 2


def test_corrupt_state_file_does_not_crash(queue_path):
    queue_path.write_text("not valid json {{{")
    q = PublishQueue(str(queue_path))  # should not raise
    assert q.next_pending() is None


# ---- stats() --------------------------------------------------------

def test_stats_counts_by_status(queue):
    e1 = queue.enqueue("/a", "a")
    e2 = queue.enqueue("/b", "b")
    e3 = queue.enqueue("/c", "c")
    e4 = queue.enqueue("/d", "d")

    queue.mark_publishing(e1)
    queue.mark_published(e2)
    queue.mark_failed(e3, "err", max_retries=1)  # immediately failed (cap=1)

    stats = queue.stats()
    # e1=publishing, e2=published, e3=failed, e4=pending
    assert stats == {
        "pending": 1,      # e4
        "publishing": 1,   # e1
        "published": 1,    # e2
        "failed": 1,       # e3
        "total": 4,
    }


# ---- Scheduler smoke test (PostizPublisher mocked) -------------------

def test_scheduler_publishes_all_entries_with_interval_zero(queue_path, monkeypatch):
    """With interval=0 the scheduler should drain the queue promptly."""
    import sys

    calls = []

    class FakePublisher:
        def __init__(self, output_dir, source_link=None):
            pass

        def publish(self):
            calls.append(time.time())
            return {"success": True, "platforms": ["twitter"]}

    # Make the scheduler's lazy `from .publisher import PostizPublisher` resolve to our fake.
    monkeypatch.setitem(
        sys.modules, "social_media_publisher.publisher",
        type("M", (), {"PostizPublisher": FakePublisher})(),
    )

    queue = PublishQueue(str(queue_path))
    queue.enqueue("/a", "a")
    queue.enqueue("/b", "b")

    sched = PublishScheduler(queue, interval_minutes=0, max_retries=3)
    sched.start()
    try:
        # Wait until both are published.
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sched.stop(timeout=5)

    assert len(calls) == 2
    stats = queue.stats()
    assert stats["published"] == 2
    assert stats["pending"] == 0


def test_scheduler_marks_failed_on_exception(queue_path, monkeypatch):
    import sys

    class FakePublisher:
        def __init__(self, output_dir, source_link=None):
            pass

        def publish(self):
            raise RuntimeError("upload exploded")

    monkeypatch.setitem(
        sys.modules, "social_media_publisher.publisher",
        type("M", (), {"PostizPublisher": FakePublisher})(),
    )

    queue = PublishQueue(str(queue_path))
    queue.enqueue("/a", "a")

    sched = PublishScheduler(queue, interval_minutes=0, max_retries=2)
    sched.start()
    try:
        deadline = time.time() + 5
        while queue.stats()["failed"] == 0 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sched.stop(timeout=5)

    stats = queue.stats()
    assert stats["failed"] == 1
    assert stats["pending"] == 0
