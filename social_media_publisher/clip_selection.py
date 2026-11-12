"""
Batch clip selection: rank the day's finished clips and enqueue only the best N.

The editing stage produces one clip per capture, minutes apart, so there is no
natural "batch" moment in the pipeline. This module introduces one: finished
clips are collected as candidates, and once editing goes quiet a single LLM call
ranks the whole batch so only the top N are handed to the publish queue.

The ranking is best-effort by construction. ``rank_candidates`` never raises and
always returns a total order over the batch, falling back to a deterministic
audio-score ordering when the LLM is unavailable or returns something unusable.
Losing clips are never deleted -- they stay in their output directory and the
decision is recorded in the selection state file.
"""

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional


# How often the selection scheduler re-checks its trigger conditions.
_IDLE_POLL_SECONDS = 5.0

# Where the OpenAI key lives. This is NOT the root .env that monitor.py loads,
# so this module reads the reelsfy env file itself (same wiring as reelsfy.py).
_ENV_FILE = Path(__file__).resolve().parent.parent / "reels-clips-automator" / ".env"
_DEFAULT_MODEL = "gpt-5-mini"

# Seconds to wait on the ranking call before giving up and using the fallback.
_RANKING_TIMEOUT_SECONDS = 90


def _srt_to_seconds(value: Optional[str]) -> Optional[float]:
    """Parse an SRT timestamp ("00:00:29,160") into seconds."""
    if not value or not isinstance(value, str):
        return None
    match = re.match(r"^(\d+):(\d{2}):(\d{2})[,.](\d{1,3})$", value.strip())
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000.0
    )


def _read_audio_score(output_dir: str) -> Optional[float]:
    """Read the first 'Combined Score' out of audio_analysis_summary.txt."""
    path = Path(output_dir) / "audio_analysis_summary.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^Combined Score:\s*([0-9]*\.?[0-9]+)\s*$", text, re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


@dataclass
class ClipCandidate:
    """One finished clip awaiting a selection decision.

    Every metadata field degrades independently: a clip whose content.txt is
    missing or unparseable still becomes a candidate (with ``metadata_error``
    set) so it is ranked at the bottom rather than silently dropped.
    """

    video_name: str
    output_dir: str
    source_link: Optional[str] = None
    edited_at: float = field(default_factory=time.time)
    title: str = ""
    hook_text: str = ""
    hook_duration: Optional[float] = None
    duration: Optional[float] = None
    tags: str = ""
    audio_score: float = 0.0
    metadata_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ClipCandidate":
        return cls(
            video_name=data.get("video_name", ""),
            output_dir=data.get("output_dir", ""),
            source_link=data.get("source_link"),
            edited_at=data.get("edited_at", time.time()),
            title=data.get("title", ""),
            hook_text=data.get("hook_text", ""),
            hook_duration=data.get("hook_duration"),
            duration=data.get("duration"),
            tags=data.get("tags", ""),
            audio_score=data.get("audio_score", 0.0),
            metadata_error=data.get("metadata_error"),
        )

    def fallback_key(self):
        """Deterministic ordering used when the LLM ranking is unusable.

        Strongest audio first; earliest edit wins ties so the order is stable
        and independent of dict ordering.
        """
        return (-self.audio_score, self.edited_at)


def read_candidate(
    output_dir: str,
    video_name: str,
    source_link: Optional[str] = None,
    edited_at: Optional[float] = None,
) -> ClipCandidate:
    """Build a :class:`ClipCandidate` from a reelsfy output directory."""
    candidate = ClipCandidate(
        video_name=video_name,
        output_dir=str(output_dir),
        source_link=source_link,
        edited_at=time.time() if edited_at is None else edited_at,
    )

    score = _read_audio_score(candidate.output_dir)
    if score is not None:
        candidate.audio_score = score

    content_path = Path(candidate.output_dir) / "content.txt"
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        candidate.metadata_error = "content.txt not found"
        return candidate
    except (json.JSONDecodeError, OSError) as exc:
        candidate.metadata_error = f"content.txt unreadable: {exc}"
        return candidate

    segments = content.get("segments") or []
    if not segments:
        candidate.metadata_error = "content.txt has no segments"
        return candidate

    segment = segments[0]
    candidate.title = segment.get("title") or content.get("metadata", {}).get("title", "")
    candidate.hook_text = segment.get("hook_text") or ""
    candidate.tags = segment.get("tags") or content.get("metadata", {}).get("tags", "")
    duration = segment.get("duration")
    if isinstance(duration, (int, float)):
        candidate.duration = float(duration)

    hook_start = _srt_to_seconds(segment.get("hook_start_time"))
    hook_end = _srt_to_seconds(segment.get("hook_end_time"))
    if hook_start is not None and hook_end is not None and hook_end >= hook_start:
        candidate.hook_duration = hook_end - hook_start

    return candidate


class SelectionStore:
    """
    Disk-backed record of which clips are awaiting a decision and which have had
    one.

    Persistence mirrors :class:`~social_media_publisher.publish_queue.PublishQueue`:
    the same temp-file-plus-``os.replace`` atomic write, and a corrupt state file
    is tolerated rather than crashing the pipeline. Pending candidates are
    persisted as soon as an edit finishes, so a crash mid-day does not lose them.
    """

    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self._lock = threading.RLock()
        self._pending: List[ClipCandidate] = []
        self._decided: Dict[str, Dict] = {}
        self._load()

    # ---- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"⚠️  Could not load selection state from {self.state_path}; starting fresh.")
            return

        self._pending = [ClipCandidate.from_dict(c) for c in data.get("pending", [])]
        self._decided = dict(data.get("decided", {}))

    def _save(self) -> None:
        """Atomically write state. Caller must hold ``self._lock``."""
        data = {
            "pending": [c.to_dict() for c in self._pending],
            "decided": self._decided,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=self.state_path.name + ".", suffix=".tmp", dir=str(self.state_path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.state_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ---- public API -----------------------------------------------------

    def add_candidate(self, candidate: ClipCandidate) -> None:
        """Record a finished clip as awaiting a decision."""
        with self._lock:
            if candidate.video_name in self._decided:
                return
            if any(c.video_name == candidate.video_name for c in self._pending):
                return
            self._pending.append(candidate)
            self._save()

    def pending(self) -> List[ClipCandidate]:
        with self._lock:
            return list(self._pending)

    def mark_decided(
        self,
        video_name: str,
        rank: int,
        score: Optional[float],
        selected: bool,
        reason: str = "",
    ) -> None:
        """Record a selection decision and drop the clip from the pending list."""
        with self._lock:
            self._decided[video_name] = {
                "rank": rank,
                "score": score,
                "selected": selected,
                "reason": reason,
                "decided_at": time.time(),
            }
            self._pending = [c for c in self._pending if c.video_name != video_name]
            self._save()

    def is_decided(self, video_name: str) -> bool:
        with self._lock:
            return video_name in self._decided

    def decisions(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._decided)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "decided": len(self._decided),
                "selected": sum(
                    1 for d in self._decided.values() if d.get("selected")
                ),
            }

# ---- ranking ---------------------------------------------------------


@dataclass
class RankedClip:
    """A candidate together with the verdict the ranking produced for it."""

    candidate: ClipCandidate
    score: Optional[float] = None
    reason: str = ""


@dataclass
class Ranking:
    """
    A total order over a batch of candidates.

    ``ranked`` always contains every candidate exactly once; ``selected`` and
    ``rejected`` are views over it so that losing clips can still have their
    decision recorded instead of being discarded by truncation.
    """

    ranked: List[RankedClip]
    limit: int
    source: str = "fallback"
    error: Optional[str] = None

    @property
    def selected(self) -> List[RankedClip]:
        return self.ranked[: max(0, self.limit)]

    @property
    def rejected(self) -> List[RankedClip]:
        return self.ranked[max(0, self.limit):]


def _build_client():
    """Create an OpenAI client wired to the reelsfy endpoint, or raise."""
    import openai
    from dotenv import dotenv_values

    config = dotenv_values(_ENV_FILE)
    api_key = (config.get("OPENAI_API_KEY") or "").strip("'\"")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {_ENV_FILE}")

    endpoint = (config.get("AI_ENDPOINT") or os.getenv("AI_ENDPOINT") or "").strip("'\"")
    # A hung ranking call must not hold the selection thread forever: the client
    # times out and the deterministic fallback takes over.
    return openai.OpenAI(
        api_key=api_key, base_url=endpoint or None, timeout=_RANKING_TIMEOUT_SECONDS
    )


def _resolve_model() -> str:
    from dotenv import dotenv_values

    config = dotenv_values(_ENV_FILE)
    configured = (config.get("RANKING_MODEL") or "").strip("'\"")
    if configured:
        return configured
    fallback = (config.get("OPENAI_MODEL") or "").strip("'\"")
    return fallback or os.getenv("OPENAI_MODEL") or _DEFAULT_MODEL


def _describe(candidate: ClipCandidate) -> str:
    """One block of batch context for the ranking prompt."""
    hook = candidate.hook_text or "(none)"
    duration = f"{candidate.duration:.1f}s" if candidate.duration is not None else "unknown"
    hook_span = (
        f"{candidate.hook_duration:.1f}s" if candidate.hook_duration is not None else "n/a"
    )
    return (
        f"- video_name: {candidate.video_name}\n"
        f"  title: {candidate.title or '(none)'}\n"
        f"  hook_text: {hook}\n"
        f"  hook_span: {hook_span}\n"
        f"  clip_duration: {duration}\n"
        f"  tags: {candidate.tags or '(none)'}\n"
        f"  audio_energy_score: {candidate.audio_score:.2f}\n"
        f"  metadata_error: {candidate.metadata_error or 'none'}"
    )


_RANKING_SYSTEM = (
    "You are a Social Media Content Strategist. You rank candidate short-form "
    "video clips by how likely they are to perform well on Instagram Reels, "
    "YouTube Shorts and TikTok. Judge hook strength, emotional punch, "
    "curiosity, relatability, topic breadth and how well the caption sells the "
    "clip. You never invent clips: you only reorder and score the candidates "
    "you are given, returning every one of them exactly once."
)


def _build_ranking_prompt(candidates: List[ClipCandidate]) -> str:
    listing = "\n".join(_describe(c) for c in candidates)
    return f"""Rank these {len(candidates)} finished video clips by viral potential, best first.

{listing}

Return ONLY a JSON object with this exact shape:
{{"ranking": [{{"video_name": "<exact name from above>", "score": <0-100 integer>, "reason": "<one short sentence>"}}]}}

Rules:
- Include EVERY candidate listed above exactly once. Do not omit any.
- Use the exact "video_name" strings given above; do not shorten or rename them.
- Sort by descending "score".
- "reason" is one short sentence explaining why that clip ranks where it does."""


def _request_ranking_json(client, model: str, messages):
    """Ask the endpoint for structured JSON, degrading like reelsfy does.

    Custom OpenAI-compatible endpoints may reject ``response_format``; when the
    call carrying it fails, retry once without it rather than losing the batch.
    """
    base_kwargs = {"model": model, "messages": messages, "n": 1, "stop": None}
    try:
        return client.chat.completions.create(
            response_format={"type": "json_object"}, **base_kwargs
        )
    except Exception:
        return client.chat.completions.create(**base_kwargs)


def _fallback_ranking(
    candidates: List[ClipCandidate], limit: int, error: Optional[str]
) -> Ranking:
    """Deterministic order used whenever the LLM ranking is unusable."""
    ordered = sorted(candidates, key=lambda c: c.fallback_key())
    return Ranking(
        ranked=[RankedClip(candidate=c, score=c.audio_score, reason="fallback: audio energy") for c in ordered],
        limit=limit,
        source="fallback",
        error=error,
    )


def rank_candidates(
    candidates: List[ClipCandidate],
    limit: int,
    client=None,
    model: Optional[str] = None,
) -> Ranking:
    """
    Rank a batch with a single LLM call.

    Never raises and always returns a total order over every candidate. Unknown
    names in the response are dropped and omitted candidates are appended in
    fallback order, so a partial or nonsensical response still yields a usable
    ranking.
    """
    if not candidates:
        return Ranking(ranked=[], limit=limit, source="fallback")

    if client is None:
        try:
            client = _build_client()
        except Exception as exc:
            return _fallback_ranking(candidates, limit, f"no LLM client: {exc}")

    try:
        messages = [
            {"role": "system", "content": _RANKING_SYSTEM},
            {"role": "user", "content": _build_ranking_prompt(candidates)},
        ]
        response = _request_ranking_json(
            client, model or _resolve_model(), messages
        )
        raw = response.choices[0].message.content
    except Exception as exc:
        return _fallback_ranking(candidates, limit, f"ranking call failed: {exc}")

    try:
        text = str(raw).replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        returned = parsed["ranking"]
        if not isinstance(returned, list):
            raise ValueError("ranking is not a list")
    except Exception as exc:
        return _fallback_ranking(candidates, limit, f"unparsable ranking: {exc}")

    by_name = {c.video_name: c for c in candidates}
    ranked: List[RankedClip] = []
    seen = set()
    for item in returned:
        if not isinstance(item, dict):
            continue
        name = item.get("video_name")
        candidate = by_name.get(name)
        # Drop unknown names and duplicates rather than trusting the model.
        if candidate is None or name in seen:
            continue
        seen.add(name)
        score = item.get("score")
        ranked.append(
            RankedClip(
                candidate=candidate,
                score=float(score) if isinstance(score, (int, float)) else None,
                reason=str(item.get("reason") or ""),
            )
        )

    if not ranked:
        return _fallback_ranking(candidates, limit, "ranking contained no known clips")

    omitted = [c for c in candidates if c.video_name not in seen]
    for candidate in sorted(omitted, key=lambda c: c.fallback_key()):
        ranked.append(
            RankedClip(candidate=candidate, score=candidate.audio_score,
                       reason="not ranked by model; fallback order")
        )

    return Ranking(
        ranked=ranked,
        limit=limit,
        source="llm" if not omitted else "llm-partial",
        error=None if not omitted else f"{len(omitted)} candidate(s) omitted by model",
    )


# ---- selection -------------------------------------------------------


def select_and_enqueue(
    store: SelectionStore,
    publish_queue,
    limit: int,
    client=None,
    model: Optional[str] = None,
    logger=print,
) -> Optional[Ranking]:
    """
    Rank the pending batch and enqueue only the best ``limit`` clips.

    Enqueue happens BEFORE the decision is recorded. If the process dies in
    between, the candidate is still pending while already sitting in the
    publish queue -- ``active_video_names`` then keeps it from being published
    twice on the next run, which is why the reverse order would be unsafe.

    Returns the :class:`Ranking`, or None when there is nothing to decide.
    """
    candidates = store.pending()
    if not candidates:
        return None

    already_queued = set(publish_queue.active_video_names())
    ranking = rank_candidates(candidates, limit=limit, client=client, model=model)

    logger(f"🏆 Ranked {len(candidates)} clip(s) — selecting top {limit} ({ranking.source})")
    if ranking.source != "llm" and ranking.error:
        # Never degrade silently. A ranking that quietly falls back to edit order
        # (e.g. the LLM client cannot even be constructed under this interpreter)
        # must say so, or the batch looks ranked when it was not.
        logger(f"⚠️  Ranking degraded to {ranking.source} order - {ranking.error}")
    for rank, item in enumerate(ranking.ranked, start=1):
        name = item.candidate.video_name
        score = f"{item.score:.0f}" if isinstance(item.score, (int, float)) else "-"
        logger(f"   {rank}. {name} [{score}] {item.reason}")

    for item in ranking.selected:
        candidate = item.candidate
        if candidate.video_name in already_queued:
            logger(f"⏭️  '{candidate.video_name}' is already queued; not enqueuing again")
            continue
        publish_queue.enqueue(
            candidate.output_dir, candidate.video_name, candidate.source_link
        )
        logger(f"📥 Queued '{candidate.video_name}' for publishing")

    for rank, item in enumerate(ranking.ranked, start=1):
        candidate = item.candidate
        store.mark_decided(
            candidate.video_name,
            rank=rank,
            score=item.score,
            selected=rank <= limit,
            reason=item.reason,
        )

    stats = store.stats()
    logger(
        f"✅ Selection done: {stats['selected']} selected, "
        f"{stats['decided'] - stats['selected']} not selected "
        f"({stats['decided']} decided this run)"
    )
    return ranking


class SelectionScheduler:
    """
    Daemon worker that closes a batch and runs the selection when editing goes quiet.

    Clips are edited one at a time, so the batch has no natural boundary. This
    fires when three things line up: there is at least one pending candidate,
    editing is not busy, and either the quiet period since the last finished
    edit has elapsed or the batch cap has been reached.

    ``is_editing_busy``, ``last_edit_finished`` and ``now_fn`` are injected so
    the trigger is testable without an editor or a real clock.
    """

    def __init__(
        self,
        store: SelectionStore,
        publish_queue,
        limit: int,
        quiet_seconds: float = 1200.0,
        batch_cap: int = 0,
        enabled: bool = True,
        is_editing_busy=None,
        on_decision=None,
        client=None,
        model: Optional[str] = None,
        poll_seconds: float = _IDLE_POLL_SECONDS,
        now_fn=time.time,
        logger=print,
    ):
        self.store = store
        self.publish_queue = publish_queue
        self.limit = limit
        self.quiet_seconds = max(0.0, quiet_seconds)
        self.batch_cap = max(0, batch_cap)
        self.enabled = enabled
        self.is_editing_busy = is_editing_busy or (lambda: False)
        self._last_edit_finished_at: Optional[float] = None
        self.on_decision = on_decision
        self.client = client
        self.model = model
        self.poll_seconds = poll_seconds
        self.now_fn = now_fn
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- trigger inputs -------------------------------------------------

    def on_edit_finished(self) -> None:
        """Record that an edit just completed, restarting the quiet period."""
        self._last_edit_finished_at = self.now_fn()

    def prime_as_quiet(self) -> None:
        """Treat the quiet period as already elapsed.

        For restored pending candidates: their edits finished long ago, so a
        restart must not wait another full quiet period before selecting.
        """
        self._last_edit_finished_at = self.now_fn() - self.quiet_seconds

    def last_edit_finished(self) -> Optional[float]:
        return self._last_edit_finished_at

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="selection-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.maybe_select()
            except Exception as exc:  # never let the scheduler thread die
                self.logger(f"⚠️  Selection pass failed: {exc}")
            self._stop_event.wait(self.poll_seconds)

    # ---- the trigger ----------------------------------------------------

    def _should_select(self) -> bool:
        if not self.enabled:
            return False
        pending = self.store.pending()
        if not pending:
            return False
        if self.is_editing_busy():
            return False
        if self.batch_cap > 0 and len(pending) >= self.batch_cap:
            return True
        last = self.last_edit_finished()
        if last is None:
            return False
        return (self.now_fn() - last) >= self.quiet_seconds

    def maybe_select(self) -> bool:
        """Run a selection if the trigger fires. Returns whether it ran."""
        if not self._should_select():
            return False
        ranking = select_and_enqueue(
            self.store,
            self.publish_queue,
            self.limit,
            client=self.client,
            model=self.model,
            logger=self.logger,
        )
        if ranking is not None and self.on_decision:
            self.on_decision(ranking)
        return ranking is not None


# ---- dry-run CLI ------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """
    Print the ranking for existing output directories without queueing anything.

    The CLI is read-only by design: it ranks and prints, never writing the
    queue or state files. Useful for judging the ranking against clips that
    are already on disk before trusting it with a live batch::

        python3 -m social_media_publisher.clip_selection output/<dir>...

    Pass ``--fallback`` to rank with the deterministic offline ordering instead
    of calling the LLM.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="clip_selection",
        description="Rank finished clips and publish only the best N.",
    )
    parser.add_argument("directories", nargs="+", help="Output directories to rank.")
    parser.add_argument("--limit", type=int, default=6,
                        help="How many clips would be published (default: 6).")
    parser.add_argument("--source-link", default=None,
                        help="Source link recorded for each ranked clip.")
    parser.add_argument("--fallback", action="store_true",
                        help="Use the deterministic offline ordering (no LLM call).")
    parser.add_argument("--model", default=None,
                        help="Override the ranking model.")
    args = parser.parse_args(argv)

    candidates = []
    for raw in args.directories:
        path = Path(raw)
        if not path.is_dir():
            print(f"✗  '{raw}' is not a directory")
            return 1
        candidates.append(read_candidate(str(path), path.name, args.source_link))

    client = None
    if not args.fallback:
        try:
            client = _build_client()
        except Exception as exc:
            print(f"⚠️  No LLM client ({exc}); falling back to audio-energy ordering")

    ranking = rank_candidates(candidates, limit=args.limit, client=client, model=args.model)

    print(f"🏆 Ranking {len(candidates)} clip(s) — would publish the top "
          f"{args.limit} (source: {ranking.source})")
    if ranking.error:
        print(f"   note: {ranking.error}")
    print()
    for rank, item in enumerate(ranking.ranked, start=1):
        mark = "✅ would publish" if rank <= args.limit else "— not selected"
        score = f"{item.score:.0f}" if isinstance(item.score, (int, float)) else "-"
        candidate = item.candidate
        print(f"{rank:>2}. {mark}  {candidate.video_name}  [score {score}]")
        print(f"        title: {candidate.title or '(none)'}")
        print(f"        hook:  {candidate.hook_text or '(none)'}")
        print(f"        why:   {item.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
