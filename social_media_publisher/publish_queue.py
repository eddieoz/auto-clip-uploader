"""
Disk-backed publish queue and drip scheduler.

Videos are edited up front and appended to a queue; a scheduler thread publishes
them one at a time, spaced by a configurable interval. State persists to a JSON
file so the queue survives process restarts.
"""

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


# Status values used throughout the queue lifecycle.
STATUS_PENDING = "pending"
STATUS_PUBLISHING = "publishing"
STATUS_PUBLISHED = "published"
STATUS_FAILED = "failed"

# Polling cadence when the scheduler has nothing to publish.
_IDLE_POLL_SECONDS = 5.0


@dataclass
class QueueEntry:
    """A single video waiting to be (or already) published."""

    id: str
    output_dir: str
    video_name: str
    source_link: Optional[str] = None
    queued_at: float = field(default_factory=time.time)
    status: str = STATUS_PENDING
    attempts: int = 0
    published_at: Optional[float] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "QueueEntry":
        # Tolerate missing fields from older state files.
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            output_dir=data["output_dir"],
            video_name=data.get("video_name", Path(data["output_dir"]).name),
            source_link=data.get("source_link"),
            queued_at=data.get("queued_at", time.time()),
            status=data.get("status", STATUS_PENDING),
            attempts=data.get("attempts", 0),
            published_at=data.get("published_at"),
            last_error=data.get("last_error"),
        )


class PublishQueue:
    """
    Thread-safe, disk-backed FIFO queue of videos to publish.

    The queue and the last-publish timestamp are persisted to ``state_path``
    (a JSON file) via atomic temp-file writes, so a crash between writes never
    leaves a truncated state file.
    """

    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self._lock = threading.RLock()
        self._entries: List[QueueEntry] = []
        self._last_publish_time: Optional[float] = None
        self._load()

    # ---- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt state file should not crash the whole pipeline.
            print(f"⚠️  Could not load publish queue state from {self.state_path}; starting fresh.")
            return

        self._entries = [QueueEntry.from_dict(e) for e in data.get("entries", [])]
        self._last_publish_time = data.get("last_publish_time")

        # Crash recovery: anything marked "publishing" was interrupted mid-flight.
        recovered = 0
        for entry in self._entries:
            if entry.status == STATUS_PUBLISHING:
                entry.status = STATUS_PENDING
                recovered += 1
        if recovered:
            print(f"♻️  Recovered {recovered} interrupted publish(s) back to pending.")
        self._save()

    def _save(self) -> None:
        """Atomically write state. Caller must hold ``self._lock``."""
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "last_publish_time": self._last_publish_time,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then rename — atomic on POSIX.
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

    def enqueue(self, output_dir: str, video_name: str, source_link: Optional[str] = None) -> QueueEntry:
        """Add a video to the back of the queue."""
        entry = QueueEntry(
            id=str(uuid.uuid4()),
            output_dir=str(output_dir),
            video_name=video_name,
            source_link=source_link,
        )
        with self._lock:
            self._entries.append(entry)
            self._save()
        return entry

    def next_pending(self) -> Optional[QueueEntry]:
        """Return the next pending entry (FIFO), or None if the queue is drained."""
        with self._lock:
            for entry in self._entries:
                if entry.status == STATUS_PENDING:
                    return entry
            return None

    def mark_publishing(self, entry: QueueEntry) -> None:
        with self._lock:
            entry.status = STATUS_PUBLISHING
            self._save()

    def mark_published(self, entry: QueueEntry) -> None:
        with self._lock:
            entry.status = STATUS_PUBLISHED
            entry.published_at = time.time()
            self._last_publish_time = entry.published_at
            self._save()

    def mark_failed(self, entry: QueueEntry, error: str, max_retries: int) -> None:
        """Record a failed attempt. Re-queues for retry until attempts exceed max_retries."""
        with self._lock:
            entry.attempts += 1
            entry.last_error = error
            if entry.attempts >= max_retries:
                entry.status = STATUS_FAILED
            else:
                entry.status = STATUS_PENDING  # eligible for another attempt
            self._save()

    def get_last_publish_time(self) -> Optional[float]:
        with self._lock:
            return self._last_publish_time

    def stats(self) -> Dict[str, int]:
        """Counts by status, for the dashboard."""
        with self._lock:
            counts = {"pending": 0, "publishing": 0, "published": 0, "failed": 0}
            for entry in self._entries:
                counts[entry.status] = counts.get(entry.status, 0) + 1
            counts["total"] = len(self._entries)
            return counts


class PublishScheduler:
    """
    Daemon worker that drains a :class:`PublishQueue` on a fixed interval.

    Spacing is measured from each publish's start: posts fire at T=0, T=interval,
    T=2*interval, ... If the queue was idle for longer than the interval, the
    next entry publishes immediately (``last_publish_time`` is never reset).
    """

    def __init__(
        self,
        queue: PublishQueue,
        interval_minutes: float = 0.0,
        max_retries: int = 3,
        on_publish: Optional[Callable[[QueueEntry, Dict], None]] = None,
    ):
        self.queue = queue
        self.interval_seconds = max(0.0, interval_minutes * 60.0)
        self.max_retries = max(1, max_retries)
        self.on_publish = on_publish
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="publish-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            entry = self.queue.next_pending()
            if entry is None:
                self._stop_event.wait(_IDLE_POLL_SECONDS)
                continue

            # Honor the configured interval before publishing.
            self._wait_for_interval()

            if self._stop_event.is_set():
                break

            self._publish_one(entry)

    def _wait_for_interval(self) -> None:
        """Block until enough time has elapsed since the last publish."""
        if self.interval_seconds <= 0:
            return
        last = self.queue.get_last_publish_time()
        if last is None:
            return  # first publish fires immediately
        elapsed = time.time() - last
        remaining = self.interval_seconds - elapsed
        if remaining <= 0:
            return
        # Interruptible sleep — wakes early on stop().
        self._stop_event.wait(remaining)

    def _publish_one(self, entry: QueueEntry) -> None:
        """Publish a single entry, then record the outcome."""
        self.queue.mark_publishing(entry)
        try:
            # Lazy import keeps unit tests from pulling in the heavy publisher stack.
            from .publisher import PostizPublisher

            publisher = PostizPublisher(entry.output_dir, source_link=entry.source_link)
            result = publisher.publish()

            if result and result.get("success"):
                self.queue.mark_published(entry)
            elif result and result.get("error") == "rate_limited":
                # Server-side rate limit — do not consume a retry; retry shortly.
                entry.status = STATUS_PENDING
                with self.queue._lock:
                    self.queue._save()
            else:
                error = (result or {}).get("error", "unknown_error")
                message = (result or {}).get("message", str(result))
                self.queue.mark_failed(entry, f"{error}: {message}", self.max_retries)

            if self.on_publish:
                self.on_publish(entry, result or {})
        except Exception as e:  # never let the scheduler thread die
            self.queue.mark_failed(entry, repr(e), self.max_retries)
            if self.on_publish:
                try:
                    self.on_publish(entry, {"error": repr(e)})
                except Exception:
                    pass
