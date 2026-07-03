#!/usr/bin/env python3
"""
Tests for the module-level batch rate limiter in social_media_publisher.publisher.

Covers:
- permits granted immediately while under the limit
- callers block until cooldown elapses, then proceed
- reconfiguration mid-session
"""

import sys
import threading
import time
from pathlib import Path

# Allow running standalone: `python tests/test_rate_limiter.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def _new_limiter():
    from social_media_publisher.publisher import _BatchRateLimiter
    rl = _BatchRateLimiter()
    rl.configure(3, 1.0)  # 3 permits, 1s cooldown — keep tests fast
    return rl


def test_under_limit_acquires_immediately():
    rl = _new_limiter()
    t0 = time.monotonic()
    rl.acquire(2)
    assert time.monotonic() - t0 < 0.05


def test_blocks_until_cooldown_then_proceeds():
    rl = _new_limiter()  # limit 3, cooldown 1s
    results = []
    done = []

    def worker(n):
        t0 = time.monotonic()
        rl.acquire(1)
        results.append((n, round(time.monotonic() - t0, 2)))
        done.append(n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(done) == 5, "all workers must eventually acquire"
    waiters = sorted(r[1] for r in results)
    # first 3 immediate, last 2 wait ~1s cooldown
    assert all(w < 0.1 for w in waiters[:3])
    assert all(w >= 0.9 for w in waiters[3:])


def test_reconfigure_changes_limit():
    rl = _new_limiter()
    rl.configure(10, 60.0)
    assert rl._limit == 10 and rl._cooldown == 60.0
    t0 = time.monotonic()
    rl.acquire(10)
    assert time.monotonic() - t0 < 0.05


if __name__ == "__main__":
    # ponytail: runnable self-check, no pytest needed
    test_under_limit_acquires_immediately()
    test_blocks_until_cooldown_then_proceeds()
    test_reconfigure_changes_limit()
    print("rate limiter self-check: OK")
