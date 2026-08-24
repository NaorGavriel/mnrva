import asyncio
import time

import pytest

from rate_limiter import RateLimiter


def test_init_rejects_non_positive_budgets() -> None:
    """A zero or negative budget can never be serviced, so it's rejected at construction."""
    with pytest.raises(ValueError):
        RateLimiter(max_requests_per_minute=0, max_tokens_per_minute=100)
    with pytest.raises(ValueError):
        RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=0)


def test_refill_tops_up_both_buckets_proportional_to_elapsed_time() -> None:
    """After 1 second at 60 req/min and 6000 tok/min, ~1 request and ~100 tokens refill."""
    limiter = RateLimiter(max_requests_per_minute=60, max_tokens_per_minute=6000)
    limiter._available_requests = 0.0
    limiter._available_tokens = 0.0
    limiter._last_refill = time.monotonic() - 1.0

    limiter._refill()

    assert limiter._available_requests == pytest.approx(1.0, abs=0.05)
    assert limiter._available_tokens == pytest.approx(100.0, abs=5)


def test_refill_caps_at_the_configured_max() -> None:
    """A long-idle limiter refills to its max, not beyond it."""
    limiter = RateLimiter(max_requests_per_minute=60, max_tokens_per_minute=6000)
    limiter._last_refill = time.monotonic() - 3600

    limiter._refill()

    assert limiter._available_requests == 60
    assert limiter._available_tokens == 6000


async def test_acquire_deducts_capacity_when_already_available() -> None:
    """A call within budget succeeds immediately and debits both buckets."""
    limiter = RateLimiter(max_requests_per_minute=60, max_tokens_per_minute=6000)

    await limiter.acquire(estimated_tokens=100)

    assert limiter._available_requests == pytest.approx(59, abs=0.5)
    assert limiter._available_tokens == pytest.approx(5900, abs=5)


async def test_acquire_raises_when_estimated_tokens_exceeds_the_budget() -> None:
    """A single call larger than the whole per-minute token budget can never be served."""
    limiter = RateLimiter(max_requests_per_minute=60, max_tokens_per_minute=100)

    with pytest.raises(ValueError):
        await limiter.acquire(estimated_tokens=101)


async def test_acquire_blocks_until_the_request_bucket_refills() -> None:
    """An empty request bucket forces acquire to wait for a refill, not return instantly."""
    limiter = RateLimiter(max_requests_per_minute=6000, max_tokens_per_minute=10_000_000)
    limiter._available_requests = 0.0

    start = time.monotonic()
    await limiter.acquire(estimated_tokens=1)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.005


async def test_acquire_serves_concurrent_callers_in_arrival_order() -> None:
    """Three callers blocked on the same empty bucket complete in the order they called acquire."""
    limiter = RateLimiter(max_requests_per_minute=6000, max_tokens_per_minute=10_000_000)
    limiter._available_requests = 0.0
    completion_order: list[str] = []

    async def acquire_and_record(name: str) -> None:
        await limiter.acquire(estimated_tokens=1)
        completion_order.append(name)

    await asyncio.gather(
        acquire_and_record("a"), acquire_and_record("b"), acquire_and_record("c")
    )

    assert completion_order == ["a", "b", "c"]
