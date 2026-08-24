import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

MAX_REQUESTS_PER_MINUTE = int(os.environ["OPENAI_MAX_REQUESTS_PER_MINUTE"])
MAX_TOKENS_PER_MINUTE = int(os.environ["OPENAI_MAX_TOKENS_PER_MINUTE"])


class RateLimiter:
    """Paces outbound OpenAI calls against two shared per-minute budgets
    (requests and tokens), so every caller that acquires it draws from the
    same account-level throttle instead of tracking its own.

    One concern: pacing calls. Retries/backoff on 429s stay with the
    `openai` SDK's own `max_retries`, not this class.
    """

    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int) -> None:
        """Start both buckets full, sized at `max_requests_per_minute`/`max_tokens_per_minute`."""
        if max_requests_per_minute <= 0 or max_tokens_per_minute <= 0:
            raise ValueError("max_requests_per_minute and max_tokens_per_minute must be positive")
        self._max_requests_per_minute = max_requests_per_minute
        self._max_tokens_per_minute = max_tokens_per_minute
        self._available_requests = float(max_requests_per_minute)
        self._available_tokens = float(max_tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until both buckets have capacity for one call costing
        `estimated_tokens`, then deduct that capacity.

        The lock is held across any waiting (not just the final check), so
        concurrent callers are serviced in the order they called `acquire`.
        """
        if estimated_tokens > self._max_tokens_per_minute:
            raise ValueError(
                f"estimated_tokens={estimated_tokens} exceeds the "
                f"{self._max_tokens_per_minute}/minute token budget; this call can never be served"
            )
        async with self._lock:
            while True:
                self._refill()
                if self._available_requests >= 1 and self._available_tokens >= estimated_tokens:
                    self._available_requests -= 1
                    self._available_tokens -= estimated_tokens
                    return
                await asyncio.sleep(self._seconds_until_available(estimated_tokens))

    def _refill(self) -> None:
        """Top up both buckets for elapsed time since the last refill, capped at their max."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._available_requests = min(
            self._max_requests_per_minute,
            self._available_requests + elapsed * self._max_requests_per_minute / 60,
        )
        self._available_tokens = min(
            self._max_tokens_per_minute,
            self._available_tokens + elapsed * self._max_tokens_per_minute / 60,
        )

    def _seconds_until_available(self, estimated_tokens: int) -> float:
        """How long to wait, from now, for both buckets to cover one more
        request plus `estimated_tokens`."""
        request_deficit = max(0.0, 1 - self._available_requests)
        token_deficit = max(0.0, estimated_tokens - self._available_tokens)
        request_wait = request_deficit * 60 / self._max_requests_per_minute
        token_wait = token_deficit * 60 / self._max_tokens_per_minute
        return max(request_wait, token_wait)


rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
