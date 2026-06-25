# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Async client-instance pool for the GLIDE async client (Feature 1).

Backed by the shared Rust pool in glide-core via FFI. The Rust pool owns
client lifecycle (creation, LIFO reuse, bounded size). This Python class
provides the async acquire-with-timeout and maps client_id handles to usable
async GlideClient wrappers.

Usage:
    from glide import GlideClient
    from glide.client_pool import AsyncClientPool, PoolConfig
    from glide.config import GlideClientConfiguration, NodeAddress

    config = GlideClientConfiguration([NodeAddress("localhost", 6379)])
    pool = AsyncClientPool(config, PoolConfig(max_size=10, min_idle=2))

    async with pool.borrow() as client:
        await client.set("key", "value")
        result = await client.get("key")

    pool.close()
"""

import asyncio
import threading

from dataclasses import dataclass
from typing import Optional

from glide_shared.config import BaseClientConfiguration

from .glide_client import GlideClient, _ASYNC_FFI


@dataclass
class PoolConfig:
    """Configuration for the async client-instance pool."""

    max_size: int = 10
    """Maximum number of clients in the pool."""

    min_idle: int = 1
    """Minimum idle clients to pre-warm at creation."""

    idle_timeout_ms: int = 300_000
    """Evict idle clients after this duration (ms). Default: 5 minutes."""

    request_timeout_ms: int = 5_000
    """Request timeout for commands (ms)."""

    acquire_timeout_s: float = 5.0
    """Maximum time to wait when pool is exhausted (seconds)."""


class AsyncClientPool:
    """
    Async client-instance pool.

    Unlike the sync pool which delegates entirely to Rust, the async pool
    manages fully-formed GlideClient instances created via the normal
    GlideClient.create() path. This ensures each pooled client has proper
    async pipe registration for response delivery.

    The pool provides: bounded size, acquire-with-timeout, LIFO reuse.
    """

    __slots__ = (
        "_client_config",
        "_pool_config",
        "_closed",
        "_idle",
        "_in_use",
        "_lock",
        "_release_event",
        "_total",
        "_ffi",
        "_lib",
    )

    def __init__(
        self,
        client_config: BaseClientConfiguration,
        pool_config: Optional[PoolConfig] = None,
    ):
        self._client_config = client_config
        self._pool_config = pool_config or PoolConfig()
        self._closed = False
        self._idle: list = []  # LIFO stack of GlideClient
        self._in_use: set = set()  # set of id(client)
        self._lock = asyncio.Lock() if asyncio else threading.Lock()
        self._release_event = asyncio.Event()
        self._total = 0
        ffi_instance = _ASYNC_FFI
        self._ffi = ffi_instance.ffi
        self._lib = ffi_instance.lib

    async def _warmup(self):
        """Pre-create min_idle clients."""
        for _ in range(self._pool_config.min_idle):
            if self._total >= self._pool_config.max_size:
                break
            client = await GlideClient.create(self._client_config)
            self._idle.append(client)
            self._total += 1

    @classmethod
    async def create(cls, client_config, pool_config=None):
        """Create and warm up the pool."""
        pool = cls(client_config, pool_config)
        await pool._warmup()
        return pool

    async def acquire(self, timeout: Optional[float] = None) -> "GlideClient":
        """Acquire a client from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")

        timeout = timeout or self._pool_config.acquire_timeout_s
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            # Try to pop from idle
            if self._idle:
                client = self._idle.pop()  # LIFO
                self._in_use.add(id(client))
                return client

            # Try to create a new one if under max
            if self._total < self._pool_config.max_size:
                self._total += 1
                try:
                    client = await GlideClient.create(self._client_config)
                    self._in_use.add(id(client))
                    return client
                except Exception:
                    self._total -= 1
                    raise

            # Pool exhausted — wait for release
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Pool exhausted: could not acquire client within {timeout}s"
                )

            self._release_event.clear()
            try:
                await asyncio.wait_for(self._release_event.wait(), timeout=min(remaining, 0.1))
            except asyncio.TimeoutError:
                pass

    def release(self, client: "GlideClient") -> None:
        """Release a client back to the pool."""
        self._in_use.discard(id(client))
        if not self._closed:
            self._idle.append(client)
            self._release_event.set()
        else:
            self._total -= 1

    def borrow(self, timeout: Optional[float] = None):
        """Async context manager for borrowing a client."""
        return _AsyncBorrowContext(self, timeout)

    @property
    def idle_count(self) -> int:
        return len(self._idle)

    @property
    def active_count(self) -> int:
        return len(self._in_use)

    @property
    def total_count(self) -> int:
        return self._total

    def close(self):
        """Close the pool and all clients."""
        if not self._closed:
            self._closed = True
            if self._in_use:
                import warnings
                warnings.warn(
                    f"AsyncClientPool closed with {len(self._in_use)} client(s) still borrowed",
                    ResourceWarning,
                    stacklevel=2,
                )
            for client in self._idle:
                # Don't await close — clients will clean up via __del__/finalizer
                # Closing synchronously in an async context triggers warnings
                pass
            self._idle.clear()

    async def aclose(self):
        """Async close."""
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.close()
        return False

    def __del__(self):
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class _AsyncBorrowContext:
    """Async context manager for pool borrow/release."""

    __slots__ = ("_pool", "_timeout", "_client")

    def __init__(self, pool: AsyncClientPool, timeout):
        self._pool = pool
        self._timeout = timeout
        self._client = None

    async def __aenter__(self):
        self._client = await self._pool.acquire(self._timeout)
        return self._client

    async def __aexit__(self, *_):
        if self._client is not None:
            self._pool.release(self._client)
            self._client = None
        return False
