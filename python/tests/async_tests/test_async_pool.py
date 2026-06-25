# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests for Feature 1: Client-Instance Pooling (Python async).
Mirrors sync test_sync_pool.py and Java ClientPoolIntegrationTest for
cross-language parity.

Requires a running Valkey server (standalone).
"""

import asyncio
import uuid

import pytest

from glide import (
    AsyncClientPool,
    GlideClientConfiguration,
    NodeAddress,
    PoolConfig,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def pool_config():
    """Pool config for a standalone server."""
    return GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
    )


class TestAsyncClientPool:
    """Async pool lifecycle tests."""

    async def test_pool_create_acquire_release(self, pool_config):
        """Create pool, acquire client, execute commands, release, close."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=3, min_idle=1))

        assert pool.idle_count >= 1

        async with pool.borrow() as client:
            key = f"async-pool-{uuid.uuid4().hex[:8]}"
            await client.set(key, "hello")
            val = await client.get(key)
            assert val == b"hello"
            await client.delete([key])

        pool.close()

    async def test_pool_reuse(self, pool_config):
        """LIFO: same client returned after release."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=3, min_idle=1))

        client1 = await pool.acquire()
        id1 = id(client1)
        pool.release(client1)
        await asyncio.sleep(0.05)

        client2 = await pool.acquire()
        id2 = id(client2)
        pool.release(client2)

        assert id1 == id2
        pool.close()

    async def test_pool_metrics(self, pool_config):
        """Metrics reflect pool state."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=3, min_idle=2))

        assert pool.idle_count >= 1
        assert pool.total_count >= 1

        pool.close()

    async def test_pool_exhaustion_timeout(self, pool_config):
        """Timeout when pool is exhausted."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=1, min_idle=1))

        # Acquire the only client
        client = await pool.acquire()

        # Second acquire should timeout
        with pytest.raises(TimeoutError):
            await pool.acquire(timeout=0.5)

        pool.release(client)
        pool.close()

    async def test_pool_concurrent_access(self, pool_config):
        """Multiple tasks borrow/release concurrently."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=4, min_idle=4))

        errors = []

        async def worker(task_idx):
            try:
                async with pool.borrow() as client:
                    key = f"async-pool-concurrent-{task_idx}-{uuid.uuid4().hex[:6]}"
                    await client.set(key, f"task-{task_idx}")
                    val = await client.get(key)
                    assert val == f"task-{task_idx}".encode()
                    await client.delete([key])
            except Exception as e:
                errors.append(e)

        tasks = [asyncio.create_task(worker(i)) for i in range(8)]
        await asyncio.gather(*tasks)

        assert not errors, f"Worker errors: {errors}"
        pool.close()

    async def test_pool_close_rejects_acquire(self, pool_config):
        """Closed pool rejects acquire."""
        pool = await AsyncClientPool.create(pool_config, PoolConfig(max_size=2, min_idle=1))

        pool.close()

        with pytest.raises(RuntimeError, match="closed"):
            await pool.acquire()
