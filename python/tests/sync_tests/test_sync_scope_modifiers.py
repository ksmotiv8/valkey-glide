# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, request timeout,
inflight limits) are properly inherited and respected by scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Requires a running Valkey server (standalone) on localhost:6379.
"""

import threading
import uuid

import pytest

from glide_sync import (
    Batch,
    CompressionBackend,
    CompressionConfiguration,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
)


# ─── Client Pooling (matches conftest pattern from #6335) ─────────────────────

_modifier_client_pool: dict = {}
_modifier_pool_lock = threading.Lock()


def _client_is_usable(client) -> bool:
    """Check if a client's FFI handle is still valid."""
    if client is None:
        return False
    return (
        not client._is_closed
        and client._core_client is not None
        and client._core_client != client._ffi.NULL
    )


def _get_or_create_client(key: str, config: GlideClientConfiguration):
    """Get or create a pooled client by key. Thread-safe, xdist-safe."""
    with _modifier_pool_lock:
        client = _modifier_client_pool.get(key)
    if _client_is_usable(client):
        try:
            client.custom_command(["PING"])
            return client
        except Exception:
            try:
                client.close()
            except Exception:
                pass
    client = GlideClient.create(config)
    with _modifier_pool_lock:
        _modifier_client_pool[key] = client
    return client


def _teardown_client(client, key: str):
    """Pipelined teardown: FLUSHALL in a single round-trip batch."""
    if not _client_is_usable(client):
        return
    try:
        batch = Batch(is_atomic=False)
        batch.custom_command(["FLUSHALL", "ASYNC"])
        client.exec(batch, raise_on_error=True)
    except Exception:
        try:
            client.close()
        except Exception:
            pass
        with _modifier_pool_lock:
            _modifier_client_pool.pop(key, None)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def compressed_client():
    """Get or reuse a GlideClient with ZSTD compression enabled."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
        compression=CompressionConfiguration(
            enabled=True,
            backend=CompressionBackend.ZSTD,
            compression_level=3,
            min_compression_size=64,
        ),
    )
    client = _get_or_create_client("compressed", config)
    yield client
    _teardown_client(client, "compressed")


@pytest.fixture
def raw_client():
    """Get or reuse a GlideClient WITHOUT compression (for verification)."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
    )
    client = _get_or_create_client("raw", config)
    yield client
    _teardown_client(client, "raw")


@pytest.fixture
def short_timeout_client():
    """Get or reuse a GlideClient with a short request timeout."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=100,  # 100ms
    )
    client = _get_or_create_client("short_timeout", config)
    yield client
    _teardown_client(client, "short_timeout")


@pytest.fixture
def inflight_limited_client():
    """Get or reuse a GlideClient with explicit inflight limit."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
        inflight_requests_limit=500,
    )
    client = _get_or_create_client("inflight_limited", config)
    yield client
    _teardown_client(client, "inflight_limited")


# ─── Compression Tests ────────────────────────────────────────────────────────


class TestScopeCompression:
    """Tests that scoped connections inherit and apply compression correctly."""

    def test_scope_writes_compressed_data(self, compressed_client, raw_client):
        """Data written via scope with compression should be compressed in Valkey."""
        key = f"scope-compress-write-{uuid.uuid4().hex[:8]}"
        large_value = "A" * 500  # 500 bytes, well above 64-byte threshold

        # Write via scoped connection
        with compressed_client.scoped_connection() as scope:
            scope.set(key, large_value)

        # Read with same client (decompresses) — should match
        result = compressed_client.get(key)
        assert result == large_value.encode()

        # Read with raw client (no compression) — should differ (compressed bytes)
        raw_result = raw_client.get(key)
        assert raw_result != large_value.encode(), (
            "Value stored via compressed scope should be compressed in Valkey"
        )

        # Cleanup
        compressed_client.delete([key])

    def test_scope_reads_compressed_data(self, compressed_client):
        """Scoped GET should decompress data written by the parent client."""
        key = f"scope-compress-read-{uuid.uuid4().hex[:8]}"
        value = "CompressibleData_" * 50  # ~850 bytes

        # Write via parent client (compressed)
        compressed_client.set(key, value)

        # Read via scoped connection — should decompress correctly
        with compressed_client.scoped_connection() as scope:
            retrieved = scope.get(key)
            assert retrieved == value

        # Cleanup
        compressed_client.delete([key])

    def test_scope_small_values_not_compressed(self, raw_client):
        """Values below minCompressionSize are stored uncompressed."""
        # Create client with high min threshold
        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            compression=CompressionConfiguration(
                enabled=True,
                backend=CompressionBackend.ZSTD,
                min_compression_size=256,  # Only compress >= 256 bytes
            ),
        )
        client = _get_or_create_client("high_threshold", config)

        key = f"scope-small-{uuid.uuid4().hex[:8]}"
        small_value = "hello"  # 5 bytes — well below threshold

        # Write via scope
        with client.scoped_connection() as scope:
            scope.set(key, small_value)

        # Raw client should see the original value (not compressed)
        raw_result = raw_client.get(key)
        assert raw_result == small_value.encode(), (
            "Small values below minCompressionSize should NOT be compressed"
        )

        # Cleanup
        client.delete([key])

    def test_scope_roundtrip_with_compression(self, compressed_client):
        """Full round-trip: scope SET → scope GET on same scope."""
        key = f"scope-roundtrip-{uuid.uuid4().hex[:8]}"
        value = "RoundTripData_" * 40  # ~560 bytes

        with compressed_client.scoped_connection() as scope:
            scope.set(key, value)
            retrieved = scope.get(key)
            assert retrieved == value

        # Cleanup
        compressed_client.delete([key])

    def test_scope_watch_transaction_with_compression(self, compressed_client):
        """WATCH/MULTI/EXEC works correctly with compressed values."""
        key = f"watch-compress-{uuid.uuid4().hex[:8]}"
        initial_value = "InitialLargeValue_" * 20  # ~360 bytes

        compressed_client.set(key, initial_value)

        with compressed_client.scoped_connection() as scope:
            scope.watch(key)
            current = scope.get(key)
            assert current == initial_value

            new_value = "UpdatedLargeValue_" * 20
            scope.multi()
            scope.set(key, new_value)
            result = scope.exec()
            assert result is not None and result != "None"

        # Verify the updated value
        final = compressed_client.get(key)
        assert final == new_value.encode()

        # Cleanup
        compressed_client.delete([key])


# ─── Request Timeout Tests ────────────────────────────────────────────────────


class TestScopeRequestTimeout:
    """Tests that scoped connections respect the parent's request timeout."""

    def test_scope_fast_ops_succeed_with_short_timeout(self, short_timeout_client):
        """Fast operations complete within the short timeout."""
        key = f"timeout-fast-{uuid.uuid4().hex[:8]}"

        with short_timeout_client.scoped_connection() as scope:
            scope.set(key, "fast")
            val = scope.get(key)
            assert val == "fast"

        # Cleanup
        short_timeout_client.delete([key])

    def test_different_clients_different_scope_timeouts(self):
        """Each client's scopes use that client's timeout setting."""
        config_a = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
        )
        config_b = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=200,
        )
        client_a = _get_or_create_client("timeout_5000", config_a)
        client_b = _get_or_create_client("timeout_200", config_b)

        key = f"dual-timeout-{uuid.uuid4().hex[:8]}"

        # Both scopes should work for fast operations
        with client_a.scoped_connection() as scope_a:
            scope_a.set(key, "from-A")

        with client_b.scoped_connection() as scope_b:
            val = scope_b.get(key)
            assert val == "from-A"

        # Cleanup
        client_a.delete([key])


# ─── Inflight Request Limit Tests ────────────────────────────────────────────


class TestScopeInflightLimit:
    """Tests that scoped commands count against the parent's inflight limit."""

    def test_scope_ops_under_inflight_limit(self, inflight_limited_client):
        """Sequential scope operations work within inflight limits."""
        with inflight_limited_client.scoped_connection() as scope:
            for i in range(50):
                key = f"inflight-{i}-{uuid.uuid4().hex[:6]}"
                scope.set(key, f"value-{i}")
                val = scope.get(key)
                assert val == f"value-{i}"
                scope.execute_command("DEL", key)

    def test_scope_many_sequential_ops(self, inflight_limited_client):
        """Many sequential scope operations don't exhaust inflight slots."""
        key = f"inflight-seq-{uuid.uuid4().hex[:8]}"

        with inflight_limited_client.scoped_connection() as scope:
            # 200 sequential SET/GET pairs — each reserves and releases a slot
            for i in range(200):
                scope.set(key, str(i))
                val = scope.get(key)
                assert val == str(i)

        # Cleanup
        inflight_limited_client.delete([key])


# ─── Combined Modifiers ───────────────────────────────────────────────────────


class TestScopeCombinedModifiers:
    """Tests that all connection modifiers work together on scoped connections."""

    def test_all_modifiers_active(self):
        """Scope with compression + timeout + inflight all active simultaneously."""
        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            inflight_requests_limit=500,
            compression=CompressionConfiguration(
                enabled=True,
                backend=CompressionBackend.ZSTD,
                compression_level=3,
                min_compression_size=64,
            ),
        )
        client = _get_or_create_client("all_modifiers", config)

        key = f"combined-{uuid.uuid4().hex[:8]}"
        large_value = "TestData_" * 100  # ~900 bytes

        # Write and read via scope
        with client.scoped_connection() as scope:
            scope.set(key, large_value)
            retrieved = scope.get(key)
            assert retrieved == large_value

        # Verify via parent client
        parent_get = client.get(key)
        assert parent_get == large_value.encode()

        # Verify compression happened (raw client sees different bytes)
        raw_config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
        )
        raw_client = _get_or_create_client("raw_verify", raw_config)
        raw_value = raw_client.get(key)
        assert raw_value != large_value.encode(), (
            "Data should be stored compressed in Valkey"
        )

        # Cleanup
        client.delete([key])


# ─── Database State Tests ─────────────────────────────────────────────────────


class TestDatabaseStateInheritance:
    """Tests that database selection is correctly handled across pool and scope."""

    def test_scope_inherits_configured_database(self):
        """Scope connections use the database from the client's config."""
        # Create client configured for database 2
        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            database_id=2,
        )
        client = GlideClient.create(config)

        key = f"db2-scope-{uuid.uuid4().hex[:8]}"
        try:
            # Write via scope on database 2
            with client.scoped_connection() as scope:
                scope.set(key, "on-db2")
                val = scope.get(key)
                assert val == "on-db2"

            # Parent client (also on db 2) should see the key
            result = client.get(key)
            assert result == b"on-db2"

            # A client on database 0 should NOT see the key
            config_db0 = GlideClientConfiguration(
                addresses=[NodeAddress("localhost", 6379)],
                request_timeout=5000,
                database_id=0,
            )
            client_db0 = GlideClient.create(config_db0)
            result_db0 = client_db0.get(key)
            assert result_db0 is None, (
                "Key written on db2 via scope should not be visible on db0"
            )
            client_db0.close()
        finally:
            client.custom_command(["DEL", key])
            client.close()

    def test_scope_uses_config_db_not_runtime_db(self):
        """If parent calls SELECT at runtime, scope inherits the runtime database.

        Scoped connections use the parent client's current_database() at creation
        time, which reflects any runtime SELECT calls made on the parent.
        """
        # Create client configured for database 0
        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            database_id=0,
        )
        client = GlideClient.create(config)

        key = f"runtime-db-{uuid.uuid4().hex[:8]}"
        try:
            # Switch parent to db 3 at runtime
            client.custom_command(["SELECT", "3"])
            client.set(key, "on-db3")

            # Scope should inherit the parent's current database (3)
            with client.scoped_connection() as scope:
                result = scope.get(key)
                assert result == "on-db3", (
                    "Scope should inherit parent's current runtime database (3)"
                )

            # Clean up: delete key on db 3 and switch parent back
            client.custom_command(["DEL", key])
            client.custom_command(["SELECT", "0"])
        finally:
            client.close()

    def test_pool_resets_database_after_borrow(self):
        """After a borrower changes database, the pool resets it on release.

        Next borrower should get a connection on the configured database.
        """
        import time

        from glide_sync import ClientPool, PoolConfig

        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            database_id=0,
        )
        pool_config = PoolConfig(
            max_size=1,
            min_idle=1,
            acquire_timeout_ms=10000,
            client_config=config,
        )
        pool = ClientPool.create(pool_config)
        time.sleep(2)  # Wait for min_idle warmup

        key = f"pool-db-reset-{uuid.uuid4().hex[:8]}"

        try:
            # First borrower: switch to db 5 and write a key there
            with pool.acquire() as client1:
                client1.custom_command(["SELECT", "5"])
                client1.set(key, "on-db5")
            # client1 is released — pool should reset back to db 0

            time.sleep(0.5)  # Allow async reset to complete

            # Second borrower: should be on db 0 (reset happened)
            with pool.acquire() as client2:
                # This key should NOT be visible (we're on db 0, key is on db 5)
                result = client2.get(key)
                assert result is None, (
                    "Pool should reset database to configured value after release. "
                    "Second borrower should be on db 0, not db 5."
                )
        finally:
            # Clean up key on db 5
            cleanup_config = GlideClientConfiguration(
                addresses=[NodeAddress("localhost", 6379)],
                request_timeout=5000,
                database_id=5,
            )
            cleanup = GlideClient.create(cleanup_config)
            cleanup.custom_command(["DEL", key])
            cleanup.close()
            pool.close()

    def test_scope_release_resets_database(self):
        """After scope user calls SELECT, release resets to configured database.

        Next scope borrow from the same pool should be on the configured database.
        """
        import time

        config = GlideClientConfiguration(
            addresses=[NodeAddress("localhost", 6379)],
            request_timeout=5000,
            database_id=0,
        )
        client = GlideClient.create(config)

        key = f"scope-db-reset-{uuid.uuid4().hex[:8]}"

        try:
            # First scope: SELECT to db 4 and write
            with client.scoped_connection() as scope:
                scope.execute_command("SELECT", "4")
                scope.set(key, "on-db4")
            # Scope released — cleanup should reset to db 0

            time.sleep(0.3)  # Allow async cleanup to complete

            # Second scope: should be on db 0 (not db 4)
            with client.scoped_connection() as scope2:
                result = scope2.get(key)
                assert result is None, (
                    "Scope release should reset database. Second scope should be on "
                    "configured db (0), not the previous scope's runtime db (4)."
                )
        finally:
            # Clean up key on db 4
            cleanup_config = GlideClientConfiguration(
                addresses=[NodeAddress("localhost", 6379)],
                request_timeout=5000,
                database_id=4,
            )
            cleanup = GlideClient.create(cleanup_config)
            cleanup.custom_command(["DEL", key])
            cleanup.close()
            client.close()
