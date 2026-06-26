# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, request timeout,
inflight limits) are properly inherited and respected by scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Requires a running Valkey server (standalone) on localhost:6379.
"""

import uuid

import pytest

from glide_sync import (
    CompressionBackend,
    CompressionConfiguration,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def compressed_client():
    """Create a GlideClient with ZSTD compression enabled."""
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
    c = GlideClient.create(config)
    yield c
    c.close()


@pytest.fixture
def raw_client():
    """Create a GlideClient WITHOUT compression (for verification)."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
    )
    c = GlideClient.create(config)
    yield c
    c.close()


@pytest.fixture
def short_timeout_client():
    """Create a GlideClient with a short request timeout."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=100,  # 100ms
    )
    c = GlideClient.create(config)
    yield c
    c.close()


@pytest.fixture
def inflight_limited_client():
    """Create a GlideClient with explicit inflight limit."""
    config = GlideClientConfiguration(
        addresses=[NodeAddress("localhost", 6379)],
        request_timeout=5000,
        inflight_requests_limit=500,
    )
    c = GlideClient.create(config)
    yield c
    c.close()


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
        client = GlideClient.create(config)

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
        client.close()

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
        client_a = GlideClient.create(config_a)
        client_b = GlideClient.create(config_b)

        key = f"dual-timeout-{uuid.uuid4().hex[:8]}"

        # Both scopes should work for fast operations
        with client_a.scoped_connection() as scope_a:
            scope_a.set(key, "from-A")

        with client_b.scoped_connection() as scope_b:
            val = scope_b.get(key)
            assert val == "from-A"

        # Cleanup
        client_a.delete([key])
        client_a.close()
        client_b.close()


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
        client = GlideClient.create(config)

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
        raw_client = GlideClient.create(raw_config)
        raw_value = raw_client.get(key)
        assert raw_value != large_value.encode(), (
            "Data should be stored compressed in Valkey"
        )

        # Cleanup
        client.delete([key])
        raw_client.close()
        client.close()
