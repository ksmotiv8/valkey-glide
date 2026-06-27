# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, database selection)
are properly inherited and respected by async scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Tests run in both standalone and cluster modes where applicable.

Requires a running Valkey server (standalone, and optionally cluster).

Run with:
    PYTHONPATH="glide-async/python:glide-shared:$PYTHONPATH" \
    pytest tests/async_tests/test_async_scope_modifiers.py --noconftest -v
"""

import asyncio
import uuid

import pytest
import pytest_asyncio
from glide import (
    CompressionBackend,
    CompressionConfiguration,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    InfoSection,
    NodeAddress,
)
from packaging import version

from tests.utils.utils import get_cluster_addresses as _get_cluster_addresses
from tests.utils.utils import get_standalone_address as _get_standalone_address

pytestmark = pytest.mark.asyncio


async def _get_server_version(client) -> str:
    """Get server version string from a connected client."""
    info_str = await client.info([InfoSection.SERVER])
    for line in info_str.split("\n"):
        if line.startswith("valkey_version:") or line.startswith("redis_version:"):
            return line.split(":")[1].strip()
    return "0.0.0"


def _skip_cluster_if_unavailable():
    """Skip test if no cluster endpoints are configured."""
    try:
        cluster = pytest.valkey_cluster  # type: ignore[attr-defined]
        if cluster is None or len(cluster.nodes_addr) == 0:
            pytest.skip("No cluster endpoints available")
    except AttributeError:
        pytest.skip("No cluster endpoints available (pytest.valkey_cluster not set)")


# ─── Fixtures (standalone, --noconftest compatible) ───────────────────────────


@pytest_asyncio.fixture
async def compressed_client():
    """Create an async GlideClient with ZSTD compression enabled."""
    config = GlideClientConfiguration(
        addresses=[_get_standalone_address()],
        request_timeout=5000,
        compression=CompressionConfiguration(
            enabled=True,
            backend=CompressionBackend.ZSTD,
            compression_level=3,
            min_compression_size=64,
        ),
    )
    client = await GlideClient.create(config)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def raw_client():
    """Create an async GlideClient WITHOUT compression (for verification)."""
    config = GlideClientConfiguration(
        addresses=[_get_standalone_address()],
        request_timeout=5000,
    )
    client = await GlideClient.create(config)
    yield client
    await client.aclose()


# ─── Compression Tests ────────────────────────────────────────────────────────


class TestAsyncScopeCompression:
    """Tests that async scoped connections inherit and apply compression correctly."""

    async def test_scope_writes_compressed_data(self, compressed_client, raw_client):
        """Data written via scope with compression should be compressed in Valkey."""
        key = f"async-scope-compress-write-{uuid.uuid4().hex[:8]}"
        large_value = "A" * 500  # 500 bytes, well above 64-byte threshold

        # Write via scoped connection
        async with await compressed_client.scoped_connection() as scope:
            await scope.set(key, large_value)

        # Read with same client (decompresses) — should match
        result = await compressed_client.get(key)
        assert result == large_value.encode()

        # Read with raw client (no compression) — should differ (compressed bytes)
        raw_result = await raw_client.get(key)
        assert (
            raw_result != large_value.encode()
        ), "Value stored via compressed scope should be compressed in Valkey"

        # Cleanup
        await compressed_client.delete([key])

    async def test_scope_reads_compressed_data(self, compressed_client):
        """Scoped GET should decompress data written by the parent client."""
        key = f"async-scope-compress-read-{uuid.uuid4().hex[:8]}"
        value = "CompressibleData_" * 50  # ~850 bytes

        # Write via parent client (compressed)
        await compressed_client.set(key, value)

        # Read via scoped connection — should decompress correctly
        async with await compressed_client.scoped_connection() as scope:
            retrieved = await scope.get(key)
            assert retrieved == value

        # Cleanup
        await compressed_client.delete([key])


# ─── Database State Tests ─────────────────────────────────────────────────────


class TestAsyncDatabaseStateInheritance:
    """Tests that database selection is correctly handled across scope lifecycle."""

    async def test_scope_inherits_configured_database(self):
        """Scope connections use the database from the client's config."""
        # Create client configured for database 2
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=5000,
            database_id=2,
        )
        client = await GlideClient.create(config)

        key = f"async-db2-scope-{uuid.uuid4().hex[:8]}"
        try:
            # Write via scope on database 2
            async with await client.scoped_connection() as scope:
                await scope.set(key, "on-db2")
                val = await scope.get(key)
                assert val == "on-db2"

            # Parent client (also on db 2) should see the key
            result = await client.get(key)
            assert result == b"on-db2"

            # A client on database 0 should NOT see the key
            config_db0 = GlideClientConfiguration(
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=0,
            )
            client_db0 = await GlideClient.create(config_db0)
            result_db0 = await client_db0.get(key)
            assert (
                result_db0 is None
            ), "Key written on db2 via scope should not be visible on db0"
            await client_db0.aclose()
        finally:
            await client.custom_command(["DEL", key])
            await client.aclose()

    async def test_scope_inherits_runtime_select(self):
        """If parent calls SELECT at runtime, scope inherits the runtime database."""
        # Create client configured for database 0
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=5000,
            database_id=0,
        )
        client = await GlideClient.create(config)

        key = f"async-runtime-db-{uuid.uuid4().hex[:8]}"
        try:
            # Switch parent to db 3 at runtime
            await client.custom_command(["SELECT", "3"])
            await client.set(key, "on-db3")

            # Scope should inherit the parent's current database (3)
            async with await client.scoped_connection() as scope:
                result = await scope.get(key)
                assert (
                    result == "on-db3"
                ), "Scope should inherit parent's current runtime database (3)"

            # Clean up: delete key on db 3 and switch parent back
            await client.custom_command(["DEL", key])
            await client.custom_command(["SELECT", "0"])
        finally:
            await client.aclose()

    async def test_scope_release_resets_database(self):
        """After scope user calls SELECT, release resets to configured database.

        Next scope borrow from the same pool should be on the configured database.
        """
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=5000,
            database_id=0,
        )
        client = await GlideClient.create(config)

        key = f"async-scope-db-reset-{uuid.uuid4().hex[:8]}"

        try:
            # First scope: SELECT to db 4 and write
            async with await client.scoped_connection() as scope:
                await scope.execute_command("SELECT", "4")
                await scope.set(key, "on-db4")
            # Scope released — cleanup should reset to db 0

            await asyncio.sleep(0.3)  # Allow async cleanup to complete

            # Second scope: should be on db 0 (not db 4)
            async with await client.scoped_connection() as scope2:
                result = await scope2.get(key)
                assert result is None, (
                    "Scope release should reset database. Second scope should be on "
                    "configured db (0), not the previous scope's runtime db (4)."
                )
        finally:
            # Clean up key on db 4
            cleanup_config = GlideClientConfiguration(
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=4,
            )
            cleanup = await GlideClient.create(cleanup_config)
            await cleanup.custom_command(["DEL", key])
            await cleanup.aclose()
            await client.aclose()


# ─── Cluster Mode Basic Scope Tests ──────────────────────────────────────────


class TestAsyncClusterScopeBasicOperations:
    """Tests that basic scope operations work in cluster mode (async)."""

    @pytest_asyncio.fixture
    async def cluster_client(self):
        """Create a GlideClusterClient for basic scope tests."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client = await GlideClusterClient.create(config)
        yield client
        await client.aclose()

    async def test_cluster_scope_acquire_and_release(self, cluster_client):
        """Scope can be acquired and released on an async cluster client."""
        async with await cluster_client.scoped_connection() as scope:
            result = await scope.execute_command("PING")
            assert result == "PONG"

    async def test_cluster_scope_get_set(self, cluster_client):
        """Basic GET/SET works via async scoped connection in cluster mode."""
        key = f"{{scope-test}}-async-cluster-basic-{uuid.uuid4().hex[:8]}"

        async with await cluster_client.scoped_connection() as scope:
            await scope.set(key, "cluster-value")
            val = await scope.get(key)
            assert val == "cluster-value"

        # Verify via parent client
        result = await cluster_client.get(key)
        assert result == b"cluster-value"

        # Cleanup
        await cluster_client.delete([key])

    async def test_cluster_scope_watch_multi_exec(self, cluster_client):
        """WATCH/MULTI/EXEC works correctly via async scoped connection in cluster mode."""
        key = f"{{scope-test}}-async-cluster-occ-{uuid.uuid4().hex[:8]}"
        await cluster_client.set(key, "0")

        async with await cluster_client.scoped_connection() as scope:
            await scope.watch(key)
            current = await scope.get(key)
            assert current == "0"

            await scope.multi()
            await scope.set(key, "1")
            result = await scope.exec()
            assert result is not None and result != "None"

        # Verify
        final = await cluster_client.get(key)
        assert final == b"1"

        # Cleanup
        await cluster_client.delete([key])

    async def test_cluster_scope_watch_conflict_aborts_exec(self, cluster_client):
        """WATCH detects external modification and EXEC returns nil in cluster mode (async)."""
        key = f"{{scope-test}}-async-cluster-conflict-{uuid.uuid4().hex[:8]}"
        await cluster_client.set(key, "original")

        async with await cluster_client.scoped_connection() as scope:
            await scope.watch(key)
            await scope.get(key)

            # Modify externally via the main client
            await cluster_client.set(key, "modified-externally")

            await scope.multi()
            await scope.set(key, "from-scope")
            result = await scope.exec()
            # EXEC returns None when transaction is aborted
            assert result is None or result == "None"

        # Verify external modification persists
        val = await cluster_client.get(key)
        assert val == b"modified-externally"

        # Cleanup
        await cluster_client.delete([key])


# ─── Cluster Mode Compression Tests ──────────────────────────────────────────


class TestAsyncClusterScopeCompression:
    """Tests that scoped connections work with compression in cluster mode (async)."""

    @pytest_asyncio.fixture
    async def cluster_compressed_client(self):
        """Create an async GlideClusterClient with ZSTD compression enabled."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
            compression=CompressionConfiguration(
                enabled=True,
                backend=CompressionBackend.ZSTD,
                compression_level=3,
                min_compression_size=64,
            ),
        )
        client = await GlideClusterClient.create(config)
        yield client
        await client.aclose()

    @pytest_asyncio.fixture
    async def cluster_raw_client(self):
        """Create an async GlideClusterClient WITHOUT compression (for verification)."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client = await GlideClusterClient.create(config)
        yield client
        await client.aclose()

    async def test_cluster_scope_writes_compressed_data(
        self, cluster_compressed_client, cluster_raw_client
    ):
        """Data written via scope with compression in cluster mode should be compressed."""
        key = f"{{scope-test}}-async-cluster-compress-write-{uuid.uuid4().hex[:8]}"
        large_value = "A" * 500  # 500 bytes, well above 64-byte threshold

        # Write via scoped connection
        async with await cluster_compressed_client.scoped_connection() as scope:
            await scope.set(key, large_value)

        # Read with same client (decompresses) — should match
        result = await cluster_compressed_client.get(key)
        assert result == large_value.encode()

        # Read with raw client (no compression) — should differ (compressed bytes)
        raw_result = await cluster_raw_client.get(key)
        assert (
            raw_result != large_value.encode()
        ), "Value stored via compressed scope should be compressed in Valkey (cluster)"

        # Cleanup
        await cluster_compressed_client.delete([key])

    async def test_cluster_scope_reads_compressed_data(self, cluster_compressed_client):
        """Scoped GET should decompress data written by parent client in cluster mode."""
        key = f"{{scope-test}}-async-cluster-compress-read-{uuid.uuid4().hex[:8]}"
        value = "CompressibleData_" * 50  # ~850 bytes

        # Write via parent client (compressed)
        await cluster_compressed_client.set(key, value)

        # Read via scoped connection — should decompress correctly
        async with await cluster_compressed_client.scoped_connection() as scope:
            retrieved = await scope.get(key)
            assert retrieved == value

        # Cleanup
        await cluster_compressed_client.delete([key])

    async def test_cluster_scope_roundtrip_with_compression(
        self, cluster_compressed_client
    ):
        """Full round-trip: scope SET → scope GET on same scope in cluster mode."""
        key = f"{{scope-test}}-async-cluster-roundtrip-{uuid.uuid4().hex[:8]}"
        value = "RoundTripData_" * 40  # ~560 bytes

        async with await cluster_compressed_client.scoped_connection() as scope:
            await scope.set(key, value)
            retrieved = await scope.get(key)
            assert retrieved == value

        # Cleanup
        await cluster_compressed_client.delete([key])


# ─── Cluster Mode Database Tests (Valkey 9+ only) ────────────────────────────


class TestAsyncClusterDatabaseStateInheritance:
    """Tests that database selection works in cluster mode (requires Valkey 9+, async)."""

    @pytest_asyncio.fixture
    async def cluster_client_db2(self):
        """Create an async GlideClusterClient configured for database 2 (Valkey 9+)."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
            database_id=2,
        )
        try:
            client = await GlideClusterClient.create(config)
        except Exception:
            pytest.skip("Cluster database selection not supported (requires Valkey 9+)")
        # Verify version
        ver = await _get_server_version(client)
        if version.parse(ver) < version.parse("9.0.0"):
            await client.aclose()
            pytest.skip(
                f"Requires Valkey 9+ for cluster database selection (got {ver})"
            )
        yield client
        await client.aclose()

    async def test_cluster_scope_inherits_configured_database(self, cluster_client_db2):
        """Scope connections in cluster mode use the database from the client's config."""
        key = f"{{scope-test}}-async-cluster-db2-{uuid.uuid4().hex[:8]}"

        # Write via scope on database 2
        async with await cluster_client_db2.scoped_connection() as scope:
            await scope.set(key, "on-db2")
            val = await scope.get(key)
            assert val == "on-db2"

        # Parent client (also on db 2) should see the key
        result = await cluster_client_db2.get(key)
        assert result == b"on-db2"

        # A client on database 0 should NOT see the key
        config_db0 = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client_db0 = await GlideClusterClient.create(config_db0)
        result_db0 = await client_db0.get(key)
        assert (
            result_db0 is None
        ), "Key written on db2 via scope should not be visible on db0 (cluster)"
        await client_db0.aclose()

        # Cleanup
        await cluster_client_db2.custom_command(["DEL", key])

    async def test_cluster_scope_release_resets_database(self, cluster_client_db2):
        """After scope user calls SELECT on cluster, release resets to configured database."""
        key = f"{{scope-test}}-async-cluster-db-reset-{uuid.uuid4().hex[:8]}"

        # First scope: SELECT to db 4 and write
        async with await cluster_client_db2.scoped_connection() as scope:
            await scope.execute_command("SELECT", "4")
            await scope.set(key, "on-db4")
        # Scope released — cleanup should reset to db 2

        await asyncio.sleep(0.3)  # Allow async cleanup to complete

        # Second scope: should be on db 2 (configured), not db 4
        async with await cluster_client_db2.scoped_connection() as scope2:
            result = await scope2.get(key)
            assert result is None, (
                "Scope release should reset database. Second scope should be on "
                "configured db (2), not the previous scope's runtime db (4)."
            )

        # Cleanup key on db 4
        config_db4 = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
            database_id=4,
        )
        try:
            cleanup = await GlideClusterClient.create(config_db4)
            await cleanup.custom_command(["DEL", key])
            await cleanup.aclose()
        except Exception:
            pass  # Best-effort cleanup
