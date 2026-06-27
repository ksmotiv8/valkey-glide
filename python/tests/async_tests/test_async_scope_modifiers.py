# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, database selection)
are properly inherited and respected by async scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Tests run in both standalone and cluster modes via @pytest.mark.parametrize.

Requires a running Valkey server (standalone, and optionally cluster).

Run with:
    PYTHONPATH="glide-async/python:glide-shared:$PYTHONPATH" \
    pytest tests/async_tests/test_async_scope_modifiers.py --noconftest -v
"""

import asyncio
import uuid

import pytest
from glide import (
    CompressionBackend,
    CompressionConfiguration,
    GlideClient,
    GlideClientConfiguration,
    GlideClusterClient,
    GlideClusterClientConfiguration,
    InfoSection,
)
from packaging import version

from tests.utils.utils import get_cluster_addresses as _get_cluster_addresses
from tests.utils.utils import get_standalone_address as _get_standalone_address

pytestmark = pytest.mark.asyncio


async def _get_server_version(client) -> str:
    """Get server version string from a connected client."""
    info_result = await client.info([InfoSection.SERVER])
    # Cluster clients return dict[str, str], standalone returns str
    if isinstance(info_result, dict):
        info_str = next(iter(info_result.values()), "")
    else:
        info_str = info_result
    for line in str(info_str).split("\n"):
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


def _skip_standalone_if_unavailable():
    """Skip test if no standalone endpoints are configured."""
    try:
        _get_standalone_address()
    except Exception:
        pytest.skip("No standalone endpoints available")


# ─── Helpers for parameterized mode ───────────────────────────────────────────


def _make_key(cluster_mode: bool, prefix: str) -> str:
    """Generate a key with hash tag for cluster mode."""
    uid = uuid.uuid4().hex[:8]
    if cluster_mode:
        return f"{{scope-test}}-{prefix}-{uid}"
    return f"scope-test-{prefix}-{uid}"


async def _create_client(cluster_mode: bool, **extra_config):
    """Create a GlideClient or GlideClusterClient based on mode."""
    if cluster_mode:
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            **extra_config,
        )
        return await GlideClusterClient.create(config)
    else:
        _skip_standalone_if_unavailable()
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            **extra_config,
        )
        return await GlideClient.create(config)


async def _close_client(client):
    """Close a client (works for both types)."""
    try:
        await client.aclose()
    except Exception:
        pass


# ─── Compression Tests ────────────────────────────────────────────────────────


class TestAsyncScopeCompression:
    """Tests that async scoped connections inherit and apply compression correctly."""

    async def _get_compressed_client(self, cluster_mode: bool):
        """Create a client with ZSTD compression enabled."""
        return await _create_client(
            cluster_mode,
            request_timeout=5000,
            compression=CompressionConfiguration(
                enabled=True,
                backend=CompressionBackend.ZSTD,
                compression_level=3,
                min_compression_size=64,
            ),
        )

    async def _get_raw_client(self, cluster_mode: bool):
        """Create a client WITHOUT compression (for verification)."""
        return await _create_client(cluster_mode, request_timeout=5000)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_writes_compressed_data(self, cluster_mode):
        """Data written via scope with compression should be compressed in Valkey."""
        compressed_client = await self._get_compressed_client(cluster_mode)
        raw_client = await self._get_raw_client(cluster_mode)
        try:
            key = _make_key(cluster_mode, "compress-write")
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
        finally:
            await _close_client(compressed_client)
            await _close_client(raw_client)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_reads_compressed_data(self, cluster_mode):
        """Scoped GET should decompress data written by the parent client."""
        compressed_client = await self._get_compressed_client(cluster_mode)
        try:
            key = _make_key(cluster_mode, "compress-read")
            value = "CompressibleData_" * 50  # ~850 bytes

            # Write via parent client (compressed)
            await compressed_client.set(key, value)

            # Read via scoped connection — should decompress correctly
            async with await compressed_client.scoped_connection() as scope:
                retrieved = await scope.get(key)
                assert retrieved == value

            # Cleanup
            await compressed_client.delete([key])
        finally:
            await _close_client(compressed_client)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_roundtrip_with_compression(self, cluster_mode):
        """Full round-trip: scope SET → scope GET on same scope."""
        compressed_client = await self._get_compressed_client(cluster_mode)
        try:
            key = _make_key(cluster_mode, "roundtrip")
            value = "RoundTripData_" * 40  # ~560 bytes

            async with await compressed_client.scoped_connection() as scope:
                await scope.set(key, value)
                retrieved = await scope.get(key)
                assert retrieved == value

            # Cleanup
            await compressed_client.delete([key])
        finally:
            await _close_client(compressed_client)


# ─── Basic Scope Operations ───────────────────────────────────────────────────


class TestAsyncScopeBasicOperations:
    """Tests that basic scope operations work in both modes (async)."""

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_acquire_and_release(self, cluster_mode):
        """Scope can be acquired and released on an async client."""
        client = await _create_client(cluster_mode, request_timeout=5000)
        try:
            async with await client.scoped_connection() as scope:
                result = await scope.execute_command("PING")
                assert result == "PONG"
        finally:
            await _close_client(client)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_get_set(self, cluster_mode):
        """Basic GET/SET works via async scoped connection."""
        client = await _create_client(cluster_mode, request_timeout=5000)
        try:
            key = _make_key(cluster_mode, "basic")

            async with await client.scoped_connection() as scope:
                await scope.set(key, "value")
                val = await scope.get(key)
                assert val == "value"

            # Verify via parent client
            result = await client.get(key)
            assert result == b"value"

            # Cleanup
            await client.delete([key])
        finally:
            await _close_client(client)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_watch_multi_exec(self, cluster_mode):
        """WATCH/MULTI/EXEC works correctly via async scoped connection."""
        client = await _create_client(cluster_mode, request_timeout=5000)
        try:
            key = _make_key(cluster_mode, "occ")
            await client.set(key, "0")

            async with await client.scoped_connection() as scope:
                await scope.watch(key)
                current = await scope.get(key)
                assert current == "0"

                await scope.multi()
                await scope.set(key, "1")
                result = await scope.exec()
                assert result is not None and result != "None"

            # Verify
            final = await client.get(key)
            assert final == b"1"

            # Cleanup
            await client.delete([key])
        finally:
            await _close_client(client)

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_watch_conflict_aborts_exec(self, cluster_mode):
        """WATCH detects external modification and EXEC returns nil (async)."""
        client = await _create_client(cluster_mode, request_timeout=5000)
        try:
            key = _make_key(cluster_mode, "conflict")
            await client.set(key, "original")

            async with await client.scoped_connection() as scope:
                await scope.watch(key)
                await scope.get(key)

                # Modify externally via the main client
                await client.set(key, "modified-externally")

                await scope.multi()
                await scope.set(key, "from-scope")
                result = await scope.exec()
                # EXEC returns None when transaction is aborted
                assert result is None or result == "None"

            # Verify external modification persists
            val = await client.get(key)
            assert val == b"modified-externally"

            # Cleanup
            await client.delete([key])
        finally:
            await _close_client(client)


# ─── Database State Tests ─────────────────────────────────────────────────────


class TestAsyncDatabaseStateInheritance:
    """Tests that database selection is correctly handled across scope lifecycle."""

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_inherits_configured_database(self, cluster_mode):
        """Scope connections use the database from the client's config."""
        if cluster_mode:
            _skip_cluster_if_unavailable()
            config = GlideClusterClientConfiguration(
                addresses=_get_cluster_addresses(),
                request_timeout=5000,
                database_id=2,
            )
            try:
                client = await GlideClusterClient.create(config)
            except Exception:
                pytest.skip(
                    "Cluster database selection not supported (requires Valkey 9+)"
                )
            ver = await _get_server_version(client)
            if version.parse(ver) < version.parse("9.0.0"):
                await client.aclose()
                pytest.skip(
                    f"Requires Valkey 9+ for cluster database selection (got {ver})"
                )
        else:
            _skip_standalone_if_unavailable()
            config = GlideClientConfiguration(
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=2,
            )
            client = await GlideClient.create(config)

        key = _make_key(cluster_mode, "db2-scope")
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
            if cluster_mode:
                config_db0 = GlideClusterClientConfiguration(
                    addresses=_get_cluster_addresses(),
                    request_timeout=5000,
                )
                client_db0 = await GlideClusterClient.create(config_db0)
            else:
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

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_inherits_runtime_select(self, cluster_mode):
        """If parent calls SELECT at runtime, scope inherits the runtime database."""
        if cluster_mode:
            _skip_cluster_if_unavailable()
            config = GlideClusterClientConfiguration(
                addresses=_get_cluster_addresses(),
                request_timeout=5000,
            )
            try:
                client = await GlideClusterClient.create(config)
            except Exception:
                pytest.skip(
                    "Cluster database selection not supported (requires Valkey 9+)"
                )
            ver = await _get_server_version(client)
            if version.parse(ver) < version.parse("9.0.0"):
                await client.aclose()
                pytest.skip(
                    f"Requires Valkey 9+ for cluster database selection (got {ver})"
                )
        else:
            _skip_standalone_if_unavailable()
            config = GlideClientConfiguration(
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=0,
            )
            client = await GlideClient.create(config)

        key = _make_key(cluster_mode, "runtime-db")
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

    @pytest.mark.parametrize("cluster_mode", [True, False])
    async def test_scope_release_resets_database(self, cluster_mode):
        """After scope user calls SELECT, release resets to configured database.

        Next scope borrow from the same pool should be on the configured database.
        """
        if cluster_mode:
            _skip_cluster_if_unavailable()
            config = GlideClusterClientConfiguration(
                addresses=_get_cluster_addresses(),
                request_timeout=5000,
                database_id=2,
            )
            try:
                client = await GlideClusterClient.create(config)
            except Exception:
                pytest.skip(
                    "Cluster database selection not supported (requires Valkey 9+)"
                )
            ver = await _get_server_version(client)
            if version.parse(ver) < version.parse("9.0.0"):
                await client.aclose()
                pytest.skip(
                    f"Requires Valkey 9+ for cluster database selection (got {ver})"
                )
            configured_db = 2
        else:
            _skip_standalone_if_unavailable()
            config = GlideClientConfiguration(
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=0,
            )
            client = await GlideClient.create(config)
            configured_db = 0

        key = _make_key(cluster_mode, "scope-db-reset")
        try:
            # First scope: SELECT to db 4 and write
            async with await client.scoped_connection() as scope:
                await scope.execute_command("SELECT", "4")
                await scope.set(key, "on-db4")
            # Scope released — cleanup should reset to configured db

            await asyncio.sleep(0.3)  # Allow async cleanup to complete

            # Second scope: should be on configured db (not db 4)
            async with await client.scoped_connection() as scope2:
                result = await scope2.get(key)
                assert result is None, (
                    f"Scope release should reset database. Second scope should be on "
                    f"configured db ({configured_db}), not the previous scope's "
                    f"runtime db (4)."
                )
        finally:
            # Clean up key on db 4
            if cluster_mode:
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
                    pass
            else:
                config_db4 = GlideClientConfiguration(
                    addresses=[_get_standalone_address()],
                    request_timeout=5000,
                    database_id=4,
                )
                cleanup = await GlideClient.create(config_db4)
                await cleanup.custom_command(["DEL", key])
                await cleanup.aclose()
            await client.aclose()
