# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, database selection)
are properly inherited and respected by async scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Requires a running Valkey server (standalone) on localhost:6379.

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
    NodeAddress,
)

pytestmark = pytest.mark.asyncio


def _get_standalone_address():
    """Get the standalone server address from conftest (CI) or fallback to localhost."""
    try:
        cluster = pytest.standalone_cluster  # type: ignore[attr-defined]
        addr = cluster.nodes_addr[0]
        return NodeAddress(addr.host, addr.port)
    except (AttributeError, IndexError):
        return NodeAddress("localhost", 6379)


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
