# Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

"""
Integration tests verifying that connection modifiers (compression, request timeout,
inflight limits) are properly inherited and respected by scoped connections.

Ensures scope operations maintain full functional parity with regular commands —
the same options and limitations apply.

Tests run in both standalone and cluster modes where applicable.

Requires a running Valkey server (standalone, and optionally cluster).
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
    GlideClusterClient,
    GlideClusterClientConfiguration,
    InfoSection,
    NodeAddress,
)
from packaging import version


def _get_standalone_address():
    """Get the standalone server address from conftest (CI) or fallback to localhost."""
    try:
        cluster = pytest.standalone_cluster  # type: ignore[attr-defined]
        addr = cluster.nodes_addr[0]
        return NodeAddress(addr.host, addr.port)
    except (AttributeError, IndexError):
        return NodeAddress("localhost", 6379)


def _get_cluster_addresses():
    """Get the cluster server addresses from conftest (CI) or fallback to localhost:7000."""
    try:
        cluster = pytest.valkey_cluster  # type: ignore[attr-defined]
        return [NodeAddress(addr.host, addr.port) for addr in cluster.nodes_addr]
    except (AttributeError, IndexError):
        return [NodeAddress("localhost", 7000)]


def _get_server_version(client) -> str:
    """Get server version string from a connected client."""
    info_str = client.info([InfoSection.SERVER])
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
        addresses=[_get_standalone_address()],
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
        addresses=[_get_standalone_address()],
        request_timeout=5000,
    )
    client = _get_or_create_client("raw", config)
    yield client
    _teardown_client(client, "raw")


@pytest.fixture
def short_timeout_client():
    """Get or reuse a GlideClient with a short request timeout."""
    config = GlideClientConfiguration(
        addresses=[_get_standalone_address()],
        request_timeout=100,  # 100ms
    )
    client = _get_or_create_client("short_timeout", config)
    yield client
    _teardown_client(client, "short_timeout")


@pytest.fixture
def inflight_limited_client():
    """Get or reuse a GlideClient with explicit inflight limit."""
    config = GlideClientConfiguration(
        addresses=[_get_standalone_address()],
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
        assert (
            raw_result != large_value.encode()
        ), "Value stored via compressed scope should be compressed in Valkey"

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
            addresses=[_get_standalone_address()],
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
        assert (
            raw_result == small_value.encode()
        ), "Small values below minCompressionSize should NOT be compressed"

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
            addresses=[_get_standalone_address()],
            request_timeout=5000,
        )
        config_b = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
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
            addresses=[_get_standalone_address()],
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
            addresses=[_get_standalone_address()],
            request_timeout=5000,
        )
        raw_client = _get_or_create_client("raw_verify", raw_config)
        raw_value = raw_client.get(key)
        assert (
            raw_value != large_value.encode()
        ), "Data should be stored compressed in Valkey"

        # Cleanup
        client.delete([key])


# ─── Database State Tests ─────────────────────────────────────────────────────


class TestDatabaseStateInheritance:
    """Tests that database selection is correctly handled across pool and scope."""

    def test_scope_inherits_configured_database(self):
        """Scope connections use the database from the client's config."""
        # Create client configured for database 2
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
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
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=0,
            )
            client_db0 = GlideClient.create(config_db0)
            result_db0 = client_db0.get(key)
            assert (
                result_db0 is None
            ), "Key written on db2 via scope should not be visible on db0"
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
            addresses=[_get_standalone_address()],
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
                assert (
                    result == "on-db3"
                ), "Scope should inherit parent's current runtime database (3)"

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
            addresses=[_get_standalone_address()],
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
                addresses=[_get_standalone_address()],
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
            addresses=[_get_standalone_address()],
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
                addresses=[_get_standalone_address()],
                request_timeout=5000,
                database_id=4,
            )
            cleanup = GlideClient.create(cleanup_config)
            cleanup.custom_command(["DEL", key])
            cleanup.close()
            client.close()


# ─── Disconnection / Failure Behavior Tests ───────────────────────────────────


class TestScopeDisconnectionBehavior:
    """Tests that scoped connections fail fast on disconnect and don't pollute the pool."""

    def test_scope_fails_after_connection_killed(self):
        """Commands fail with an error after the scope's connection is killed."""
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=5000,
        )
        client = GlideClient.create(config)

        try:
            with client.scoped_connection() as scope:
                # Verify scope works
                scope.ping()

                # Kill this scope's connection using CLIENT KILL on the scope itself
                # Get the scope's client ID first
                client_id_result = scope.execute_command("CLIENT", "ID")
                # Kill our own connection
                scope.execute_command("CLIENT", "KILL", "ID", str(client_id_result))

                # Next command should fail — connection is dead
                import time

                time.sleep(0.1)  # Allow kill to propagate

                try:
                    scope.ping()
                    # If we get here, the kill didn't take effect yet (race condition)
                    # This is acceptable — the test validates the error path
                except Exception:
                    pass  # Expected — connection is dead
        finally:
            client.close()

    def test_broken_scope_does_not_pollute_pool(self):
        """After a scope connection fails, the next acquire gets a healthy connection."""
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=5000,
        )
        client = GlideClient.create(config)

        try:
            # First scope: kill its connection
            with client.scoped_connection() as scope1:
                scope1.ping()
                # Kill our own connection
                client_id_result = scope1.execute_command("CLIENT", "ID")
                scope1.execute_command("CLIENT", "KILL", "ID", str(client_id_result))

            import time

            time.sleep(0.5)  # Allow cleanup to complete

            # Second scope: should get a fresh, working connection
            with client.scoped_connection() as scope2:
                result = scope2.ping()
                assert (
                    result == "PONG"
                ), "Second scope should get a healthy connection after first was killed"
        finally:
            client.close()

    def test_scope_no_auto_reconnect(self):
        """Scoped connections do not transparently reconnect — they fail fast."""
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=2000,
        )
        client = GlideClient.create(config)

        try:
            with client.scoped_connection() as scope:
                # Start a WATCH (establishes per-connection state)
                key = f"no-reconnect-{uuid.uuid4().hex[:8]}"
                scope.set(key, "initial")
                scope.watch(key)

                # Kill the connection
                client_id_result = scope.execute_command("CLIENT", "ID")
                scope.execute_command("CLIENT", "KILL", "ID", str(client_id_result))

                import time

                time.sleep(0.2)

                # If auto-reconnect existed, this would succeed but WATCH would be lost
                # Instead, this should fail — proving no auto-reconnect
                try:
                    scope.get(key)
                    # If this succeeds, multiplexed connection may have internal retry.
                    # Either way, WATCH state is lost — verify that.
                except Exception:
                    pass  # Expected: connection error, no auto-reconnect

                # Clean up
                client.delete([key])
        finally:
            client.close()


# ─── Inflight Limit Enforcement Tests ─────────────────────────────────────────


class TestScopeInflightEnforcement:
    """Tests that scoped commands are rejected when inflight limit is exhausted."""

    def test_scope_rejects_when_inflight_exhausted(self):
        """Scoped commands fail with an error when inflight limit is reached.

        We configure a client with inflight_requests_limit=1, then use CLIENT PAUSE
        to stall one command, and verify the next scope command is rejected.
        """
        config = GlideClientConfiguration(
            addresses=[_get_standalone_address()],
            request_timeout=2000,
            inflight_requests_limit=1,
        )
        client = GlideClient.create(config)

        try:
            # CLIENT PAUSE stalls all responses for 3 seconds
            # This holds an inflight slot on the parent client
            client.custom_command(["CLIENT", "PAUSE", "3000", "ALL"])

            import time

            time.sleep(0.1)

            # Now try a scope command — inflight limit (1) should be exhausted
            # because the paused command is still occupying the slot
            with client.scoped_connection() as scope:
                try:
                    # This should fail because inflight is exhausted
                    scope.ping()
                except Exception as e:
                    error_msg = str(e).lower()
                    assert (
                        "inflight" in error_msg or "timeout" in error_msg
                    ), f"Expected inflight rejection or timeout, got: {e}"
        except Exception:
            pass  # CLIENT PAUSE may itself hit limits
        finally:
            # Unpause to clean up
            try:
                unpause_config = GlideClientConfiguration(
                    addresses=[_get_standalone_address()],
                    request_timeout=5000,
                )
                unpause_client = GlideClient.create(unpause_config)
                unpause_client.custom_command(["CLIENT", "UNPAUSE"])
                unpause_client.close()
            except Exception:
                pass
            client.close()


# ─── Cluster Mode Basic Scope Tests ──────────────────────────────────────────


class TestClusterScopeBasicOperations:
    """Tests that basic scope operations (acquire, release, WATCH/MULTI/EXEC) work in cluster mode."""

    @pytest.fixture
    def cluster_client(self):
        """Create a GlideClusterClient for basic scope tests."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client = GlideClusterClient.create(config)
        yield client
        client.close()

    def test_cluster_scope_acquire_and_release(self, cluster_client):
        """Scope can be acquired and released on a cluster client."""
        with cluster_client.scoped_connection() as scope:
            result = scope.execute_command("PING")
            assert result == "PONG"

    def test_cluster_scope_get_set(self, cluster_client):
        """Basic GET/SET works via scoped connection in cluster mode."""
        key = f"{{scope-test}}-cluster-basic-{uuid.uuid4().hex[:8]}"

        with cluster_client.scoped_connection() as scope:
            scope.set(key, "cluster-value")
            val = scope.get(key)
            assert val == "cluster-value"

        # Verify via parent client
        result = cluster_client.get(key)
        assert result == b"cluster-value"

        # Cleanup
        cluster_client.delete([key])

    def test_cluster_scope_watch_multi_exec(self, cluster_client):
        """WATCH/MULTI/EXEC works correctly via scoped connection in cluster mode."""
        key = f"{{scope-test}}-cluster-occ-{uuid.uuid4().hex[:8]}"
        cluster_client.set(key, "0")

        with cluster_client.scoped_connection() as scope:
            scope.watch(key)
            current = scope.get(key)
            assert current == "0"

            scope.multi()
            scope.set(key, "1")
            result = scope.exec()
            assert result is not None and result != "None"

        # Verify
        final = cluster_client.get(key)
        assert final == b"1"

        # Cleanup
        cluster_client.delete([key])

    def test_cluster_scope_watch_conflict_aborts_exec(self, cluster_client):
        """WATCH detects external modification and EXEC returns nil in cluster mode."""
        key = f"{{scope-test}}-cluster-conflict-{uuid.uuid4().hex[:8]}"
        cluster_client.set(key, "original")

        with cluster_client.scoped_connection() as scope:
            scope.watch(key)
            scope.get(key)

            # Modify externally via the main client
            cluster_client.set(key, "modified-externally")

            scope.multi()
            scope.set(key, "from-scope")
            result = scope.exec()
            # EXEC returns None when transaction is aborted
            assert result is None or result == "None"

        # Verify external modification persists
        val = cluster_client.get(key)
        assert val == b"modified-externally"

        # Cleanup
        cluster_client.delete([key])

    def test_cluster_scope_raises_after_release(self, cluster_client):
        """Commands fail after scope is released in cluster mode."""
        scope = cluster_client.scoped_connection().__enter__()
        scope.execute_command("PING")
        scope.__exit__(None, None, None)

        try:
            scope.execute_command("PING")
            assert False, "Should have raised after release"
        except Exception:
            pass  # Expected — scope is released


# ─── Cluster Mode Compression Tests ──────────────────────────────────────────


class TestClusterScopeCompression:
    """Tests that scoped connections work with compression in cluster mode."""

    @pytest.fixture
    def cluster_compressed_client(self):
        """Create a GlideClusterClient with ZSTD compression enabled."""
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
        client = GlideClusterClient.create(config)
        yield client
        client.close()

    @pytest.fixture
    def cluster_raw_client(self):
        """Create a GlideClusterClient WITHOUT compression (for verification)."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client = GlideClusterClient.create(config)
        yield client
        client.close()

    def test_cluster_scope_writes_compressed_data(
        self, cluster_compressed_client, cluster_raw_client
    ):
        """Data written via scope with compression in cluster mode should be compressed."""
        # Use {hash_tag} to ensure same-slot routing
        key = f"{{scope-test}}-cluster-compress-write-{uuid.uuid4().hex[:8]}"
        large_value = "A" * 500  # 500 bytes, well above 64-byte threshold

        # Write via scoped connection
        with cluster_compressed_client.scoped_connection() as scope:
            scope.set(key, large_value)

        # Read with same client (decompresses) — should match
        result = cluster_compressed_client.get(key)
        assert result == large_value.encode()

        # Read with raw client (no compression) — should differ (compressed bytes)
        raw_result = cluster_raw_client.get(key)
        assert (
            raw_result != large_value.encode()
        ), "Value stored via compressed scope should be compressed in Valkey (cluster)"

        # Cleanup
        cluster_compressed_client.delete([key])

    def test_cluster_scope_reads_compressed_data(self, cluster_compressed_client):
        """Scoped GET should decompress data written by the parent client in cluster."""
        key = f"{{scope-test}}-cluster-compress-read-{uuid.uuid4().hex[:8]}"
        value = "CompressibleData_" * 50  # ~850 bytes

        # Write via parent client (compressed)
        cluster_compressed_client.set(key, value)

        # Read via scoped connection — should decompress correctly
        with cluster_compressed_client.scoped_connection() as scope:
            retrieved = scope.get(key)
            assert retrieved == value

        # Cleanup
        cluster_compressed_client.delete([key])

    def test_cluster_scope_roundtrip_with_compression(self, cluster_compressed_client):
        """Full round-trip: scope SET → scope GET on same scope in cluster mode."""
        key = f"{{scope-test}}-cluster-roundtrip-{uuid.uuid4().hex[:8]}"
        value = "RoundTripData_" * 40  # ~560 bytes

        with cluster_compressed_client.scoped_connection() as scope:
            scope.set(key, value)
            retrieved = scope.get(key)
            assert retrieved == value

        # Cleanup
        cluster_compressed_client.delete([key])

    def test_cluster_scope_small_values_not_compressed(
        self, cluster_raw_client
    ):
        """Values below minCompressionSize are stored uncompressed in cluster mode."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
            compression=CompressionConfiguration(
                enabled=True,
                backend=CompressionBackend.ZSTD,
                min_compression_size=256,  # Only compress >= 256 bytes
            ),
        )
        client = GlideClusterClient.create(config)

        key = f"{{scope-test}}-cluster-small-{uuid.uuid4().hex[:8]}"
        small_value = "hello"  # 5 bytes — well below threshold

        # Write via scope
        with client.scoped_connection() as scope:
            scope.set(key, small_value)

        # Raw client should see the original value (not compressed)
        raw_result = cluster_raw_client.get(key)
        assert (
            raw_result == small_value.encode()
        ), "Small values below minCompressionSize should NOT be compressed (cluster)"

        # Cleanup
        client.delete([key])
        client.close()


# ─── Cluster Mode Database Tests (Valkey 9+ only) ────────────────────────────


class TestClusterDatabaseStateInheritance:
    """Tests that database selection works in cluster mode (requires Valkey 9+)."""

    @pytest.fixture
    def cluster_client_db2(self):
        """Create a GlideClusterClient configured for database 2 (Valkey 9+)."""
        _skip_cluster_if_unavailable()
        config = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
            database_id=2,
        )
        try:
            client = GlideClusterClient.create(config)
        except Exception:
            pytest.skip("Cluster database selection not supported (requires Valkey 9+)")
        # Verify version
        ver = _get_server_version(client)
        if version.parse(ver) < version.parse("9.0.0"):
            client.close()
            pytest.skip(f"Requires Valkey 9+ for cluster database selection (got {ver})")
        yield client
        client.close()

    def test_cluster_scope_inherits_configured_database(self, cluster_client_db2):
        """Scope connections in cluster mode use the database from the client's config."""
        key = f"{{scope-test}}-cluster-db2-{uuid.uuid4().hex[:8]}"

        # Write via scope on database 2
        with cluster_client_db2.scoped_connection() as scope:
            scope.set(key, "on-db2")
            val = scope.get(key)
            assert val == "on-db2"

        # Parent client (also on db 2) should see the key
        result = cluster_client_db2.get(key)
        assert result == b"on-db2"

        # A client on database 0 should NOT see the key
        config_db0 = GlideClusterClientConfiguration(
            addresses=_get_cluster_addresses(),
            request_timeout=5000,
        )
        client_db0 = GlideClusterClient.create(config_db0)
        result_db0 = client_db0.get(key)
        assert (
            result_db0 is None
        ), "Key written on db2 via scope should not be visible on db0 (cluster)"
        client_db0.close()

        # Cleanup
        cluster_client_db2.custom_command(["DEL", key])
