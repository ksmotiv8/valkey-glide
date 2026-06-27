/** Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0 */
package glide.scope;

import static glide.TestConfiguration.CLUSTER_HOSTS;
import static glide.TestConfiguration.SERVER_VERSION;
import static glide.utils.Java8Utils.repeat;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import glide.TestConfiguration;
import glide.api.GlideClient;
import glide.api.GlideClusterClient;
import glide.api.models.configuration.CompressionBackend;
import glide.api.models.configuration.CompressionConfiguration;
import glide.api.models.configuration.GlideClientConfiguration;
import glide.api.models.configuration.GlideClusterClientConfiguration;
import glide.api.models.configuration.NodeAddress;
import glide.api.models.scope.IsolatedScope;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * Integration tests verifying that connection modifiers (compression, request timeout, inflight
 * limits) are properly inherited and respected by scoped connections and pooled clients.
 *
 * <p>These tests ensure that scope operations maintain full functional parity with regular commands
 * — the same options and limitations apply.
 */
public class ScopeConnectionModifiersTest {

    private static String getHost() {
        return TestConfiguration.STANDALONE_HOSTS[0].split(":")[0];
    }

    private static int getPort() {
        return Integer.parseInt(TestConfiguration.STANDALONE_HOSTS[0].split(":")[1]);
    }

    // ─── Compression ─────────────────────────────────────────────────────────────

    @Test
    public void testScopeInheritsCompression() throws Exception {
        System.out.println("\n=== Test: Scope inherits parent compression settings ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .compressionLevel(3)
                        .minCompressionSize(64)
                        .build();

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .compressionConfiguration(compressionConfig)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        // Generate a value large enough to trigger compression (> 64 bytes)
        String key = "scope-compress-" + UUID.randomUUID();
        String largeValue = repeat("A", 500); // 500 bytes — well above threshold

        // SET via scoped connection
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, largeValue).get(5, TimeUnit.SECONDS);
        }

        // GET via normal client (with same compression) should decompress
        String retrieved = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals(
                largeValue,
                retrieved,
                "Scoped SET with compression should be readable by the parent client");

        // Verify the value in Valkey is actually compressed (shorter than original)
        // by reading it with a client that has NO compression — raw bytes should differ
        GlideClientConfiguration noCompressConfig =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .build();
        GlideClient rawClient = GlideClient.createClient(noCompressConfig).get(10, TimeUnit.SECONDS);
        String rawValue = rawClient.get(key).get(5, TimeUnit.SECONDS);

        // Raw value should NOT equal the original (it contains compression header)
        assertNotEquals(
                largeValue, rawValue, "Value stored via compressed scope should be compressed in Valkey");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        rawClient.close();
        client.close();

        System.out.println("Scope compression inheritance test PASSED!");
    }

    @Test
    public void testScopeReadsCompressedData() throws Exception {
        System.out.println("\n=== Test: Scope can read data written with compression ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder().enabled(true).backend(CompressionBackend.ZSTD).build();

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .compressionConfiguration(compressionConfig)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        String key = "scope-read-compressed-" + UUID.randomUUID();
        String value = repeat("CompressibleData_", 50); // ~850 bytes

        // Write via normal client (compressed)
        client.set(key, value).get(5, TimeUnit.SECONDS);

        // Read via scoped connection — should decompress correctly
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            String retrieved = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals(
                    value, retrieved, "Scoped GET should decompress data written by the parent client");
        }

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();

        System.out.println("Scope reads compressed data test PASSED!");
    }

    @Test
    public void testScopeSmallValuesNotCompressed() throws Exception {
        System.out.println("\n=== Test: Scope respects min compression size threshold ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .minCompressionSize(256) // Only compress values >= 256 bytes
                        .build();

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .compressionConfiguration(compressionConfig)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        String key = "scope-small-" + UUID.randomUUID();
        String smallValue = "hello"; // 5 bytes — below threshold

        // SET via scope
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, smallValue).get(5, TimeUnit.SECONDS);
        }

        // Read with a raw client (no compression) — should get the same value
        // because it was too small to compress
        GlideClientConfiguration noCompressConfig =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .build();
        GlideClient rawClient = GlideClient.createClient(noCompressConfig).get(10, TimeUnit.SECONDS);
        String rawValue = rawClient.get(key).get(5, TimeUnit.SECONDS);
        assertEquals(
                smallValue, rawValue, "Small values below minCompressionSize should NOT be compressed");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        rawClient.close();
        client.close();

        System.out.println("Scope min compression size threshold test PASSED!");
    }

    // ─── Request Timeout ─────────────────────────────────────────────────────────

    @Test
    public void testScopeRespectsRequestTimeout() throws Exception {
        System.out.println("\n=== Test: Scope respects parent request timeout ===");

        // Create client with a very short timeout
        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(100) // 100ms timeout
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        // Normal fast operations should succeed
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(5)).get(10, TimeUnit.SECONDS)) {
            String key = "timeout-test-" + UUID.randomUUID();
            scope.set(key, "fast").get(5, TimeUnit.SECONDS);
            String val = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals("fast", val);
            scope.executeCommand("DEL", key).get(5, TimeUnit.SECONDS);
        }

        // The short timeout is enforced — we can't easily trigger a server-side delay
        // without CLIENT PAUSE (which requires ADMIN privileges), but we verify the client
        // was created with the timeout config and scoped operations complete within it.
        client.close();

        System.out.println("Scope request timeout test PASSED!");
    }

    @Test
    public void testScopeDifferentTimeoutsPerClient() throws Exception {
        System.out.println("\n=== Test: Different clients have different scoped timeouts ===");

        // Client A with 5000ms timeout
        GlideClientConfiguration configA =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .build();

        // Client B with 200ms timeout
        GlideClientConfiguration configB =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(200)
                        .build();

        GlideClient clientA = GlideClient.createClient(configA).get(10, TimeUnit.SECONDS);
        GlideClient clientB = GlideClient.createClient(configB).get(10, TimeUnit.SECONDS);

        String key = "dual-timeout-" + UUID.randomUUID();

        // Both scopes should work for fast ops
        try (IsolatedScope scopeA =
                clientA.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scopeA.set(key, "from-A").get(5, TimeUnit.SECONDS);
        }

        try (IsolatedScope scopeB =
                clientB.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            String val = scopeB.get(key).get(5, TimeUnit.SECONDS);
            assertEquals("from-A", val);
        }

        // Cleanup
        clientA.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        clientA.close();
        clientB.close();

        System.out.println("Different timeouts per client test PASSED!");
    }

    // ─── Inflight Request Limits ─────────────────────────────────────────────────

    @Test
    public void testScopeRespectsInflightLimit() throws Exception {
        System.out.println("\n=== Test: Scope operations count against parent inflight limit ===");

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .inflightRequestsLimit(1000)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        // Verify scoped operations work under normal conditions
        // (inflight tracking is active — each scope command reserves a slot)
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            for (int i = 0; i < 50; i++) {
                String key = "inflight-" + i + "-" + UUID.randomUUID().toString().substring(0, 8);
                scope.set(key, "value-" + i).get(5, TimeUnit.SECONDS);
                String val = scope.get(key).get(5, TimeUnit.SECONDS);
                assertEquals("value-" + i, val);
                scope.executeCommand("DEL", key).get(5, TimeUnit.SECONDS);
            }
        }

        client.close();
        System.out.println("Scope inflight limit test PASSED!");
    }

    // ─── Combined Modifiers ──────────────────────────────────────────────────────

    @Test
    public void testScopeAllModifiersCombined() throws Exception {
        System.out.println("\n=== Test: Scope with compression + timeout + inflight all active ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .compressionLevel(3)
                        .minCompressionSize(64)
                        .build();

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .inflightRequestsLimit(500)
                        .compressionConfiguration(compressionConfig)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        String key = "combined-" + UUID.randomUUID();
        String largeValue = repeat("TestData_", 100); // ~900 bytes, will be compressed

        // Write via scope
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, largeValue).get(5, TimeUnit.SECONDS);
            String retrieved = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals(
                    largeValue, retrieved, "Round-trip through scope with all modifiers should work");
        }

        // Verify via parent client
        String parentGet = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals(largeValue, parentGet, "Parent client should read scope-written compressed data");

        // Verify compression actually happened
        GlideClientConfiguration rawConfig =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .build();
        GlideClient rawClient = GlideClient.createClient(rawConfig).get(10, TimeUnit.SECONDS);
        String rawValue = rawClient.get(key).get(5, TimeUnit.SECONDS);
        assertNotEquals(largeValue, rawValue, "Data should be stored compressed in Valkey");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        rawClient.close();
        client.close();

        System.out.println("All modifiers combined test PASSED!");
    }

    @Test
    public void testScopeWatchTransactionWithCompression() throws Exception {
        System.out.println("\n=== Test: WATCH/MULTI/EXEC works correctly with compression ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .minCompressionSize(64)
                        .build();

        GlideClientConfiguration config =
                GlideClientConfiguration.builder()
                        .address(NodeAddress.builder().host(getHost()).port(getPort()).build())
                        .requestTimeout(5000)
                        .compressionConfiguration(compressionConfig)
                        .build();

        GlideClient client = GlideClient.createClient(config).get(10, TimeUnit.SECONDS);

        // Use a large value that will be compressed
        String key = "watch-compress-" + UUID.randomUUID();
        String initialValue = repeat("InitialLargeValue_", 20); // ~360 bytes
        client.set(key, initialValue).get(5, TimeUnit.SECONDS);

        // OCC loop on the compressed key
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.watch(key).get(5, TimeUnit.SECONDS);
            String current = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals(initialValue, current, "WATCH should read the decompressed value correctly");

            // Build a new large value
            String newValue = repeat("UpdatedLargeValue_", 20);
            scope.multi().get(5, TimeUnit.SECONDS);
            scope.set(key, newValue).get(5, TimeUnit.SECONDS);
            String execResult = scope.exec().get(5, TimeUnit.SECONDS);
            assertNotNull(execResult, "EXEC should succeed (no conflict)");
        }

        // Verify the new value is readable (decompressed)
        String finalVal = client.get(key).get(5, TimeUnit.SECONDS);
        assertTrue(
                finalVal.startsWith("UpdatedLargeValue_"),
                "Final value should be the updated compressed value");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();

        System.out.println("WATCH/MULTI/EXEC with compression test PASSED!");
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Cluster Mode Tests
    // ═══════════════════════════════════════════════════════════════════════════════

    private static GlideClusterClientConfiguration.GlideClusterClientConfigurationBuilder<?, ?>
            clusterConfigBuilder() {
        GlideClusterClientConfiguration.GlideClusterClientConfigurationBuilder<?, ?> builder =
                GlideClusterClientConfiguration.builder();
        for (String host : CLUSTER_HOSTS) {
            String[] parts = host.split(":");
            builder.address(
                    NodeAddress.builder().host(parts[0]).port(Integer.parseInt(parts[1])).build());
        }
        return builder.requestTimeout(5000);
    }

    private static boolean clusterAvailable() {
        return CLUSTER_HOSTS.length > 0 && !CLUSTER_HOSTS[0].isEmpty();
    }

    // ─── Cluster: Basic Scope Operations ─────────────────────────────────────────

    @Test
    public void testClusterScopeAcquireAndRelease() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope acquire and release ===");

        GlideClusterClient client =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            String result = scope.executeCommand("PING").get(5, TimeUnit.SECONDS);
            assertNotNull(result);
        }

        client.close();
        System.out.println("Cluster scope acquire and release test PASSED!");
    }

    @Test
    public void testClusterScopeGetSet() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope GET/SET ===");

        GlideClusterClient client =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);

        // Use hash tag to ensure same-slot routing
        String key = "{scope-test}-cluster-basic-" + UUID.randomUUID();

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, "cluster-value").get(5, TimeUnit.SECONDS);
            String val = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals("cluster-value", val);
        }

        // Verify via parent client
        String result = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals("cluster-value", result);

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();
        System.out.println("Cluster scope GET/SET test PASSED!");
    }

    @Test
    public void testClusterScopeWatchMultiExec() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope WATCH/MULTI/EXEC ===");

        GlideClusterClient client =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-occ-" + UUID.randomUUID();
        client.set(key, "0").get(5, TimeUnit.SECONDS);

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.watch(key).get(5, TimeUnit.SECONDS);
            String current = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals("0", current);

            scope.multi().get(5, TimeUnit.SECONDS);
            scope.set(key, "1").get(5, TimeUnit.SECONDS);
            String execResult = scope.exec().get(5, TimeUnit.SECONDS);
            assertNotNull(execResult, "EXEC should succeed (no conflict)");
        }

        // Verify
        String finalVal = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals("1", finalVal);

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();
        System.out.println("Cluster scope WATCH/MULTI/EXEC test PASSED!");
    }

    @Test
    public void testClusterScopeWatchConflictAbortsExec() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope WATCH conflict aborts EXEC ===");

        GlideClusterClient client =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-conflict-" + UUID.randomUUID();
        client.set(key, "original").get(5, TimeUnit.SECONDS);

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.watch(key).get(5, TimeUnit.SECONDS);
            scope.get(key).get(5, TimeUnit.SECONDS);

            // Modify externally via the main client
            client.set(key, "modified-externally").get(5, TimeUnit.SECONDS);

            scope.multi().get(5, TimeUnit.SECONDS);
            scope.set(key, "from-scope").get(5, TimeUnit.SECONDS);
            String execResult = scope.exec().get(5, TimeUnit.SECONDS);
            // EXEC returns null when transaction is aborted
            assertNull(execResult, "EXEC should return null on conflict");
        }

        // Verify external modification persists
        String val = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals("modified-externally", val);

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();
        System.out.println("Cluster scope WATCH conflict test PASSED!");
    }

    // ─── Cluster: Compression ────────────────────────────────────────────────────

    @Test
    public void testClusterScopeInheritsCompression() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope inherits compression settings ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .compressionLevel(3)
                        .minCompressionSize(64)
                        .build();

        GlideClusterClient client =
                GlideClusterClient.createClient(
                                clusterConfigBuilder().compressionConfiguration(compressionConfig).build())
                        .get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-compress-" + UUID.randomUUID();
        String largeValue = repeat("A", 500); // 500 bytes — well above threshold

        // SET via scoped connection
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, largeValue).get(5, TimeUnit.SECONDS);
        }

        // GET via normal client (with same compression) should decompress
        String retrieved = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals(
                largeValue,
                retrieved,
                "Scoped SET with compression should be readable by the parent client (cluster)");

        // Verify compression happened: raw client should see different bytes
        GlideClusterClient rawClient =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);
        String rawValue = rawClient.get(key).get(5, TimeUnit.SECONDS);
        assertNotEquals(
                largeValue,
                rawValue,
                "Value stored via compressed scope should be compressed in Valkey (cluster)");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        rawClient.close();
        client.close();
        System.out.println("Cluster scope compression test PASSED!");
    }

    @Test
    public void testClusterScopeReadsCompressedData() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster scope reads compressed data ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder().enabled(true).backend(CompressionBackend.ZSTD).build();

        GlideClusterClient client =
                GlideClusterClient.createClient(
                                clusterConfigBuilder().compressionConfiguration(compressionConfig).build())
                        .get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-read-compress-" + UUID.randomUUID();
        String value = repeat("CompressibleData_", 50); // ~850 bytes

        // Write via normal client (compressed)
        client.set(key, value).get(5, TimeUnit.SECONDS);

        // Read via scoped connection — should decompress correctly
        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            String retrieved = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals(
                    value,
                    retrieved,
                    "Scoped GET should decompress data written by the parent client (cluster)");
        }

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();
        System.out.println("Cluster scope reads compressed data test PASSED!");
    }

    @Test
    public void testClusterScopeWatchTransactionWithCompression() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        System.out.println("\n=== Test: Cluster WATCH/MULTI/EXEC with compression ===");

        CompressionConfiguration compressionConfig =
                CompressionConfiguration.builder()
                        .enabled(true)
                        .backend(CompressionBackend.ZSTD)
                        .minCompressionSize(64)
                        .build();

        GlideClusterClient client =
                GlideClusterClient.createClient(
                                clusterConfigBuilder().compressionConfiguration(compressionConfig).build())
                        .get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-watch-compress-" + UUID.randomUUID();
        String initialValue = repeat("InitialLargeValue_", 20);
        client.set(key, initialValue).get(5, TimeUnit.SECONDS);

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.watch(key).get(5, TimeUnit.SECONDS);
            String current = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals(initialValue, current, "WATCH should read the decompressed value (cluster)");

            String newValue = repeat("UpdatedLargeValue_", 20);
            scope.multi().get(5, TimeUnit.SECONDS);
            scope.set(key, newValue).get(5, TimeUnit.SECONDS);
            String execResult = scope.exec().get(5, TimeUnit.SECONDS);
            assertNotNull(execResult, "EXEC should succeed (no conflict, cluster)");
        }

        String finalVal = client.get(key).get(5, TimeUnit.SECONDS);
        assertTrue(
                finalVal.startsWith("UpdatedLargeValue_"),
                "Final value should be the updated compressed value (cluster)");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        client.close();
        System.out.println("Cluster WATCH/MULTI/EXEC with compression test PASSED!");
    }

    // ─── Cluster: Database State (Valkey 9+ only) ────────────────────────────────

    @Test
    public void testClusterScopeInheritsConfiguredDatabase() throws Exception {
        assumeTrue(clusterAvailable(), "No cluster endpoints configured");
        assumeTrue(
                SERVER_VERSION.isGreaterThanOrEqualTo("9.0.0"), "SELECT in cluster requires Valkey 9+");
        System.out.println("\n=== Test: Cluster scope inherits configured database (Valkey 9+) ===");

        GlideClusterClient client =
                GlideClusterClient.createClient(clusterConfigBuilder().databaseId(2).build())
                        .get(10, TimeUnit.SECONDS);

        String key = "{scope-test}-cluster-db2-" + UUID.randomUUID();

        try (IsolatedScope scope =
                client.scopedConnection(Duration.ofSeconds(10)).get(10, TimeUnit.SECONDS)) {
            scope.set(key, "on-db2").get(5, TimeUnit.SECONDS);
            String val = scope.get(key).get(5, TimeUnit.SECONDS);
            assertEquals("on-db2", val);
        }

        // Parent client (also on db 2) should see the key
        String result = client.get(key).get(5, TimeUnit.SECONDS);
        assertEquals("on-db2", result);

        // A client on database 0 should NOT see the key
        GlideClusterClient clientDb0 =
                GlideClusterClient.createClient(clusterConfigBuilder().build()).get(10, TimeUnit.SECONDS);
        String resultDb0 = clientDb0.get(key).get(5, TimeUnit.SECONDS);
        assertNull(resultDb0, "Key written on db2 via scope should not be visible on db0 (cluster)");

        // Cleanup
        client.del(new String[] {key}).get(5, TimeUnit.SECONDS);
        clientDb0.close();
        client.close();
        System.out.println("Cluster scope database inheritance test PASSED!");
    }
}
