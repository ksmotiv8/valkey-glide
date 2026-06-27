// Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

package integTest

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	glide "github.com/valkey-io/valkey-glide/go/v2"
	"github.com/valkey-io/valkey-glide/go/v2/config"
)

func standaloneConfig() *config.ClientConfiguration {
	host := "localhost"
	port := 6379
	// Use the --standalone-endpoints flag provided by CI (same flag as test suite)
	if standaloneHosts != nil && *standaloneHosts != "" {
		parts := strings.SplitN(*standaloneHosts, ",", 2)
		hostPort := strings.SplitN(parts[0], ":", 2)
		if len(hostPort) == 2 {
			host = hostPort[0]
			if p, err := strconv.Atoi(hostPort[1]); err == nil {
				port = p
			}
		}
	}
	return config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(5000 * time.Millisecond)
}

// skipIfNoStandaloneEndpoints skips the test if no --standalone-endpoints flag
// was provided. This prevents pool/scope tests from running in specialized
// CI targets (long-timeout-test, pubsub-test) that don't provision a standalone server.
func skipIfNoStandaloneEndpoints(t *testing.T) {
	t.Helper()
	if standaloneHosts == nil || *standaloneHosts == "" {
		t.Skip("No --standalone-endpoints provided; skipping pool/scope test")
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 1: ClientPool tests
// ═══════════════════════════════════════════════════════════════════════════════

func TestPoolCreateAndMetrics(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        3,
		MinIdle:        2,
		AcquireTimeout: 10 * time.Second,
	})
	require.NoError(t, err)
	defer pool.Close()

	// Wait for min_idle warmup
	time.Sleep(3 * time.Second)

	assert.GreaterOrEqual(t, pool.IdleCount(), 1)
	assert.GreaterOrEqual(t, pool.TotalCount(), 1)
}

func TestPoolAcquireAndCommands(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        3,
		MinIdle:        1,
		AcquireTimeout: 10 * time.Second,
	})
	require.NoError(t, err)
	defer pool.Close()

	time.Sleep(3 * time.Second)

	ctx := context.Background()
	clientID, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, clientID, int64(0))

	client, err := pool.GetClient(clientID)
	require.NoError(t, err)

	// Execute commands via the PooledClient wrapper
	key := fmt.Sprintf("go-pool-test-%d", time.Now().UnixNano())
	_, err = client.Set(ctx, key, "hello")
	require.NoError(t, err)

	val, err := client.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "hello", val.Value())

	client.Client.Del(ctx, []string{key})

	// Close() on PooledClient releases back to pool (doesn't destroy connection)
	client.Close()
}

func TestPoolReuse(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        3,
		MinIdle:        1,
		AcquireTimeout: 10 * time.Second,
	})
	require.NoError(t, err)
	defer pool.Close()

	time.Sleep(3 * time.Second)

	ctx := context.Background()
	id1, _ := pool.Acquire(ctx)
	pool.Release(id1)
	time.Sleep(100 * time.Millisecond)

	id2, _ := pool.Acquire(ctx)
	pool.Release(id2)

	// LIFO: same client_id
	assert.Equal(t, id1, id2)
}

func TestPoolExhaustionTimeout(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        1,
		MinIdle:        1,
		AcquireTimeout: 10 * time.Second,
	})
	require.NoError(t, err)
	defer pool.Close()

	time.Sleep(3 * time.Second)

	ctx := context.Background()
	id1, _ := pool.Acquire(ctx)

	// Second acquire should timeout
	_, err = pool.AcquireWithTimeout(ctx, 500*time.Millisecond)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "timed out")

	pool.Release(id1)
}

func TestPoolConcurrentAccess(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        4,
		MinIdle:        4,
		AcquireTimeout: 15 * time.Second,
	})
	require.NoError(t, err)
	defer pool.Close()

	time.Sleep(4 * time.Second)

	ctx := context.Background()
	var wg sync.WaitGroup
	errCh := make(chan error, 8)

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			clientID, err := pool.Acquire(ctx)
			if err != nil {
				errCh <- err
				return
			}

			client, err := pool.GetClient(clientID)
			if err != nil {
				errCh <- err
				return
			}

			key := fmt.Sprintf("go-pool-concurrent-%d", idx)
			_, err = client.Set(ctx, key, fmt.Sprintf("val-%d", idx))
			if err != nil {
				errCh <- err
				return
			}
			client.Client.Del(ctx, []string{key})
			client.Close() // returns to pool
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent access error: %v", err)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 2: IsolatedScope tests
// ═══════════════════════════════════════════════════════════════════════════════

func TestScopeAcquirePingRelease(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	assert.False(t, scope.IsReleased())

	result, err := scope.Ping(ctx)
	require.NoError(t, err)
	assert.Equal(t, "PONG", result)

	scope.Close()
	assert.True(t, scope.IsReleased())
}

func TestScopeGetSet(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-scope-test-%d", time.Now().UnixNano())

	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	defer scope.Close()

	_, err = scope.Set(ctx, key, "hello")
	require.NoError(t, err)

	val, err := scope.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "hello", val)

	// Cleanup via main client
	client.Del(ctx, []string{key})
}

func TestScopeReturnsEmptyForMissingKey(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	defer scope.Close()

	val, err := scope.Get(ctx, fmt.Sprintf("nonexistent-%d", time.Now().UnixNano()))
	require.NoError(t, err)
	assert.Equal(t, "", val) // Nil response → empty string
}

func TestScopeWatchMultiExec(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-occ-%d", time.Now().UnixNano())
	client.Set(ctx, key, "0")

	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	defer scope.Close()

	_, err = scope.Watch(ctx, key)
	require.NoError(t, err)

	val, err := scope.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "0", val)

	_, err = scope.Multi(ctx)
	require.NoError(t, err)

	_, err = scope.Set(ctx, key, "1")
	require.NoError(t, err)

	result, err := scope.Exec(ctx)
	require.NoError(t, err)
	assert.NotEmpty(t, result) // Non-empty means success

	// Verify
	storedVal, _ := client.Get(ctx, key)
	assert.Equal(t, "1", storedVal.Value())
	client.Del(ctx, []string{key})
}

func TestScopeRaisesAfterRelease(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)

	scope.Close()

	_, err = scope.Ping(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "released")
}

func TestScopeWatchConflictAbortsExec(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-occ-conflict-%d", time.Now().UnixNano())
	client.Set(ctx, key, "original")

	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	defer scope.Close()

	_, err = scope.Watch(ctx, key)
	require.NoError(t, err)
	_, err = scope.Get(ctx, key)
	require.NoError(t, err)

	// Modify externally via the main client
	client.Set(ctx, key, "modified-externally")

	_, err = scope.Multi(ctx)
	require.NoError(t, err)
	_, err = scope.Set(ctx, key, "from-scope")
	require.NoError(t, err)

	result, err := scope.Exec(ctx)
	require.NoError(t, err)
	// EXEC returns empty string when transaction is aborted (nil response)
	assert.Empty(t, result)

	// Verify external modification persists
	val, _ := client.Get(ctx, key)
	assert.Equal(t, "modified-externally", val.Value())
	client.Del(ctx, []string{key})
}

func TestScopeOCCConcurrentIncrement(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-occ-counter-%d", time.Now().UnixNano())
	client.Set(ctx, key, "0")

	numGoroutines := 4
	incrementsPerGoroutine := 10
	expectedFinal := numGoroutines * incrementsPerGoroutine

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < incrementsPerGoroutine; i++ {
				committed := false
				for !committed {
					scope, err := client.ScopedConnection(ctx, 10*time.Second)
					if err != nil {
						errCh <- err
						return
					}
					scope.Watch(ctx, key)
					val, _ := scope.Get(ctx, key)
					current := 0
					if val != "" {
						fmt.Sscanf(val, "%d", &current)
					}
					scope.Multi(ctx)
					scope.Set(ctx, key, fmt.Sprintf("%d", current+1))
					result, _ := scope.Exec(ctx)
					scope.Close()
					if result != "" {
						committed = true
					}
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("goroutine error: %v", err)
	}

	finalVal, _ := client.Get(ctx, key)
	assert.Equal(t, fmt.Sprintf("%d", expectedFinal), finalVal.Value())
	client.Del(ctx, []string{key})
}

func TestScopeCloseIsIdempotent(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)

	scope.Ping(ctx)
	scope.Close()
	scope.Close() // Should not panic
	assert.True(t, scope.IsReleased())
}

func TestScopePoolReuse(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()

	scope1, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	scope1.Ping(ctx)
	scope1.Close()

	time.Sleep(100 * time.Millisecond)

	// Should be able to acquire again (connection reused)
	scope2, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	result, err := scope2.Ping(ctx)
	require.NoError(t, err)
	assert.Equal(t, "PONG", result)
	scope2.Close()
}

func TestPoolCloseRejectsAcquire(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	pool, err := glide.NewClientPool(standaloneConfig(), glide.PoolConfig{
		MaxSize:        2,
		MinIdle:        1,
		AcquireTimeout: 10 * time.Second,
	})
	require.NoError(t, err)

	time.Sleep(3 * time.Second)
	pool.Close()

	ctx := context.Background()
	_, err = pool.Acquire(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scope Connection Modifier Parity Tests
// ═══════════════════════════════════════════════════════════════════════════════

func compressedConfig() *config.ClientConfiguration {
	compressionConfig := config.NewCompressionConfiguration().
		WithBackend(config.ZSTD).
		WithMinCompressionSize(64)
	cfg := standaloneConfig()
	return cfg.WithCompressionConfiguration(compressionConfig)
}

func TestScopeCompressionWritesParity(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	// Client with compression
	compressedClient, err := glide.NewClient(compressedConfig())
	require.NoError(t, err)
	defer compressedClient.Close()

	// Client without compression (raw)
	rawClient, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer rawClient.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-scope-compress-%d", time.Now().UnixNano())
	largeValue := ""
	for i := 0; i < 500; i++ {
		largeValue += "A"
	}

	// Write via scoped connection (should compress)
	scope, err := compressedClient.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	_, err = scope.Set(ctx, key, largeValue)
	require.NoError(t, err)
	scope.Close()

	// Read with same client (decompresses) — should match
	val, err := compressedClient.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, largeValue, val.Value())

	// Read with raw client (no compression) — should differ
	rawVal, err := rawClient.Get(ctx, key)
	require.NoError(t, err)
	assert.NotEqual(t, largeValue, rawVal.Value(),
		"Value stored via compressed scope should be compressed in Valkey")

	compressedClient.Del(ctx, []string{key})
}

func TestScopeCompressionReadsParity(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	compressedClient, err := glide.NewClient(compressedConfig())
	require.NoError(t, err)
	defer compressedClient.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-scope-read-compress-%d", time.Now().UnixNano())
	value := ""
	for i := 0; i < 50; i++ {
		value += "CompressibleData_"
	}

	// Write via parent client (compressed)
	compressedClient.Set(ctx, key, value)

	// Read via scoped connection — should decompress correctly
	scope, err := compressedClient.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	retrieved, err := scope.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, value, retrieved)
	scope.Close()

	compressedClient.Del(ctx, []string{key})
}

func TestScopeDatabaseInheritance(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	// Client configured for database 2
	cfg := standaloneConfig().WithDatabaseId(2)

	client, err := glide.NewClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-scope-db2-%d", time.Now().UnixNano())

	// Write via scope on database 2
	scope, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	_, err = scope.Set(ctx, key, "on-db2")
	require.NoError(t, err)
	val, err := scope.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "on-db2", val)
	scope.Close()

	// Parent client (also on db 2) should see the key
	parentVal, err := client.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "on-db2", parentVal.Value())

	// A client on database 0 should NOT see the key
	db0Client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer db0Client.Close()
	db0Val, err := db0Client.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "", db0Val.Value(), "Key on db2 should not be visible on db0")

	client.Del(ctx, []string{key})
}

func TestScopeReleaseResetsDatabase(t *testing.T) {
	skipIfNoStandaloneEndpoints(t)
	client, err := glide.NewClient(standaloneConfig())
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("go-scope-db-reset-%d", time.Now().UnixNano())

	// First scope: SELECT to db 4 and write
	scope1, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	_, err = scope1.ExecuteCommand(ctx, "SELECT", "4")
	require.NoError(t, err)
	_, err = scope1.Set(ctx, key, "on-db4")
	require.NoError(t, err)
	scope1.Close()

	// Allow async cleanup
	time.Sleep(300 * time.Millisecond)

	// Second scope: should be on db 0 (reset happened)
	scope2, err := client.ScopedConnection(ctx, 10*time.Second)
	require.NoError(t, err)
	val, err := scope2.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "", val, "Scope release should reset database — second scope should be on db 0")
	scope2.Close()

	// Cleanup key on db 4
	cleanupCfg := standaloneConfig().WithDatabaseId(4)
	cleanupClient, _ := glide.NewClient(cleanupCfg)
	if cleanupClient != nil {
		cleanupClient.Del(ctx, []string{key})
		cleanupClient.Close()
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cluster Mode Tests
// ═══════════════════════════════════════════════════════════════════════════════

func clusterConfig() *config.ClusterClientConfiguration {
	host := "localhost"
	port := 7000
	// Use the --cluster-endpoints flag provided by CI
	if clusterHosts != nil && *clusterHosts != "" {
		parts := strings.SplitN(*clusterHosts, ",", 2)
		hostPort := strings.SplitN(parts[0], ":", 2)
		if len(hostPort) == 2 {
			host = hostPort[0]
			if p, err := strconv.Atoi(hostPort[1]); err == nil {
				port = p
			}
		}
	}
	return config.NewClusterClientConfiguration().
		WithAddress(&config.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(5000 * time.Millisecond)
}

func clusterAvailable() bool {
	return clusterHosts != nil && *clusterHosts != ""
}

func compressedClusterConfig() *config.ClusterClientConfiguration {
	compressionConfig := config.NewCompressionConfiguration().
		WithBackend(config.ZSTD).
		WithMinCompressionSize(64)
	cfg := clusterConfig()
	return cfg.WithCompressionConfiguration(compressionConfig)
}

func getClusterServerVersion(t *testing.T, client *glide.ClusterClient) string {
	ctx := context.Background()
	info, err := client.CustomCommand(ctx, []string{"INFO", "SERVER"})
	if err != nil {
		t.Skipf("Cannot get server version: %v", err)
		return "0.0.0"
	}
	// INFO returns multi-value (one per primary node); parse the first one
	var infoStr string
	if info.IsSingleValue() {
		infoStr = fmt.Sprintf("%v", info.SingleValue())
	} else {
		for _, v := range info.MultiValue() {
			infoStr = fmt.Sprintf("%v", v)
			break
		}
	}
	for _, line := range strings.Split(infoStr, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "valkey_version:") {
			return strings.TrimPrefix(line, "valkey_version:")
		}
	}
	for _, line := range strings.Split(infoStr, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "redis_version:") {
			return strings.TrimPrefix(line, "redis_version:")
		}
	}
	return "0.0.0"
}

func versionLessThan(ver string, target string) bool {
	verParts := strings.Split(ver, ".")
	targetParts := strings.Split(target, ".")
	for i := 0; i < len(targetParts) && i < len(verParts); i++ {
		v, _ := strconv.Atoi(verParts[i])
		tv, _ := strconv.Atoi(targetParts[i])
		if v < tv {
			return true
		}
		if v > tv {
			return false
		}
	}
	return false
}

func TestClusterScopeCompressionWritesParity(t *testing.T) {
	if !clusterAvailable() {
		t.Skip("No cluster endpoints configured")
	}

	// Client with compression
	compressedClient, err := glide.NewClusterClient(compressedClusterConfig())
	require.NoError(t, err)
	defer compressedClient.Close()

	// Client without compression (raw)
	rawClient, err := glide.NewClusterClient(clusterConfig())
	require.NoError(t, err)
	defer rawClient.Close()

	ctx := context.Background()
	key := fmt.Sprintf("{scope-test}-go-cluster-compress-%d", time.Now().UnixNano())
	largeValue := ""
	for i := 0; i < 500; i++ {
		largeValue += "A"
	}

	// Write with compression
	_, err = compressedClient.Set(ctx, key, largeValue)
	require.NoError(t, err)

	// Read with same client (decompresses) — should match
	val, err := compressedClient.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, largeValue, val.Value())

	// Read with raw client (no compression) — should differ
	rawVal, err := rawClient.Get(ctx, key)
	require.NoError(t, err)
	assert.NotEqual(t, largeValue, rawVal.Value(),
		"Value stored via compressed client should be compressed in Valkey (cluster)")

	compressedClient.Del(ctx, []string{key})
}

func TestClusterScopeCompressionReadsParity(t *testing.T) {
	if !clusterAvailable() {
		t.Skip("No cluster endpoints configured")
	}

	compressedClient, err := glide.NewClusterClient(compressedClusterConfig())
	require.NoError(t, err)
	defer compressedClient.Close()

	ctx := context.Background()
	key := fmt.Sprintf("{scope-test}-go-cluster-read-compress-%d", time.Now().UnixNano())
	value := ""
	for i := 0; i < 50; i++ {
		value += "CompressibleData_"
	}

	// Write via client (compressed)
	compressedClient.Set(ctx, key, value)

	// Read back — should decompress correctly
	retrieved, err := compressedClient.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, value, retrieved.Value())

	compressedClient.Del(ctx, []string{key})
}

func TestClusterDatabaseInheritance(t *testing.T) {
	if !clusterAvailable() {
		t.Skip("No cluster endpoints configured")
	}

	// Create cluster client — check version first (SELECT requires Valkey 9+)
	checkClient, err := glide.NewClusterClient(clusterConfig())
	require.NoError(t, err)
	ver := getClusterServerVersion(t, checkClient)
	checkClient.Close()

	if versionLessThan(ver, "9.0.0") {
		t.Skipf("SELECT in cluster requires Valkey 9+ (got %s)", ver)
	}

	// Client configured for database 2
	cfg := clusterConfig().WithDatabaseId(2)
	client, err := glide.NewClusterClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx := context.Background()
	key := fmt.Sprintf("{scope-test}-go-cluster-db2-%d", time.Now().UnixNano())

	// Write on database 2
	_, err = client.Set(ctx, key, "on-db2")
	require.NoError(t, err)

	// Read back — should see the key
	val, err := client.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "on-db2", val.Value())

	// A client on database 0 should NOT see the key
	db0Client, err := glide.NewClusterClient(clusterConfig())
	require.NoError(t, err)
	defer db0Client.Close()
	db0Val, err := db0Client.Get(ctx, key)
	require.NoError(t, err)
	assert.Equal(t, "", db0Val.Value(), "Key on db2 should not be visible on db0 (cluster)")

	client.Del(ctx, []string{key})
}
