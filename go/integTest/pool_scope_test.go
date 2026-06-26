// Copyright Valkey GLIDE Project Contributors - SPDX Identifier: Apache-2.0

package integTest

import (
	"context"
	"fmt"
	"os"
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
	// CI provides standalone endpoint via env var
	if ep := getEnvStandaloneEndpoint(); ep != nil {
		host = ep.Host
		port = ep.Port
	}
	return config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(5000 * time.Millisecond)
}

func getEnvStandaloneEndpoint() *config.NodeAddress {
	endpoints := os.Getenv("GLIDE_STANDALONE_ENDPOINTS")
	if endpoints == "" {
		return nil
	}
	// Format: "host:port" or "host:port,host:port,..."
	parts := strings.SplitN(endpoints, ",", 2)
	hostPort := strings.SplitN(parts[0], ":", 2)
	if len(hostPort) != 2 {
		return nil
	}
	port := 6379
	if p, err := strconv.Atoi(hostPort[1]); err == nil {
		port = p
	}
	return &config.NodeAddress{Host: hostPort[0], Port: port}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 1: ClientPool tests
// ═══════════════════════════════════════════════════════════════════════════════

func TestPoolCreateAndMetrics(t *testing.T) {
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
	return config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: "localhost", Port: 6379}).
		WithRequestTimeout(5000 * time.Millisecond).
		WithCompressionConfiguration(compressionConfig)
}

func TestScopeCompressionWritesParity(t *testing.T) {
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
	// Client configured for database 2
	cfg := config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: "localhost", Port: 6379}).
		WithRequestTimeout(5000 * time.Millisecond).
		WithDatabaseId(2)

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
	cleanupCfg := config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{Host: "localhost", Port: 6379}).
		WithRequestTimeout(5000 * time.Millisecond).
		WithDatabaseId(4)
	cleanupClient, _ := glide.NewClient(cleanupCfg)
	if cleanupClient != nil {
		cleanupClient.Del(ctx, []string{key})
		cleanupClient.Close()
	}
}
