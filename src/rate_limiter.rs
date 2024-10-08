use std::ops::DerefMut;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;
use deadpool::managed::Pool;
use redis::AsyncCommands;

use crate::redis_async_pool::RedisConnectionManager;

const KEY_PREFIX: &str = "rate_limiter";

#[async_trait]
pub trait SlidingWindowRateLimiter {
    async fn record_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        tokens: u64,
        size: Duration,
    ) -> Result<u64>;

    async fn fetch_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        size: Duration,
    ) -> Result<u64>;
}

pub(crate) struct RedisSlidingWindowRateLimiter {
    connection_pool: Pool<RedisConnectionManager>,
}

#[async_trait]
impl SlidingWindowRateLimiter for RedisSlidingWindowRateLimiter {
    async fn record_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        tokens: u64,
        size: Duration,
    ) -> Result<u64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let size_secs = size.as_secs();

        if size_secs == 0 {
            return Ok(0);
        }

        let current_window = (now.as_secs() / size_secs) * size_secs;
        let previous_window = (now.as_secs() / size_secs) * size_secs - size_secs;
        let current_key = format!("{}:{}:{}:{}", KEY_PREFIX, resource, subject, current_window);
        let previous_key = format!(
            "{}:{}:{}:{}",
            KEY_PREFIX, resource, subject, previous_window
        );

        let mut connection = self.connection_pool.get().await?;
        let mut redis_conn = connection.deref_mut();

        let (previous_count, current_count): (Option<u64>, Option<u64>) = redis::pipe()
            .atomic()
            .get(&previous_key)
            .incr(&current_key, tokens)
            .expire(&current_key, (size_secs * 2) as i64)
            .ignore()
            .query_async(&mut redis_conn)
            .await?;

        Ok(Self::sliding_window_count(
            previous_count,
            current_count,
            now,
            size,
        ))
    }

    async fn fetch_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        size: Duration,
    ) -> Result<u64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let size_secs = size.as_secs();

        if size_secs == 0 {
            return Ok(0);
        }

        let current_window = (now.as_secs() / size_secs) * size_secs;
        let previous_window = (now.as_secs() / size_secs) * size_secs - size_secs;
        let current_key = format!("{}:{}:{}:{}", KEY_PREFIX, resource, subject, current_window);
        let previous_key = format!(
            "{}:{}:{}:{}",
            KEY_PREFIX, resource, subject, previous_window
        );

        let mut connection = self.connection_pool.get().await?;

        let (previous_count, current_count): (Option<u64>, Option<u64>) =
            connection.get(vec![previous_key, current_key]).await?;

        Ok(Self::sliding_window_count(
            previous_count,
            current_count,
            now,
            size,
        ))
    }
}

impl RedisSlidingWindowRateLimiter {
    pub fn new(connection_pool: Pool<RedisConnectionManager>) -> Self {
        RedisSlidingWindowRateLimiter { connection_pool }
    }

    pub fn sliding_window_count(
        previous: Option<u64>,
        current: Option<u64>,
        now: Duration,
        size: Duration,
    ) -> u64 {
        let current_window = (now.as_secs() / size.as_secs()) * size.as_secs();
        let next_window = current_window + size.as_secs();
        let weight = (Duration::from_secs(next_window).as_millis() - now.as_millis()) as f64
            / size.as_millis() as f64;
        current.unwrap_or(0) + (previous.unwrap_or(0) as f64 * weight).round() as u64
    }
}

pub struct DummySlidingWindowRateLimiter {}

#[async_trait]
impl SlidingWindowRateLimiter for DummySlidingWindowRateLimiter {
    async fn record_sliding_window(
        &self,
        _resource: &str,
        _subject: &str,
        _tokens: u64,
        _size: Duration,
    ) -> Result<u64> {
        Ok(0)
    }

    async fn fetch_sliding_window(
        &self,
        _resource: &str,
        _subject: &str,
        _size: Duration,
    ) -> Result<u64> {
        Ok(0)
    }
}

pub(crate) enum SlidingWindowRateLimiterEnum {
    Redis(RedisSlidingWindowRateLimiter),
    Dummy(DummySlidingWindowRateLimiter),
}

#[async_trait]
impl SlidingWindowRateLimiter for SlidingWindowRateLimiterEnum {
    async fn record_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        tokens: u64,
        size: Duration,
    ) -> Result<u64> {
        match self {
            SlidingWindowRateLimiterEnum::Redis(redis) => {
                redis
                    .record_sliding_window(resource, subject, tokens, size)
                    .await
            }
            SlidingWindowRateLimiterEnum::Dummy(dummy) => {
                dummy
                    .record_sliding_window(resource, subject, tokens, size)
                    .await
            }
        }
    }

    async fn fetch_sliding_window(
        &self,
        resource: &str,
        subject: &str,
        size: Duration,
    ) -> Result<u64> {
        match self {
            SlidingWindowRateLimiterEnum::Redis(redis) => {
                redis.fetch_sliding_window(resource, subject, size).await
            }
            SlidingWindowRateLimiterEnum::Dummy(dummy) => {
                dummy.fetch_sliding_window(resource, subject, size).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
    use std::time::Duration;

    use deadpool::managed::{Pool, PoolConfig};
    use redis::Client;
    use testcontainers::{
        core::{IntoContainerPort, WaitFor},
        runners::AsyncRunner,
        GenericImage, ImageExt,
    };

    use crate::rate_limiter;
    use crate::rate_limiter::{
        DummySlidingWindowRateLimiter, RedisSlidingWindowRateLimiter, SlidingWindowRateLimiter,
        SlidingWindowRateLimiterEnum,
    };
    use crate::redis_async_pool::RedisConnectionManager;

    #[tokio::test]
    async fn test_ratelimiting() {
        let tokens = 10;
        let size = Duration::from_secs(2);

        // setup redis with test containers
        let free_port = find_free_port();
        let container = GenericImage::new("redis", "7.2.4")
            .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
            .with_mapped_port(free_port, 6379.tcp())
            .start()
            .await
            .expect("Redis started");

        // init redis connection
        let redis_url = format!("redis://127.0.0.1:{}/0", free_port);
        let client = Client::open(redis_url).expect("Failed to create Redis client");

        let pool_config = PoolConfig::default();
        let connection_pool = Pool::builder(RedisConnectionManager::new(client, true, None))
            .config(pool_config)
            .max_size(5)
            .build()
            .expect("Failed to create Redis pool");

        // init rate limiter
        let rate_limiter = RedisSlidingWindowRateLimiter::new(connection_pool);

        // record value
        let count = rate_limiter
            .record_sliding_window("user", "test-user-1", tokens, size)
            .await
            .expect("Failed to record sliding window");
        assert_eq!(count, tokens);

        // fetch value
        let count = rate_limiter
            .fetch_sliding_window("user", "test-user-1", size)
            .await
            .expect("Failed to fetch sliding window");
        assert_eq!(count, tokens);

        // test divide by 0
        let count = rate_limiter
            .fetch_sliding_window("user", "test-user-1", Duration::from_millis(1))
            .await
            .expect("Failed to fetch sliding window");

        assert_eq!(count, 0);

        tokio::time::sleep(Duration::from_millis(2100)).await;

        // test reset of the window
        let count = rate_limiter
            .fetch_sliding_window("test", "subject", size)
            .await
            .expect("Failed to fetch sliding window");

        assert_eq!(count, 0);

        container
            .stop()
            .await
            .expect("Failed to stop Redis container");
    }

    #[tokio::test]
    async fn test_dummy_rate_limiter() {
        let dummy_rate_limiter = DummySlidingWindowRateLimiter {};
        let count = dummy_rate_limiter
            .record_sliding_window("user", "test-user-1", 10, Duration::from_secs(1))
            .await
            .expect("Failed to record sliding window");
        assert_eq!(count, 0);

        let count = dummy_rate_limiter
            .fetch_sliding_window("user", "test-user-1", Duration::from_secs(1))
            .await
            .expect("Failed to fetch sliding window");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_enum_dummy_rate_limiter() {
        let dummy_rate_limiter =
            SlidingWindowRateLimiterEnum::Dummy(rate_limiter::DummySlidingWindowRateLimiter {});
        let count = dummy_rate_limiter
            .record_sliding_window("user", "test-user-1", 10, Duration::from_secs(1))
            .await
            .expect("Failed to record sliding window");
        assert_eq!(count, 0);

        let count = dummy_rate_limiter
            .fetch_sliding_window("user", "test-user-1", Duration::from_secs(1))
            .await
            .expect("Failed to fetch sliding window");
        assert_eq!(count, 0);
    }

    fn find_free_port() -> u16 {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.local_addr().unwrap().port()
    }
}
