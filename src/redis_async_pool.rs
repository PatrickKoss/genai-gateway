use std::borrow::Cow;
use std::{
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

use deadpool::managed::{Manager, Metrics, RecycleError, RecycleResult};
use rand::Rng;
use redis::aio::ConnectionLike;
use redis::{AsyncCommands, Cmd, Pipeline, RedisError, RedisFuture, Value};

pub enum Ttl {
    Simple(Duration),
    Fuzzy { min: Duration, fuzz: Duration },
    Once,
}

pub struct RedisConnectionManager {
    client: redis::Client,
    check_on_recycle: bool,
    connection_ttl: Option<Ttl>,
}

impl RedisConnectionManager {
    pub fn new(client: redis::Client, check_on_recycle: bool, connection_ttl: Option<Ttl>) -> Self {
        Self {
            client,
            check_on_recycle,
            connection_ttl,
        }
    }
}

impl Manager for RedisConnectionManager {
    type Type = RedisConnection;
    type Error = RedisError;

    async fn create(&self) -> Result<RedisConnection, RedisError> {
        Ok(RedisConnection {
            actual: self.client.get_multiplexed_async_connection().await?,
            expires_at: self
                .connection_ttl
                .as_ref()
                .map(|max_duration| match max_duration {
                    Ttl::Simple(ttl) => Instant::now() + *ttl,
                    Ttl::Fuzzy { min, fuzz } => {
                        Instant::now()
                            + *min
                            + Duration::from_secs_f64(
                                rand::thread_rng().gen_range(0.0, fuzz.as_secs_f64()),
                            )
                    }
                    // already expired ;)
                    Ttl::Once => Instant::now(),
                }),
        })
    }

    async fn recycle(
        &self,
        mut conn: &mut Self::Type,
        _metrics: &Metrics,
    ) -> RecycleResult<Self::Error> {
        if self.check_on_recycle {
            let _r: bool = conn.exists(b"key").await?;
        }

        match &conn.expires_at {
            // check if connection is expired
            Some(expires_at) => {
                if &Instant::now() >= expires_at {
                    Err(RecycleError::Message(Cow::from("Connection expired")))
                } else {
                    Ok(())
                }
            }
            // no expire on connections
            None => Ok(()),
        }
    }

    fn detach(&self, _obj: &mut Self::Type) {
        // No action needed as the manager does not hold references to the objects
    }
}

/// The connection created by the pool manager.
///
/// It is Deref & DerefMut to `redis::aio::MultiplexedConnection` so it can be used
/// like a regular Redis asynchronous connection.
pub struct RedisConnection {
    actual: redis::aio::MultiplexedConnection,
    expires_at: Option<Instant>,
}

impl Deref for RedisConnection {
    type Target = redis::aio::MultiplexedConnection;
    fn deref(&self) -> &Self::Target {
        &self.actual
    }
}

impl DerefMut for RedisConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.actual
    }
}

impl AsMut<redis::aio::MultiplexedConnection> for RedisConnection {
    fn as_mut(&mut self) -> &mut redis::aio::MultiplexedConnection {
        &mut self.actual
    }
}

impl AsRef<redis::aio::MultiplexedConnection> for RedisConnection {
    fn as_ref(&self) -> &redis::aio::MultiplexedConnection {
        &self.actual
    }
}

impl ConnectionLike for &mut RedisConnection {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, Value> {
        (**self).req_packed_command(cmd)
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        cmd: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> RedisFuture<'a, Vec<Value>> {
        (**self).req_packed_commands(cmd, offset, count)
    }

    fn get_db(&self) -> i64 {
        (**self).get_db()
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
    use std::ops::DerefMut;

    use deadpool::managed::{Pool, PoolConfig};
    use redis::{AsyncCommands, Client};
    use testcontainers::{
        core::{IntoContainerPort, WaitFor},
        runners::AsyncRunner,
        GenericImage, ImageExt,
    };

    use crate::redis_async_pool::RedisConnectionManager;

    #[tokio::test]
    async fn test_ratelimiting() {
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
        let connection_pool: Pool<RedisConnectionManager> =
            Pool::builder(RedisConnectionManager::new(client, true, None))
                .config(pool_config)
                .max_size(5)
                .build()
                .expect("Failed to create Redis pool");

        let mut conn = connection_pool
            .get()
            .await
            .expect("Failed to get connection");
        let _: Option<u64> = conn
            .incr("async_pool_test_key", 1)
            .await
            .expect("Failed to increment key");
        let curr: Option<u64> = conn
            .get("async_pool_test_key")
            .await
            .expect("Failed to get key");
        assert_eq!(curr, Some(1));

        let mut redis_conn = conn.deref_mut();

        let (previous_count, current_count): (Option<u64>, Option<u64>) = redis::pipe()
            .atomic()
            .get("async_pool_test_key")
            .incr("async_pool_test_key", 1)
            .expire("async_pool_test_key", (1 * 2) as i64)
            .ignore()
            .query_async(&mut redis_conn)
            .await
            .expect("Failed to execute pipeline");

        assert_eq!(previous_count, Some(1));
        assert_eq!(current_count, Some(2));

        container
            .stop()
            .await
            .expect("Failed to stop Redis container");
    }

    fn find_free_port() -> u16 {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.local_addr().unwrap().port()
    }
}
