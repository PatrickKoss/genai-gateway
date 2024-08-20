use std::borrow::Cow;
use std::{
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

use deadpool::managed::{Manager, Metrics, RecycleError, RecycleResult};
use rand::Rng;
use redis::{AsyncCommands, RedisError};

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
        conn: &mut Self::Type,
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
