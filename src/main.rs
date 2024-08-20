use std::env;

use clap::Parser;
use deadpool::managed::{Pool, PoolConfig};
use pingora::prelude::*;
use redis::Client;
use tiktoken_rs::cl100k_base;

use crate::rate_limiter::SlidingWindowRateLimiterEnum;
use http_proxy::{HttpGateway, HttpGatewayConfig};

use crate::redis_async_pool::RedisConnectionManager;

mod http_proxy;
mod rate_limiter;
mod redis_async_pool;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = true)]
    openai_tls: bool,
    #[arg(long, default_value_t = 443)]
    openai_port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    openai_domain: String,
    #[arg(long, default_value = "6188")]
    http_proxy_port: String,
    #[arg(long, default_value = "6192")]
    http_proxy_metrics_port: String,
    #[arg(long, default_value_t = false)]
    enable_rate_limiting: bool,
    #[arg(long, default_value = "redis://127.0.0.1:6379/0")]
    rate_limiting_redis_connection_string: String,
    #[arg(long, default_value_t = 5)]
    rate_limiting_redis_pool_size: usize,
}

fn main() {
    let args = Args::parse();
    let openai_tls = env::var("OPENAI_TLS")
        .unwrap_or_else(|_| args.openai_tls.to_string())
        .parse::<bool>()
        .expect("Failed to parse OPENAI_TLS");
    let openai_port = env::var("OPENAI_PORT")
        .ok()
        .unwrap_or(args.openai_port.to_string())
        .parse::<u16>()
        .expect("Failed to parse OPENAI_PORT");
    let openai_domain = env::var("OPENAI_DOMAIN")
        .ok()
        .unwrap_or(args.openai_domain.to_string());
    let http_proxy_port = env::var("HTTP_PROXY_PORT")
        .ok()
        .unwrap_or(args.http_proxy_port.to_string());
    let http_proxy_metrics_port = env::var("HTTP_PROXY_METRICS_PORT")
        .ok()
        .unwrap_or(args.http_proxy_metrics_port.to_string());
    let enable_rate_limiting = env::var("ENABLE_RATE_LIMITING")
        .unwrap_or_else(|_| args.enable_rate_limiting.to_string())
        .parse::<bool>()
        .expect("Failed to parse ENABLE_RATE_LIMITING");
    let rate_limiting_redis_connection_string = env::var("RATE_LIMITING_REDIS_CONNECTION_STRING")
        .ok()
        .unwrap_or(args.rate_limiting_redis_connection_string.to_string());
    let rate_limiting_redis_pool_size = env::var("RATE_LIMITING_REDIS_POOL_SIZE")
        .ok()
        .unwrap_or(args.rate_limiting_redis_pool_size.to_string())
        .parse::<usize>()
        .expect("Failed to parse RATE_LIMITING_REDIS_POOL_SIZE");

    env_logger::init();

    let mut server = Server::new(None).unwrap();
    server.bootstrap();

    let tokenizer = cl100k_base().expect("Failed to load tokenizer");

    let rate_limiter = if enable_rate_limiting {
        let client = Client::open(rate_limiting_redis_connection_string)
            .expect("Failed to create Redis client");

        let pool_config = PoolConfig::default();
        let connection_pool = Pool::builder(RedisConnectionManager::new(client, true, None))
            .config(pool_config)
            .max_size(rate_limiting_redis_pool_size)
            .build()
            .expect("Failed to create Redis pool");

        SlidingWindowRateLimiterEnum::Redis(rate_limiter::RedisSlidingWindowRateLimiter::new(
            connection_pool,
        ))
    } else {
        SlidingWindowRateLimiterEnum::Dummy(rate_limiter::DummySlidingWindowRateLimiter {})
    };

    let mut http_proxy = http_proxy_service(
        &server.configuration,
        HttpGateway::new(HttpGatewayConfig {
            openai_tls,
            openai_port,
            openai_domain: openai_domain.leak(),
            tokenizer,
            sliding_window_rate_limiter: rate_limiter,
        })
        .expect("Failed to create http gateway"),
    );

    http_proxy.add_tcp(format!("0.0.0.0:{}", http_proxy_port).as_str());
    server.add_service(http_proxy);

    let mut prometheus_service_http =
        pingora_core::services::listening::Service::prometheus_http_service();
    prometheus_service_http.add_tcp(format!("0.0.0.0:{}", http_proxy_metrics_port).as_str());
    server.add_service(prometheus_service_http);

    server.run_forever();
}
