#![feature(duration_constructors)]

use std::env;

use clap::Parser;
use deadpool::managed::{Pool, PoolConfig};
use pingora::prelude::*;
use redis::Client;
use tiktoken_rs::cl100k_base;

use http_proxy::{HttpGateway, HttpGatewayConfig};

use crate::http_proxy::{OpenAIConfig, RateLimitingConfig};
use crate::rate_limiter::SlidingWindowRateLimiterEnum;
use crate::redis_async_pool::RedisConnectionManager;

mod http_proxy;
mod rate_limiter;
mod redis_async_pool;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        help = "Enable TLS for downstream OpenAI compatible endpoints",
        default_value_t = true
    )]
    openai_tls: bool,
    #[arg(
        long,
        help = "Port to use for downstream OpenAI compatible endpoints",
        default_value_t = 443
    )]
    openai_port: u16,
    #[arg(
        long,
        default_value = "0.0.0.0",
        help = "Domain to use for downstream OpenAI compatible endpoints"
    )]
    openai_domain: String,
    #[arg(long, default_value = "8080", help = "Port to use for HTTP proxy")]
    http_proxy_port: String,
    #[arg(long, default_value = "9090", help = "Port to use for HTTP proxy metrics")]
    http_proxy_metrics_port: String,
    #[arg(long, default_value_t = false, help = "Enable rate limiting on user key")]
    enable_rate_limiting: bool,
    #[arg(long, default_value = "redis://127.0.0.1:6379/0", help = "Redis connection string for the rate limiter")]
    rate_limiting_redis_connection_string: String,
    #[arg(long, default_value_t = 5, help = "Redis pool size for the rate limiter")]
    rate_limiting_redis_pool_size: usize,
    #[arg(long, default_value_t = 60, help = "Rate limiting window duration size in minutes")]
    rate_limiting_window_duration_size_min: u64,
    #[arg(long, default_value_t = 1000, help = "Rate limiting max prompt tokens")]
    rate_limiting_max_prompt_tokens: u64,
    #[arg(long, default_value = "user", help = "Rate limiting user header key")]
    rate_limiting_user_header_key: String,
}

fn main() {
    let args = Args::parse();
    let openai_tls = args.openai_tls;
    let openai_port = args.openai_port;
    let openai_domain = args.openai_domain;
    let http_proxy_port = args.http_proxy_port;
    let http_proxy_metrics_port = args.http_proxy_metrics_port;
    let enable_rate_limiting = args.enable_rate_limiting;
    let rate_limiting_redis_connection_string = args.rate_limiting_redis_connection_string;
    let rate_limiting_redis_pool_size = args.rate_limiting_redis_pool_size;
    let rate_limiting_window_duration_size_min = args.rate_limiting_window_duration_size_min;
    let rate_limiting_max_prompt_tokens = args.rate_limiting_max_prompt_tokens;
    let rate_limiting_user_header_key = args.rate_limiting_user_header_key;

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
            openai_config: OpenAIConfig {
                openai_tls,
                openai_port,
                openai_domain: openai_domain.leak(),
            },
            tokenizer,
            sliding_window_rate_limiter: rate_limiter,
            rate_limiting_config: RateLimitingConfig {
                rate_limiting_max_prompt_tokens,
                rate_limiting_window_duration_size_min,
                rate_limiting_user_header_key: rate_limiting_user_header_key.leak(),
            },
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
