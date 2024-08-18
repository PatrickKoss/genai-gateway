use std::env;

use clap::Parser;
use pingora::prelude::*;
use tiktoken_rs::cl100k_base;

use http_proxy::{HttpGateway, HttpGatewayConfig};

mod http_proxy;

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
}

#[cfg(not(tarpaulin_include))]
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

    env_logger::init();

    let mut server = Server::new(None).unwrap();
    server.bootstrap();

    let tokenizer = cl100k_base().expect("Failed to load tokenizer");

    let mut http_proxy = http_proxy_service(
        &server.configuration,
        HttpGateway::new(HttpGatewayConfig {
            openai_tls,
            openai_port,
            openai_domain: openai_domain.leak(),
            tokenizer,
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
