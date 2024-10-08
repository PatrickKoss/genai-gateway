use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use bytes::Bytes;
use log::info;
use pingora::prelude::{ProxyHttp, Session};
use pingora_core::prelude::HttpPeer;
use pingora_error::Error;
use pingora_error::ErrorType::HTTPStatus;
use pingora_http::{RequestHeader, ResponseHeader};
use prometheus::{register_counter_vec, register_int_counter, CounterVec, IntCounter};
use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer};
use serde_json::from_slice;
use tiktoken_rs::CoreBPE;

use crate::rate_limiter::SlidingWindowRateLimiter;

const USER_RESOURCE: &'static str = "user";

pub struct HttpGatewayConfig<R: SlidingWindowRateLimiter + Send + Sync> {
    pub openai_config: OpenAIConfig,
    pub tokenizer: CoreBPE,
    pub sliding_window_rate_limiter: R,
    pub rate_limiting_config: RateLimitingConfig,
}

pub struct RateLimitingConfig {
    pub rate_limiting_window_duration_size_min: u64,
    pub rate_limiting_max_prompt_tokens: u64,
    pub rate_limiting_user_header_key: &'static str,
}

pub struct OpenAIConfig {
    pub openai_tls: bool,
    pub openai_port: u16,
    pub openai_domain: &'static str,
}

pub struct HttpGateway<R: SlidingWindowRateLimiter + Send + Sync> {
    tokenizer: CoreBPE,
    gateway_metrics: HttpGateWayMetrics,
    down_stream_peer: HttpGateWayDownstreamPeer,
    sliding_window_rate_limiter: R,
    rate_limiting_config: RateLimitingConfig,
}

struct HttpGateWayDownstreamPeer {
    tls: bool,
    addr: &'static str,
    port: u16,
}

struct HttpGateWayMetrics {
    prompt_token_counter: &'static IntCounter,
    completion_token_counter: &'static IntCounter,
    total_token_counter: &'static IntCounter,
    prompt_token_by_model_counter: &'static CounterVec,
    completion_token_by_model_counter: &'static CounterVec,
    total_token_by_model_counter: &'static CounterVec,
    prompt_token_by_user_by_model_counter: &'static CounterVec,
    completion_token_by_user_by_model_counter: &'static CounterVec,
    total_token_by_user_by_model_counter: &'static CounterVec,
}

pub struct Ctx {
    buffer: CtxBuffer,
    openai_request: Option<OpenAIRequest>,
    rate_limiting: Option<RateLimitingCtx>,
}

pub struct RateLimitingCtx {
    pub user: String,
}

#[derive(Clone)]
struct OpenAIRequest {
    model: String,
    is_chat_completion_stream_request: bool,
    is_completion_stream_request: bool,
    is_usage_request: bool,
    stream_prompt_tokens: u64,
}

struct CtxBuffer {
    resp_buffer: Vec<u8>,
    req_buffer: Vec<u8>,
}

#[derive(Debug)]
struct TokenResponse {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct Message {
    content: String,
}

fn default_false() -> bool {
    false
}

#[derive(Deserialize, Debug)]
struct ChatCompletionRequestBody {
    model: String,
    #[serde(default = "default_false")]
    stream: bool,
    messages: Vec<Message>,
}

#[derive(Deserialize, Debug)]
struct CompletionRequestBody {
    model: String,
    #[serde(default = "default_false")]
    stream: bool,
    #[serde(deserialize_with = "deserialize_prompt")]
    prompt: Vec<String>,
}

fn deserialize_prompt<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
    where
        D: Deserializer<'de>,
{
    struct StringOrVec;

    impl<'de> Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or an array of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
        {
            Ok(vec![value.to_string()])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(value) = seq.next_element()? {
                vec.push(value);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

#[derive(Deserialize, Debug)]
struct EmbeddingsRequestBody {
    model: String,
}

#[derive(Deserialize, Debug)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct UsageResponse {
    usage: Usage,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionStreamingResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Deserialize, Debug)]
struct CompletionStreamingResponse {
    choices: Vec<CompletionChoice>,
}

#[derive(Deserialize, Debug)]
struct CompletionChoice {
    text: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionChoice {
    delta: Delta,
}

#[derive(Deserialize, Debug)]
struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[async_trait]
impl<R: SlidingWindowRateLimiter + Send + Sync> ProxyHttp for HttpGateway<R> {
    type CTX = Ctx;
    fn new_ctx(&self) -> Self::CTX {
        Ctx {
            buffer: CtxBuffer {
                resp_buffer: vec![],
                req_buffer: vec![],
            },
            openai_request: None,
            rate_limiting: None,
        }
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        _ctx: &mut Self::CTX,
    ) -> pingora_error::Result<Box<HttpPeer>> {
        let addr = (self.down_stream_peer.addr, self.down_stream_peer.port);
        let peer = Box::new(HttpPeer::new(
            addr,
            self.down_stream_peer.tls,
            self.down_stream_peer.addr.to_string(),
        ));
        Ok(peer)
    }

    async fn request_body_filter(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> pingora_error::Result<()>
    where
        Self::CTX: Send + Sync,
    {
        self.fill_openai_request(session, body, end_of_stream, ctx)?;

        Ok(())
    }

    async fn upstream_request_filter(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> pingora_error::Result<()>
    where
        Self::CTX: Send + Sync,
    {
        upstream_request.insert_header("Host", self.down_stream_peer.addr)?;
        upstream_request.insert_header("Content-Type", "application/json")?;

        let req = session.req_header().headers.clone();
        let rate_limiting_user = req
            .get(self.rate_limiting_config.rate_limiting_user_header_key)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        ctx.rate_limiting = Some(RateLimitingCtx {
            user: rate_limiting_user.to_string(),
        });

        let count = self
            .sliding_window_rate_limiter
            .fetch_sliding_window(
                USER_RESOURCE,
                rate_limiting_user,
                Duration::from_mins(self.rate_limiting_config.rate_limiting_max_prompt_tokens),
            )
            .await
            .map_err(|e| Error::explain(HTTPStatus(502), e.to_string()))?;
        if count > self.rate_limiting_config.rate_limiting_max_prompt_tokens {
            return Err(Error::explain(HTTPStatus(429), "rate limit exceeded"));
        }

        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut Self::CTX,
    ) -> pingora_error::Result<()>
    where
        Self::CTX: Send + Sync,
    {
        if upstream_response.status.as_u16() != 200 {
            return Err(Error::explain(
                HTTPStatus(upstream_response.status.as_u16()),
                "upstream response status is not 200",
            ));
        }

        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> pingora_error::Result<Option<Duration>>
    where
        Self::CTX: Send + Sync,
    {
        let token_response_openai = self.get_openai_response(body, end_of_stream, ctx)?;
        if let Some(tokens) = token_response_openai {
            let model = if let Some(openai_request) = &ctx.openai_request {
                &openai_request.model
            } else {
                ""
            };

            let rate_limiting_ctx = ctx.rate_limiting.as_ref().unwrap();
            let rate_limiting_user = &rate_limiting_ctx.user;

            self.gateway_metrics
                .completion_token_counter
                .inc_by(tokens.completion_tokens);
            self.gateway_metrics
                .prompt_token_counter
                .inc_by(tokens.prompt_tokens);
            self.gateway_metrics
                .total_token_counter
                .inc_by(tokens.completion_tokens + tokens.prompt_tokens);
            self.gateway_metrics
                .prompt_token_by_model_counter
                .with_label_values(&[model])
                .inc_by(tokens.prompt_tokens as f64);
            self.gateway_metrics
                .completion_token_by_model_counter
                .with_label_values(&[model])
                .inc_by(tokens.completion_tokens as f64);
            self.gateway_metrics
                .total_token_by_model_counter
                .with_label_values(&[model])
                .inc_by((tokens.completion_tokens + tokens.prompt_tokens) as f64);
            self.gateway_metrics
                .prompt_token_by_user_by_model_counter
                .with_label_values(&[rate_limiting_user, model])
                .inc_by(tokens.prompt_tokens as f64);
            self.gateway_metrics
                .completion_token_by_user_by_model_counter
                .with_label_values(&[rate_limiting_user, model])
                .inc_by(tokens.completion_tokens as f64);
            self.gateway_metrics
                .total_token_by_user_by_model_counter
                .with_label_values(&[rate_limiting_user, model])
                .inc_by((tokens.completion_tokens + tokens.prompt_tokens) as f64);

            // we willingly ignore the future here since the method can´t be async
            let _ = self.sliding_window_rate_limiter.record_sliding_window(
                USER_RESOURCE,
                rate_limiting_user.as_str(),
                tokens.completion_tokens + tokens.prompt_tokens,
                Duration::from_mins(
                    self.rate_limiting_config
                        .rate_limiting_window_duration_size_min,
                ),
            );
        }

        Ok(None)
    }

    async fn logging(&self, session: &mut Session, _e: Option<&Error>, ctx: &mut Self::CTX)
    where
        Self::CTX: Send + Sync,
    {
        let response_code = session
            .response_written()
            .map_or(0, |resp| resp.status.as_u16());
        info!(
            "{} response code: {response_code}",
            self.request_summary(session, ctx)
        );
    }
}

impl HttpGateWayMetrics {
    fn new() -> Self {
        static PROMPT_TOKEN_COUNTER: OnceLock<IntCounter> = OnceLock::new();
        static COMPLETION_TOKEN_COUNTER: OnceLock<IntCounter> = OnceLock::new();
        static TOTAL_TOKEN_COUNTER: OnceLock<IntCounter> = OnceLock::new();
        static PROMPT_TOKEN_BY_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();
        static COMPLETION_TOKEN_BY_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();
        static TOTAL_TOKEN_BY_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();
        static PROMPT_TOKEN_BY_USER_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();
        static COMPLETION_TOKEN_BY_USER_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();
        static TOTAL_TOKEN_BY_USER_MODEL_COUNTER: OnceLock<CounterVec> = OnceLock::new();

        Self {
            prompt_token_counter: PROMPT_TOKEN_COUNTER.get_or_init(|| {
                register_int_counter!("prompt_tokens_total", "Number of prompt tokens")
                    .expect("Failed to register prompt token counter")
            }),
            completion_token_counter: COMPLETION_TOKEN_COUNTER.get_or_init(|| {
                register_int_counter!("completion_tokens_total", "Number of completion tokens")
                    .expect("Failed to register completion token counter")
            }),
            total_token_counter: TOTAL_TOKEN_COUNTER.get_or_init(|| {
                register_int_counter!("tokens_total", "Number of total tokens")
                    .expect("Failed to register total token counter")
            }),
            prompt_token_by_model_counter: PROMPT_TOKEN_BY_MODEL_COUNTER.get_or_init(|| {
                register_counter_vec!(
                    "prompt_tokens_by_model_total",
                    "Number of prompt tokens by model",
                    &["model"]
                )
                .expect("Failed to register prompt token by model counter")
            }),
            completion_token_by_model_counter: COMPLETION_TOKEN_BY_MODEL_COUNTER.get_or_init(
                || {
                    register_counter_vec!(
                        "completion_tokens_by_model_total",
                        "Number of completion tokens by model",
                        &["model"]
                    )
                    .expect("Failed to register completion token by model counter")
                },
            ),
            total_token_by_model_counter: TOTAL_TOKEN_BY_MODEL_COUNTER.get_or_init(|| {
                register_counter_vec!(
                    "tokens_by_model_total",
                    "Number of total tokens by model",
                    &["model"]
                )
                .expect("Failed to register total token by model counter")
            }),
            prompt_token_by_user_by_model_counter: PROMPT_TOKEN_BY_USER_MODEL_COUNTER.get_or_init(
                || {
                    register_counter_vec!(
                        "prompt_tokens_by_user_by_model_total",
                        "Number of prompt tokens by user by model",
                        &["user", "model"]
                    )
                    .expect("Failed to register prompt token by user by model counter")
                },
            ),
            completion_token_by_user_by_model_counter: COMPLETION_TOKEN_BY_USER_MODEL_COUNTER
                .get_or_init(|| {
                    register_counter_vec!(
                        "completion_tokens_by_user_by_model_total",
                        "Number of completion tokens by user by model",
                        &["user", "model"]
                    )
                    .expect("Failed to register completion token by user by model counter")
                }),
            total_token_by_user_by_model_counter: TOTAL_TOKEN_BY_USER_MODEL_COUNTER.get_or_init(
                || {
                    register_counter_vec!(
                        "tokens_by_user_by_model_total",
                        "Number of total tokens by user by model",
                        &["user", "model"]
                    )
                    .expect("Failed to register total token by user by model counter")
                },
            ),
        }
    }
}

impl<R: SlidingWindowRateLimiter + Send + Sync> HttpGateway<R> {
    pub fn new(http_gateway_config: HttpGatewayConfig<R>) -> AnyResult<Self> {
        Ok(Self {
            tokenizer: http_gateway_config.tokenizer,
            gateway_metrics: HttpGateWayMetrics::new(),
            sliding_window_rate_limiter: http_gateway_config.sliding_window_rate_limiter,
            down_stream_peer: HttpGateWayDownstreamPeer {
                tls: http_gateway_config.openai_config.openai_tls,
                addr: http_gateway_config.openai_config.openai_domain,
                port: http_gateway_config.openai_config.openai_port,
            },
            rate_limiting_config: http_gateway_config.rate_limiting_config,
        })
    }

    fn fill_openai_request(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> pingora_error::Result<()> {
        let path = session.req_header().uri.path();

        if session.req_header().method == "POST" {
            match path {
                p if p.starts_with("/v1/chat/completions") => {
                    self.process_chat_completions(body, end_of_stream, ctx)
                }
                p if p.starts_with("/v1/completions") => {
                    self.process_completions(body, end_of_stream, ctx)
                }
                p if p.starts_with("/v1/embeddings") => {
                    self.process_embeddings(body, end_of_stream, ctx)
                }
                _ => Ok(()),
            }
        } else {
            Ok(())
        }
    }

    fn create_openai_request(
        &self,
        ctx: &mut Ctx,
        model: String,
        is_chat: bool,
        is_completion: bool,
        is_usage: bool,
        prompt_tokens: usize,
    ) -> pingora_error::Result<()> {
        ctx.openai_request = Some(OpenAIRequest {
            is_chat_completion_stream_request: is_chat,
            is_usage_request: is_usage,
            is_completion_stream_request: is_completion,
            model,
            stream_prompt_tokens: prompt_tokens as u64,
        });
        Ok(())
    }

    fn process_embeddings(
        &self,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> pingora_error::Result<()> {
        self.extend_request_buffer(ctx, body);

        if end_of_stream {
            from_slice::<EmbeddingsRequestBody>(&ctx.buffer.req_buffer)
                .map_err(|_| Error::explain(HTTPStatus(400), "invalid request body"))
                .and_then(|json_body| {
                    self.create_openai_request(ctx, json_body.model, false, false, true, 0)
                })?;
        }

        Ok(())
    }

    fn process_completions(
        &self,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> pingora_error::Result<()> {
        self.extend_request_buffer(ctx, body);

        if end_of_stream {
            from_slice::<CompletionRequestBody>(&ctx.buffer.req_buffer)
                .map_err(|_| Error::explain(HTTPStatus(400), "invalid request body"))
                .and_then(|json_body| {
                    let is_stream_request = json_body.stream;

                    let prompt_tokens = if is_stream_request {
                        json_body
                            .prompt
                            .iter()
                            .map(|message| {
                                self.tokenizer
                                    .encode_with_special_tokens(message.as_str())
                                    .len()
                            })
                            .sum()
                    } else {
                        0
                    };

                    self.create_openai_request(
                        ctx,
                        json_body.model,
                        false,
                        json_body.stream,
                        !json_body.stream,
                        prompt_tokens,
                    )
                })?;
        }

        Ok(())
    }

    fn process_chat_completions(
        &self,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> pingora_error::Result<()> {
        self.extend_request_buffer(ctx, body);

        if end_of_stream {
            from_slice::<ChatCompletionRequestBody>(&ctx.buffer.req_buffer)
                .map_err(|_| Error::explain(HTTPStatus(400), "invalid request body"))
                .map(|json_body| {
                    let model = json_body.model;
                    let is_stream_request = json_body.stream;
                    let prompt_tokens = if is_stream_request {
                        json_body
                            .messages
                            .iter()
                            .map(|message| {
                                self.tokenizer
                                    .encode_with_special_tokens(&message.content)
                                    .len()
                            })
                            .sum()
                    } else {
                        0
                    };

                    ctx.openai_request = Some(OpenAIRequest {
                        is_chat_completion_stream_request: is_stream_request,
                        is_usage_request: !is_stream_request,
                        is_completion_stream_request: false,
                        model,
                        stream_prompt_tokens: prompt_tokens as u64,
                    });
                })?;
        }

        Ok(())
    }

    fn extend_request_buffer(&self, ctx: &mut Ctx, body: &Option<Bytes>) {
        if let Some(b) = body {
            ctx.buffer.req_buffer.extend(&b[..]);
        }
    }

    fn get_openai_response(
        &self,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> pingora_error::Result<Option<TokenResponse>>
    where
        Ctx: Send + Sync,
    {
        if let Some(openai_request) = &ctx.openai_request {
            self.extend_response_buffer(&mut ctx.buffer.resp_buffer, body);

            if end_of_stream {
                if openai_request.is_chat_completion_stream_request {
                    return self.process_chat_completions_response(
                        &ctx.buffer.resp_buffer,
                        openai_request,
                    );
                } else if openai_request.is_completion_stream_request {
                    return self
                        .process_completions_response(&ctx.buffer.resp_buffer, openai_request);
                } else if openai_request.is_usage_request {
                    return self.process_usage_response(&ctx.buffer.resp_buffer);
                }
            }
        }

        Ok(None)
    }

    fn extend_response_buffer(&self, resp_buffer: &mut Vec<u8>, body: &Option<Bytes>) {
        if let Some(b) = body {
            resp_buffer.extend(&b[..]);
        }
    }

    fn process_chat_completions_response(
        &self,
        resp_buffer: &[u8],
        openai_request: &OpenAIRequest,
    ) -> pingora_error::Result<Option<TokenResponse>> {
        let responses = self.extract_responses::<ChatCompletionStreamingResponse>(resp_buffer)?;
        let token_count = self.calculate_token_count_for_chat(&responses);
        Ok(Some(TokenResponse {
            completion_tokens: token_count as u64,
            prompt_tokens: openai_request.stream_prompt_tokens,
        }))
    }

    fn process_completions_response(
        &self,
        resp_buffer: &[u8],
        openai_request: &OpenAIRequest,
    ) -> pingora_error::Result<Option<TokenResponse>> {
        let responses = self.extract_responses::<CompletionStreamingResponse>(resp_buffer)?;
        let token_count = self.calculate_token_count_for_completion(&responses);
        Ok(Some(TokenResponse {
            completion_tokens: token_count as u64,
            prompt_tokens: openai_request.stream_prompt_tokens,
        }))
    }

    fn process_usage_response(
        &self,
        resp_buffer: &[u8],
    ) -> pingora_error::Result<Option<TokenResponse>> {
        from_slice::<UsageResponse>(resp_buffer)
            .map_err(|_| Error::explain(HTTPStatus(502), "invalid response body"))
            .map(|usage_body| {
                Some(TokenResponse {
                    completion_tokens: usage_body.usage.completion_tokens,
                    prompt_tokens: usage_body.usage.prompt_tokens,
                })
            })
    }

    fn extract_responses<T: DeserializeOwned>(
        &self,
        buffer: &[u8],
    ) -> pingora_error::Result<Vec<T>> {
        let lines = buffer.split(|&byte| byte == b'\n');
        let json_lines: Vec<&[u8]> = lines
            .filter(|line| line.starts_with(b"data: {"))
            .map(|line| &line[6..])
            .collect();

        json_lines
            .iter()
            .map(|&line| {
                from_slice::<T>(line)
                    .map_err(|_| Error::explain(HTTPStatus(502), "invalid response body"))
            })
            .collect()
    }

    fn calculate_token_count_for_chat(
        &self,
        responses: &[ChatCompletionStreamingResponse],
    ) -> usize {
        responses
            .iter()
            .flat_map(|response| &response.choices)
            .map(|choice| {
                let binding = "".to_string();
                let content = choice.delta.content.as_ref().unwrap_or(&binding);
                self.tokenizer.encode_with_special_tokens(content).len()
            })
            .sum()
    }

    fn calculate_token_count_for_completion(
        &self,
        responses: &[CompletionStreamingResponse],
    ) -> usize {
        responses
            .iter()
            .flat_map(|response| &response.choices)
            .map(|choice| {
                self.tokenizer
                    .encode_with_special_tokens(&choice.text)
                    .len()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
    use std::thread;

    use anyhow::Result;
    use async_trait::async_trait;
    use httpmock::MockServer;
    use mockall::mock;
    use pingora::prelude::http_proxy_service;
    use reqwest::StatusCode;
    use serde_json::json;
    use serde_json::Value;
    use tiktoken_rs::cl100k_base;
    use tokio::time;
    use tokio::time::Duration;

    use crate::http_proxy::{HttpGateway, HttpGatewayConfig, OpenAIConfig, RateLimitingConfig};
    use crate::rate_limiter::SlidingWindowRateLimiter;

    mock! {
        SlidingWindowRateLimiterImpl {}
        #[async_trait]
        impl SlidingWindowRateLimiter for SlidingWindowRateLimiterImpl {
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
    }

    #[tokio::test]
    async fn test_get_models() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/v1/models");
                then.status(200).json_body(json!({
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo"}],
                }));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{}/v1/models", http_proxy_server_port).as_str())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);

        let bytes = res.bytes().await.expect("Failed to read response body");
        let json: Value = serde_json::from_slice(&bytes).expect("Failed to parse JSON");

        assert_eq!(json.get("object").unwrap().as_str(), Some("list"));
        assert!(json.get("data").is_some());
        let data = json.get("data").unwrap().as_array().unwrap();
        assert_eq!(data[0].get("id").unwrap().as_str(), Some("gpt-3.5-turbo"));
    }

    #[tokio::test]
    async fn test_upstream_down() {
        let free_port = find_free_port();
        let http_proxy_server_port = start_http_proxy_server(free_port).await;
        let res = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{}/v1/models", http_proxy_server_port).as_str())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn test_post_completion_invalid_response_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(
                    r#"
                    {
                    "invalid: "json",
                    }
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "prompt": "test"}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn test_post_completion_invalid_request_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_post_completion() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "prompt": "test"}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);

        let bytes = res.bytes().await.expect("Failed to read response body");
        let json: Value = serde_json::from_slice(&bytes).expect("Failed to parse JSON");

        assert!(json.get("usage").is_some());
        let data = json.get("usage").unwrap().as_object().unwrap();
        assert_eq!(data.get("prompt_tokens").unwrap().as_i64(), Some(20i64));
        assert_eq!(data.get("completion_tokens").unwrap().as_i64(), Some(10i64));
    }

    #[tokio::test]
    async fn test_post_completion_prompt_array() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "prompt": ["test"]}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);

        let bytes = res.bytes().await.expect("Failed to read response body");
        let json: Value = serde_json::from_slice(&bytes).expect("Failed to parse JSON");

        assert!(json.get("usage").is_some());
        let data = json.get("usage").unwrap().as_object().unwrap();
        assert_eq!(data.get("prompt_tokens").unwrap().as_i64(), Some(20i64));
        assert_eq!(data.get("completion_tokens").unwrap().as_i64(), Some(10i64));
    }

    #[tokio::test]
    async fn test_post_completion_stream() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(
                    r#"

                data: {"choices": [{"text": "test"}]}

                data: {"choices": [{"text": "test"}]}

                data: {"choices": [{"text": "test"}]}

                data: [DONE]
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "prompt": "test", "stream": true}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_post_completion_stream_wrong_response_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/completions");
                then.status(200).json_body(
                    r#"
                data: {"choices: [{text": test"}]}

                data: {"choices: {"text": "test"}]}

                data: [DONE]
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "prompt": "test", "stream": true}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_post_chat_completion_invalid_response_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions");
                then.status(200).json_body(
                    r#"
                    {
                    "invalid: "json",
                    }
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content":"what are the best football players all time?"}]}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn test_post_chat_completion_invalid_request_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(
                format!(
                    "http://127.0.0.1:{}/v1/chat/completions",
                    http_proxy_server_port
                )
                .as_str(),
            )
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_post_chat_completion() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content":"what are the best football players all time?"}]}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);

        let bytes = res.bytes().await.expect("Failed to read response body");
        let json: Value = serde_json::from_slice(&bytes).expect("Failed to parse JSON");

        assert!(json.get("usage").is_some());
        let data = json.get("usage").unwrap().as_object().unwrap();
        assert_eq!(data.get("prompt_tokens").unwrap().as_i64(), Some(20i64));
        assert_eq!(data.get("completion_tokens").unwrap().as_i64(), Some(10i64));
    }

    #[tokio::test]
    async fn test_post_chat_completion_stream() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions");
                then.status(200).json_body(
                    r#"

                data: {"choices": [{"delta": {"content": "test"}}]}

                data: {"choices": [{"delta": {"content": "test"}}]}

                data: {"choices": [{"delta": {"content": "test"}}]}

                data: [DONE]
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content":"what are the best football players all time?"}], "stream": true}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_post_chat_completion_stream_wrong_response_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/v1/chat/completions");
                then.status(200).json_body(
                    r#"
                data: {"choices: [{"delta": {"content": "test"}}]}

                data: [DONE]
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content":"what are the best football players all time?"}], "stream": true}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_post_embeddings_invalid_response_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/embeddings");
                then.status(200).json_body(
                    r#"
                    {
                    "invalid: "json",
                    }
                "#,
                );
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/embeddings", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "input": "test"}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn test_post_embeddings_invalid_request_body() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/embeddings");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/embeddings", http_proxy_server_port).as_str())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_post_embeddings() {
        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/embeddings");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port = start_http_proxy_server(mock_server.port()).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/embeddings", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "input": "test"}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::OK);

        let bytes = res.bytes().await.expect("Failed to read response body");
        let json: Value = serde_json::from_slice(&bytes).expect("Failed to parse JSON");

        assert!(json.get("usage").is_some());
        let data = json.get("usage").unwrap().as_object().unwrap();
        assert_eq!(data.get("prompt_tokens").unwrap().as_i64(), Some(20i64));
        assert_eq!(data.get("completion_tokens").unwrap().as_i64(), Some(10i64));
    }

    #[tokio::test]
    async fn test_post_embeddings_rate_limited() {
        let mut mock_sliding_window_rate_limiter = MockSlidingWindowRateLimiterImpl::new();
        mock_sliding_window_rate_limiter
            .expect_record_sliding_window()
            .times(1)
            .returning(|_, _, _, _| Ok(0));
        mock_sliding_window_rate_limiter
            .expect_fetch_sliding_window()
            .times(1)
            .returning(|_, _, _| Ok(101));

        let mock_server = MockServer::start();
        mock_server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/v1/embeddings");
                then.status(200).json_body(json!(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 10,
                        },
                    }
                ));
            })
            .await;

        let http_proxy_server_port =
            start_http_proxy_with_mock(mock_server.port(), mock_sliding_window_rate_limiter).await;
        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/v1/embeddings", http_proxy_server_port).as_str())
            .body(json!({"model": "gpt-3.5-turbo", "input": "test"}).to_string())
            .send()
            .await
            .expect("Failed to send request");

        assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    async fn start_http_proxy_with_mock(
        upstream_port: u16,
        mock_sliding_window_rate_limiter: MockSlidingWindowRateLimiterImpl,
    ) -> u16 {
        let mut server = pingora_core::prelude::Server::new(None).unwrap();
        server.bootstrap();

        let tokenizer = cl100k_base().expect("Failed to load tokenizer");
        let free_port = find_free_port();

        let mut http_proxy = http_proxy_service(
            &server.configuration,
            HttpGateway::new(HttpGatewayConfig {
                openai_config: OpenAIConfig {
                    openai_tls: false,
                    openai_port: upstream_port,
                    openai_domain: "0.0.0.0",
                },
                tokenizer,
                sliding_window_rate_limiter: mock_sliding_window_rate_limiter,
                rate_limiting_config: RateLimitingConfig {
                    rate_limiting_window_duration_size_min: 1,
                    rate_limiting_max_prompt_tokens: 100,
                    rate_limiting_user_header_key: "user",
                },
            })
            .expect("Failed to create http gateway"),
        );
        http_proxy.add_tcp(format!("0.0.0.0:{}", free_port).as_str());
        server.add_service(http_proxy);

        thread::spawn(|| {
            server.run_forever();
        });

        time::sleep(time::Duration::from_millis(100)).await;

        free_port
    }

    async fn start_http_proxy_server(upstream_port: u16) -> u16 {
        let mut mock_sliding_window_rate_limiter = MockSlidingWindowRateLimiterImpl::new();
        mock_sliding_window_rate_limiter
            .expect_record_sliding_window()
            .times(1)
            .returning(|_, _, _, _| Ok(0));
        mock_sliding_window_rate_limiter
            .expect_fetch_sliding_window()
            .times(1)
            .returning(|_, _, _| Ok(0));

        start_http_proxy_with_mock(upstream_port, mock_sliding_window_rate_limiter).await
    }

    fn find_free_port() -> u16 {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.local_addr().unwrap().port()
    }
}
