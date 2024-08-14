use async_trait::async_trait;
use bytes::Bytes;
use log::info;
use pingora::prelude::*;
use pingora_http::ResponseHeader;
use prometheus::{register_counter_vec, register_int_counter};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::from_slice;
use tiktoken_rs::{cl100k_base, CoreBPE};

fn main() {
    env_logger::init();

    let mut server = Server::new(None).unwrap();
    server.bootstrap();

    let tokenizer = cl100k_base().expect("Failed to load tokenizer");

    let mut http_proxy = http_proxy_service(
        &server.configuration,
        HttpGateway {
            tokenizer,
            gateway_metrics: HttpGateWayMetrics {
                prompt_token_counter: register_int_counter!(
                    "prompt_tokens_total",
                    "Number of prompt tokens"
                )
                .expect("Failed to register prompt token counter"),
                completion_token_counter: register_int_counter!(
                    "completion_tokens_total",
                    "Number of completion tokens"
                )
                .expect("Failed to register completion token counter"),
                total_token_counter: register_int_counter!(
                    "tokens_total",
                    "Number of total tokens"
                )
                .expect("Failed to register total token counter"),
                prompt_token_by_model_counter: register_counter_vec!(
                    "prompt_tokens_by_model_total",
                    "Number of prompt tokens by model",
                    &["model"]
                )
                .expect("Failed to register prompt token by model counter"),
                completion_token_by_model_counter: register_counter_vec!(
                    "completion_tokens_by_model_total",
                    "Number of completion tokens by model",
                    &["model"]
                )
                .expect("Failed to register completion token by model counter"),
                total_token_by_model_counter: register_counter_vec!(
                    "tokens_by_model_total",
                    "Number of total tokens by model",
                    &["model"]
                )
                .expect("Failed to register total token by model counter"),
            },
            down_stream_peer: HttpGateWayDownstreamPeer {
                tls: false,
                addr: "0.0.0.0",
                port: 8081,
            },
        },
    );

    http_proxy.add_tcp("0.0.0.0:6188");
    server.add_service(http_proxy);

    let mut prometheus_service_http =
        pingora_core::services::listening::Service::prometheus_http_service();
    prometheus_service_http.add_tcp("127.0.0.1:6192");
    server.add_service(prometheus_service_http);

    server.run_forever();
}

pub struct HttpGateway {
    tokenizer: CoreBPE,
    gateway_metrics: HttpGateWayMetrics,
    down_stream_peer: HttpGateWayDownstreamPeer,
}

struct HttpGateWayDownstreamPeer {
    tls: bool,
    addr: &'static str,
    port: u16,
}

struct HttpGateWayMetrics {
    prompt_token_counter: prometheus::IntCounter,
    completion_token_counter: prometheus::IntCounter,
    total_token_counter: prometheus::IntCounter,
    prompt_token_by_model_counter: prometheus::CounterVec,
    completion_token_by_model_counter: prometheus::CounterVec,
    total_token_by_model_counter: prometheus::CounterVec,
}

pub struct Ctx {
    buffer: CtxBuffer,
    openai_request: Option<OpenAIRequest>,
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

#[async_trait]
impl ProxyHttp for HttpGateway {
    type CTX = Ctx;
    fn new_ctx(&self) -> Self::CTX {
        Ctx {
            buffer: CtxBuffer {
                resp_buffer: vec![],
                req_buffer: vec![],
            },
            openai_request: None,
        }
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        _ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
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
    ) -> Result<()>
    where
        Self::CTX: Send + Sync,
    {
        self.fill_openai_request(session, body, end_of_stream, ctx)?;

        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut Self::CTX,
    ) -> Result<()>
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
    ) -> Result<Option<std::time::Duration>>
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

impl HttpGateway {
    fn fill_openai_request(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> Result<()> {
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
    ) -> Result<()> {
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
    ) -> Result<()> {
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
    ) -> Result<()> {
        self.extend_request_buffer(ctx, body);

        if end_of_stream {
            from_slice::<CompletionRequestBody>(&ctx.buffer.req_buffer)
                .map_err(|_| Error::explain(HTTPStatus(400), "invalid request body"))
                .and_then(|json_body| {
                    self.create_openai_request(
                        ctx,
                        json_body.model,
                        false,
                        json_body.stream,
                        !json_body.stream,
                        0,
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
    ) -> Result<()> {
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
    ) -> Result<Option<TokenResponse>>
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
    ) -> Result<Option<TokenResponse>> {
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
    ) -> Result<Option<TokenResponse>> {
        let responses = self.extract_responses::<CompletionStreamingResponse>(resp_buffer)?;
        let token_count = self.calculate_token_count_for_completion(&responses);
        Ok(Some(TokenResponse {
            completion_tokens: token_count as u64,
            prompt_tokens: openai_request.stream_prompt_tokens,
        }))
    }

    fn process_usage_response(&self, resp_buffer: &[u8]) -> Result<Option<TokenResponse>> {
        from_slice::<UsageResponse>(resp_buffer)
            .map_err(|_| Error::explain(HTTPStatus(502), "invalid response body"))
            .map(|usage_body| {
                Some(TokenResponse {
                    completion_tokens: usage_body.usage.completion_tokens,
                    prompt_tokens: usage_body.usage.prompt_tokens,
                })
            })
    }

    fn extract_responses<T: DeserializeOwned>(&self, buffer: &[u8]) -> Result<Vec<T>> {
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
