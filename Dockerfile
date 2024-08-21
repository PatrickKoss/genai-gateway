FROM gcr.io/distroless/static-debian11:nonroot

COPY bin/genai-gateway /genai-gateway

ENTRYPOINT ["/genai-gateway"]
