VERSION ?= v0.0.1
IMAGE_TAG_BASE ?= ghrc.io/patrickkoss/genai-gateway
IMG ?= $(IMAGE_TAG_BASE):$(VERSION)

build-linux:
	CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc cargo build --release --target x86_64-unknown-linux-gnu -Z unstable-options --artifact-dir ./bin/

docker-build-linux: build-linux
	docker buildx build --platform linux/amd64 -t $(IMG) -f Dockerfile .

test:
	cargo test

lint:
	cargo clippy --fix --allow-dirty

fmt:
	cargo fmt
