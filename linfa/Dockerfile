# Speedup the build using https://depot.dev/blog/rust-dockerfile-best-practices
# SageMaker don't support Docker Buildkit (need to confirm this?) so we can't use the other optimizations

# Builder stage
FROM rust:1.79.0 AS base
RUN cargo install cargo-chef --version ^0.1

# Planner stage
FROM base AS planner
WORKDIR /app
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Builder stage
FROM base AS builder
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bins

# Deployment package
FROM debian:stable-slim
COPY --from=builder /app/target/release/train /usr/bin/train
COPY --from=builder /app/target/release/serve /usr/bin/serve