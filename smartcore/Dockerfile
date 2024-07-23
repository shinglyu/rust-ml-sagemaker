# Compile
FROM rust:1.79.0 as builder
WORKDIR /app
COPY . /app
RUN cargo build --release --bins

# Deployment package
# Not working right now, might need
FROM debian:stable-slim
COPY --from=builder /app/target/release/train /usr/bin/train
COPY --from=builder /app/target/release/serve /usr/bin/serve

# DEBUG ONLY, remove for prod => Mount it like this https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/local_test/train_local.sh
COPY dataset/iris.csv opt/ml/input/data/training/iris.csv