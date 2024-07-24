# Rust Machine Learning on Amazon SageMaker
This repository contains example containers to run machine learning using Rust machine learning frameworks on Amazon SageMaker. This is build using the SageMaker's [Bring your own container](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-create.html) feature.

## Prerequsite
* (Optional) [Install Rust](https://www.rust-lang.org/tools/install)
    * The Rust code is built in the Rust container, so you don't need to install them locally. But if you want to debug the Rust code locally you can install them.
* Use SageMaker Studio to build the docker images
  * Alternatively, you can develop this on local computer, but you need to install `docker` locally
  * Enable docker support for your SageMaker Domain by following this [doc](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable). Do this BEFORE you create your code-editor application.
  * Open a code-editor application in SageMaker Studio
  * Install docker following the [doc](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable)
* If you want to push the container image to Elastic Container Registry (ECR), you need to set your credentials and ECR configuration in `.env`. 

## Usage
Each top-level folder contains an example container definition for a Rust machine learning framework.
* `smartcore`: [SmartCore](smartcore/src/bin/train-benchmark.rs) ([GitHub](smartcore/src/bin/train-benchmark.rs))
* `linfa`: coming soon
* `burn`: coming soon

To use the container:
1. Navigate into the folder containing the framewrok
2. Modify the `src/bin/train.rs` to add your own model training logic
3. Modify the `src/bin/serve.rs` to add your own model inference logic
4. Build the container with `make build`

### Local testing
* `make test-train`: Test model training locally
* `make test-serve`: start the model inference server for local testing
  * `make test-predict`: Send a inference request to the inference server

### Push to ECR
* Set your ECR configuration in `.env` in the top level directry
* Attach the policy `AmazonEC2ContainerRegistryPowerUser` to your IAM role/user. If you are running in SageMaker Studio, add it to the execution role of your SageMaker Domain profile.
* Go back to the framework folder, run `make create-ecr-repo` to create the ECR repository
* Run `make build-and-push` to build and push the image to ECR

# Testing
* We use `git-secrets` to scan for credentials and secrets
```
sudo apt install git-secrets
git secrets --install
git secrets --register-aws
```

## Citation
* Dataset
  * Iris dataset: Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76

## Roadmap
- [ ] Add Linfa framework
- [ ] Add Burn framework
- [ ] Add Polars framework
- [ ] Add more Rust ML framework
- [ ] Support full BYOC input/output (e.g. `hyperparameters.json`, `resourceConfig.json`, `ouptut/failure`)
- [ ] Support pipe mode
- [ ] Support batch inference
- [ ] Support data processing job
- [ ] Add example Python notebook
- [ ] Add example Rust notebook 
