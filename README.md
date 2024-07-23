# Prerequsite
* (Optional) [Install Rust](https://www.rust-lang.org/tools/install)
    * The Rust code is built in the Rust container, so you don't need to install them locally. But if you want to debug the Rust code locally you can install them.
* Use SageMaker Studio to build the docker images
  * Alternatively, you can develop this on local computer, but you need to install `docker` locally
  * Enable docker support for your SageMaker Domain by following this [doc](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable). Do this BEFORE you create your code-editor application.
  * Open a code-editor application in SageMaker Studio
  * Install docker following the [doc](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable)
# Building a Rust ML framework container
* `cd` to the framework container folder. For example, `smartcore/`.
