# When building docker in SageMaker Studio, this network is required
# See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable
ifdef SAGEMAKER_SPACE_NAME
	DOCKER_NET=--network sagemaker
else
	DOCKER_NET=
endif

build:
	docker build . -t smartcore $(DOCKER_NET)
train:
	docker run $(DOCKER_NET) smartcore train