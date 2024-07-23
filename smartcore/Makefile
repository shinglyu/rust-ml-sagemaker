# When building docker in SageMaker Studio, this network is required
# See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable
ifdef SAGEMAKER_SPACE_NAME
	DOCKER_NET=--network sagemaker
else
	DOCKER_NET=
endif

build:
	docker build . -t smartcore $(DOCKER_NET)
test-train: # Test training locally
	rm -rf test_dir/model
	rm -rf test_dir/output
	mkdir -p test_dir/model
	mkdir -p test_dir/output
	docker run $(DOCKER_NET) -v $(CURDIR)/test_dir:/opt/ml smartcore train