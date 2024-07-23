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
	docker run $(DOCKER_NET) -v $(CURDIR)/test_dir:/opt/ml --rm smartcore train
test-serve: # Start the inference server locally for testing
	docker run $(DOCKER_NET) -v $(CURDIR)/test_dir/model:/opt/ml/model -p 8080:8080 --rm smartcore serve
test-predict: # Send a inference request to the server started by test-serve
	curl -X POST -H "Content-Type: text/csv" -v --data-raw "1,2,3,4,5" http://localhost:8080/invocations