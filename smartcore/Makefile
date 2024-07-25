# When building docker in SageMaker Studio, this network is required
# See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-enable
include ../.env # For ECR configuration

TAG=smartcore

ifdef SAGEMAKER_SPACE_NAME
	DOCKER_NET=--network sagemaker
else
	DOCKER_NET=
endif

build:
	docker build . -t $(TAG) $(DOCKER_NET)
test-train: # Test training locally
	rm -rf test_dir/model
	rm -rf test_dir/output
	mkdir -p test_dir/model
	mkdir -p test_dir/output
	docker run $(DOCKER_NET) -v $(CURDIR)/test_dir:/opt/ml --rm $(TAG) train
test-serve: # Start the inference server locally for testing
	docker run $(DOCKER_NET) -v $(CURDIR)/test_dir/model:/opt/ml/model --rm $(TAG) serve
test-predict: # Send a inference request to the server started by test-serve
	curl -X POST -H "Content-Type: text/csv" -v --data-raw "1,2,3,4" http://localhost:8080/invocations
build-and-push: # Build for linux target and push to ECR
	docker build . -t $(TAG) $(DOCKER_NET) --platform=linux/amd64
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT_NUMBER).dkr.ecr.$(REGION).amazonaws.com
	docker tag $(TAG):latest $(ACCOUNT_NUMBER).dkr.ecr.$(REGION).amazonaws.com/$(TAG):latest
	docker push $(ACCOUNT_NUMBER).dkr.ecr.$(REGION).amazonaws.com/$(TAG):latest
create-ecr-repo:
	aws ecr create-repository --repository-name "${TAG}" > /dev/null


