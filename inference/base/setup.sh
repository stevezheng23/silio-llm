rm -r ./tmp/
mkdir -p ./tmp/

cp -r ./config.json ./tmp/config.json
cd ./tmp/

export TRITON_REPO=$(cat ./config.json | jq -r '.triton.repo')
export TRITON_RELEASE=$(cat ./config.json | jq -r '.triton.release')
export TRITON_PLATFORM=$(cat ./config.json | jq -r '.triton.platform')
export TRITON_MACHINE=$(cat ./config.json | jq -r '.triton.machine')
export IMAGE_NAME=$(cat ./config.json | jq -r '.image.name')
export IMAGE_TAG_PREFIX=$(cat ./config.json | jq -r '.image.tag_prefix')
export IMAGE_SUBSCRIPTION_ID=$(cat ./config.json | jq -r '.image.subscription_id')
export IMAGE_REGISTRY_NAME=$(cat ./config.json | jq -r '.image.registry_name')
export IMAGE_REGISTRY_SERVER=$(cat ./config.json | jq -r '.image.registry_server')

echo "triton repo             = $TRITON_REPO"
echo "triton release          = $TRITON_RELEASE"
echo "triton platform         = $TRITON_PLATFORM"
echo "triton machine          = $TRITON_MACHINE"
echo "image name              = $IMAGE_NAME"
echo "image tag prefix        = $IMAGE_TAG_PREFIX"
echo "image subscription id   = $IMAGE_SUBSCRIPTION_ID"
echo "image registry name     = $IMAGE_REGISTRY_NAME"
echo "image registry server   = $IMAGE_REGISTRY_SERVER"

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
az account set --subscription $IMAGE_SUBSCRIPTION_ID
az acr login --name $IMAGE_REGISTRY_NAME

export IMAGE_REPO=$IMAGE_REGISTRY_SERVER/$IMAGE_NAME
export IMAGE_TAG=$IMAGE_TAG_PREFIX.latest

git clone $TRITON_REPO ./repo
cd ./repo
export TRITON_BRANCH=r$TRITON_RELEASE
export TRITON_IMAGE_TAG=$TRITON_RELEASE-py3-min

git checkout $TRITON_BRANCH
python build.py -v \
--image=gpu-base,nvcr.io/nvidia/tritonserver:$TRITON_IMAGE_TAG \
--target-platform=$TRITON_PLATFORM \
--target-machine=$TRITON_MACHINE \
--enable-logging --enable-stats --enable-tracking --enable-metrics \
--endpoint=grpc --endpoint=http \
--backend=ensemble --backend=onnxruntime --backend=python

cd ..
mkdir ./build
cp -r ../model_repository ./build/model_repository
cp -r ../Dockerfile ./build/Dockerfile
cd ./build
docker build -t custom_image:latest .
cd ..

docker run --rm -d --shm-size 1g \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  custom_image:latest \
  tritonserver --model-repository=/model_repository
sleep 20
curl -v localhost:8000/v2/health/ready

docker tag triotnserver:latest $IMAGE_REPO:$IMAGE_TAG
docker push $IMAGE_REPO:$IMAGE_TAG

cd ..
rm -r ./tmp
