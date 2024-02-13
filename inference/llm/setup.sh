rm -r ./tmp/
mkdir -p ./tmp/

cp -r ./config.json ./tmp/config.json
cd ./tmp/

export TRITON_REPO=$(cat ./config.json | jq -r '.triton.repo')
export TRITON_NAME=$(cat ./config.json | jq -r '.triton.name')
export TRITON_REGISTRY_NAME=$(cat ./config.json | jq -r '.triton.registry_name')
export TRITON_REGISTRY_SERVER=$(cat ./config.json | jq -r '.triton.registry_server')
export IMAGE_NAME=$(cat ./config.json | jq -r '.image.name')
export IMAGE_TAG_PREFIX=$(cat ./config.json | jq -r '.image.tag_prefix')
export IMAGE_REGISTRY_NAME=$(cat ./config.json | jq -r '.image.registry_name')
export IMAGE_REGISTRY_SERVER=$(cat ./config.json | jq -r '.image.registry_server')

echo "triton repo             = $TRITON_REPO"
echo "triton name             = $TRITON_NAME"
echo "triton registry name    = $TRITON_REGISTRY_NAME"
echo "triton registry server  = $TRITON_REGISTRY_SERVER"
echo "image name              = $IMAGE_NAME"
echo "image tag prefix        = $IMAGE_TAG_PREFIX"
echo "image registry name     = $IMAGE_REGISTRY_NAME"
echo "image registry server   = $IMAGE_REGISTRY_SERVER"

export IMAGE_REPO=$IMAGE_REGISTRY_SERVER/$IMAGE_REGISTRY_NAME
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

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
az account set --subscription $IMAGE_SUBSCRIPTION_ID
az acr login --name $IMAGE_REGISTRY_NAME

docker tag triotnserver:latest $IMAGE_REPO:$IMAGE_TAG
docker push $IMAGE_REPO:$IMAGE_TAG

cd ..
cd ..
rm -r ./tmp
