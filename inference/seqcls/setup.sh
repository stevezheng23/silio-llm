for i in "$@"
  do
    case $i in
      --taskname=*)
      TASKNAME="$(i#*=)"
      shift
      ;;
    esac
  done

rm -r ./tmp/
mkdir -p ./tmp/

cp -r ./config.$TASKNAME.json ./tmp/config.json
cd ./tmp/

export MODEL_ID=$(cat ./config.json | jq -r '.model.id')
export MODEL_TYPE=$(cat ./config.json | jq -r '.model.type')
export MODEL_NUM_LABELS=$(cat ./config.json | jq -r '.model.num_labels')
export MODEL_ARTIFACT_PATH=$(cat ./config.json | jq -r '.model.artifact_path')
export MODEL_SUBSCRIPTION_ID=$(cat ./config.json | jq -r '.model.subscription_id')
export MODEL_RESOURCE_GROUP=$(cat ./config.json | jq -r '.model.resource_group')
export MODEL_WORKSPACE_NAME=$(cat ./config.json | jq -r '.model.workspace_name')

export TOKENIZER_NAME=$(cat ./config.json | jq -r '.tokenizer.name')
export TOKENIZER_MAX_LENGTH=$(cat ./config.json | jq -r '.tokenizer.max_length')
export TOKENIZER_OUTPUT_TYPE_IDS=$(cat ./config.json | jq -r '.tokenizer.output_type_ids')

export IMAGE_NAME=$(cat ./config.json | jq -r '.image.name')
export IMAGE_TAG_PREFIX=$(cat ./config.json | jq -r '.image.tag_prefix')
export IMAGE_SUBSCRIPTION_ID=$(cat ./config.json | jq -r '.image.subscription_id')
export IMAGE_REGISTRY_NAME=$(cat ./config.json | jq -r '.image.registry_name')
export IMAGE_REGISTRY_SERVER=$(cat ./config.json | jq -r '.image.registry_server')

echo "model id                  = $MODEL_ID"
echo "model type                = $MODEL_TYPE"
echo "model # labels            = $MODEL_NUM_LABELS"
echo "model artifact path       = $MODEL_ARTIFACT_PATH"
echo "model subscription id     = $MODEL_SUBSCRIPTION_ID"
echo "model resource group      = $MODEL_RESOURCE_GROUP"
echo "model workspace name      = $MODEL_WORKSPACE_NAME"

echo "tokenizer name            = $TOKENIZER_NAME"
echo "tokenizer max length      = $TOKENIZER_MAX_LENGTH"
echo "tokenizer output type ids = $TOKENIZER_OUTPUT_TYPE_IDS"

echo "image name                = $IMAGE_NAME"
echo "image tag prefix          = $IMAGE_TAG_PREFIX"
echo "image subscription id     = $IMAGE_SUBSCRIPTION_ID"
echo "image registry name       = $IMAGE_REGISTRY_NAME"
echo "image registry server     = $IMAGE_REGISTRY_SERVER"

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
sudo az extension add -n azure-cli-ml -y
az login
az account set --subscription $MODEL_SUBSCRIPTION_ID

mkdir ./model
az ml model download \
--model-id $MODEL_ID \
--target-dir ./model/ \
--workspace-name $MODEL_WORKSPACE_NAME \
--resource-group $MODEL_RESOURCE_GROUP \
--subscription-id $MODEL_SUBSCRIPTION_ID

mkdir ./build
cp -r ../Dockerfile ./build/Dockerfile
cp -r ../model_repository ./build/model_repository
cp -r ./model/$MODEL_ARTIFACT_PATH/model.onnx ./build/model_repository/predict-onnx/1/
cp -r ./model/$MODEL_ARTIFACT_PATH/labels.txt ./build/model_repository/postproc/1/
cp -r ./model/$MODEL_ARTIFACT_PATH/labels.txt ./build/model_repository/taxonomy/1/

python update_template.py \
--input_file ./build/model_repository/predict-onnx/config.pbtxt.tmpl \
--outputp_file ./build/model_repository/predict-onnx/config.pbtxt \
--num_labels $MODEL_NUM_LABELS
rm ./build/model_repository/predict-onnx/config.pbtxt.tmpl

python update_template.py \
--input_file ./build/model_repository/postproc/config.pbtxt.tmpl \
--outputp_file ./build/model_repository/postproc/config.pbtxt \
--num_labels $MODEL_NUM_LABELS
rm ./build/model_repository/postproc/config.pbtxt.tmpl

python update_template.py \
--input_file ./build/model_repository/tokenize/tokenizer_config.json.tmpl \
--outputp_file ./build/model_repository/tokenize/tokenizer_config.json \
--model $TOKENIZER_NAME \
--max_length $TOKENIZER_MAX_LENGTH \
--output_type_ids $TOKENIZER_OUTPUT_TYPE_IDS
rm ./build/model_repository/tokenize/tokenizer_config.json.tmpl

cd ./build
docker build -t custom_image:latest .
cd ..

docker run --rm -d --shm-size 1g \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  custom_image:latest \
  tritonserver --model-repository=/model_repository
sleep 20
curl -v localhost:8000/v2/health/ready

az account set --subscription $MODEL_SUBSCRIPTION_ID

export IMAGE_REPO=$IMAGE_REGISTRY_SERVER/$IMAGE_NAME
export IMAGE_TAG=$IMAGE_TAG_PREFIX.latest

docker tag custom_image:latest $IMAGE_REPO:$IMAGE_TAG
docker push $IMAGE_REPO:$IMAGE_TAG

cd ..
rm -r ./tmp
