#!/bin/sh
for i in "$@"
  do
    case $i in
      --runid=*)
      RUNID="${i#*=}"
      shift
      ;;
      --artifact=*)
      ARTIFACT="${i#*=}"
      shift
      ;;
      --registeredname=*)
      REGISTEREDNAME="${i#*=}"
      shift
      ;;
    esac
  done

echo "train run id          = ${RUNID}"
echo "artifact path         = ${ARTIFACT}"
echo "registered name       = ${REGISTEREDNAME}"

mkdir -p ./tmp/
python ./src/download_mdoel.py --run_id ${RUNID} --artifact_path ${ARTIFACT} --local_dir ./tmp/
mkdir -p ./tmp/model/
mv ./tmp/${ARTIFACT} ./tmp/model/pytorch/
ls -l ./tmp/model/pytorch/

python ./src/convert_model.py \
--model_dir=./tmp/model/pytorch/ \
--onnx_dir=./tmp/model/onnx/model.onnx \
--task_name=sentiment-analysis

cp ./tmp/model/pytorch/predict_labels.txt ./tmp/model/onnx/labels.txt

if [ -z ${REGISTEREDNAME} ]
then
  python ./src/register_model.py --model_dir=./tmp/model/pytorch --onnx_dir=./tmp/model/onnx
else
  python ./src/register_model.py --model_dir=./tmp/model/pytorch --onnx_dir=./tmp/model/onnx --registered_name=${REGISTEREDNAME}
fi
