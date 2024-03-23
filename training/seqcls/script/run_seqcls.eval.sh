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
      --dsname=*)
      DSNAME="${i#*=}"
      shift
      ;;
      --evalblob=*)
      EVALBLOB="${i#*=}"
      shift
      ;;
      --evalfile=*)
      EVALFILE="${i#*=}"
      shift
      ;;
      --labelblob=*)
      LABELBLOB="${i#*=}"
      shift
      ;;
      --labelfile=*)
      LABELFILE="${i#*=}"
      shift
      ;;
      --maxseqlen=*)
      MAXSEQLEN="${i#*=}"
      shift
      ;;
      --batchsize=*)
      BATCHSIZE="${i#*=}"
      shift
      ;;
    esac
  done

echo "train run id          = ${RUNID}"
echo "artifact path         = ${ARTIFACT}"
echo "datastore name        = ${DSNAME}"
echo "eval blob path        = ${EVALBLOB}"
echo "eval file             = ${EVALFILE}"
echo "label blob path       = ${LABELBLOB}"
echo "label file            = ${LABELFILE}"
echo "max seq len           = ${MAXSEQLEN}"
echo "batch size            = ${BATCHSIZE}"

mkdir -p ./tmp/
python ./src/download_model.py --run_id ${RUNID} --artifact_path ${ARTIFACT} --local_dir ./tmp/
mv ./tmp/${ARTIFACT} ./tmp/artifact
ls -l ./tmp/artifact

python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${EVALBLOB} --local_path ./
python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${LABELBLOB} --local_path ./

mkdir -p ./tmp/data/
cp ./tmp/artifact/predict_labels.txt ./tmp/data/${LABELFILE}
cp ${EVALBLOB} ./tmp/data/${EVALFILE}
cp ${LABELBLOB} ./tmp/data/${LABELFILE}
rm ${EVALBLOB}
rm ${LABELBLOB}
ls -l ./tmp/data/

python ./src/run_seqcls.py \
--model_name_or_path=./tmp/artifact \
--do_predict \
--train_file=./tmp/data/${EVALFILE} \
--validation_file=./tmp/data/${EVALFILE} \
--test_file=./tmp/data/${EVALFILE} \
--label_file=./tmp/data/${LABELFILE} \
--max_seq_length=${MAXSEQLEN} \
--per_device_eval_batch_size=${BATCHSIZE} \
--output_dir=./tmp/artifact \
--report_to=mlflow

mkdir -p ./tmp/result/
mv ./tmp/artifact/predict_results.json ./tmp/result/
mv ./tmp/artifact/predict_labels.txt ./tmp/result/
mv ./tmp/artifact/threshold_optimal.json ./tmp/result/

python ./src/apply_threshold.py \
--eval_file=./tmp/result/predict_results.json \
--label_file=./tmp/result/predict_labels.txt \
--threshold_file=./tmp/result/threshold_optimal.json \
--default_label="default.skip" \
--output_dir=./tmp/result/

python ./src/eval_result.py \
--eval_file=./tmp/result/optimal_results.json \
--label_file=./tmp/result/optimal_labels.txt \
--output_dir=./tmp/result/

mv ./tmp/result/classification_report.txt ./tmp/result/optimal_classification_report.txt
mv ./tmp/result/confusion_matrix.png ./tmp/result/optimal_confusion_matrix.png

python ./src/eval_result.py \
--eval_file=./tmp/result/predict_results.json \
--label_file=./tmp/result/predict_labels.txt \
--output_dir=./tmp/result/

python ./src/upload_result.py --result_dir=./tmp/result
