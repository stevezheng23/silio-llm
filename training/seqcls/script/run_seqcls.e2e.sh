#!/bin/sh
for i in "$@"
  do
    case $i in
      --numgpus=*)
      NUMGPUS="${i#*=}"
      shift
      ;;
      --randomseed=*)
      RANDOMSEED="${i#*=}"
      shift
      ;;
      --dsname=*)
      DSNAME="${i#*=}"
      shift
      ;;
      --trainblob=*)
      TRAINBLOB="${i#*=}"
      shift
      ;;
      --trainfile=*)
      TRAINFILE="${i#*=}"
      shift
      ;;
      --devblob=*)
      DEVBLOB="${i#*=}"
      shift
      ;;
      --devfile=*)
      DEVFILE="${i#*=}"
      shift
      ;;
      --testblob=*)
      TESTBLOB="${i#*=}"
      shift
      ;;
      --testfile=*)
      TESTFILE="${i#*=}"
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
      --outputdir=*)
      OUTPUTDIR="${i#*=}"
      shift
      ;;
      --modeldir=*)
      MODELDIR="${i#*=}"
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
      --learningrate=*)
      LEARNINGRATE="${i#*=}"
      shift
      ;;
      --numepochs=*)
      NUMEPOCHS="${i#*=}"
      shift
      ;;
      --gradaccsteps=*)
      GRADACCSTEPS="${i#*=}"
      shift
      ;;
      --weightdecay=*)
      WEIGHTDECAY="${i#*=}"
      shift
      ;;
      --warmupsteps=*)
      WARMUPSTEPS="${i#*=}"
      shift
      ;;
      --evalsteps=*)
      EVALSTEPS="${i#*=}"
      shift
      ;;
      --savelimit=*)
      SAVELIMIT="${i#*=}"
      shift
      ;;
      --registeredname=*)
      REGISTEREDNAME="${i#*=}"
      shift
      ;;
    esac
  done

echo "num gpus              = ${NUMGPUS}"
echo "random seed           = ${RANDOMSEED}"
echo "datastore name        = ${DSNAME}"
echo "train blob path       = ${TRAINBLOB}"
echo "train file            = ${TRAINFILE}"
echo "dev blob path         = ${DEVBLOB}"
echo "dev file              = ${DEVFILE}"
echo "test blob path        = ${TESTBLOB}"
echo "test file             = ${TESTFILE}"
echo "label blob path       = ${LABELBLOB}"
echo "label file            = ${LABELFILE}"
echo "output dir            = ${OUTPUTDIR}"
echo "model dir             = ${MODELDIR}"
echo "max seq len           = ${MAXSEQLEN}"
echo "batch size            = ${BATCHSIZE}"
echo "learning rate         = ${LEARNINGRATE}"
echo "num epochs            = ${NUMEPOCHS}"
echo "grad acc steps        = ${GRADACCSTEPS}"
echo "weight decay          = ${WEIGHTDECAY}"
echo "warmup steps          = ${WARMUPSTEPS}"
echo "eval steps            = ${EVALSTEPS}"
echo "save limit            = ${SAVELIMIT}"
echo "registered name       = ${REGISTEREDNAME}"

python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${TRAINBLOB} --local_path ./
python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${DEVBLOB} --local_path ./
python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${TESTBLOB} --local_path ./
python ./src/download_data.py --ds_name=${DSNAME} --blob_path=${LABELBLOB} --local_path ./

mkdir -p ./tmp/data/
cp ${TRAINBLOB} ./tmp/data/${TRAINFILE}
cp ${DEVBLOB} ./tmp/data/${DEVFILE}
cp ${TESTBLOB} ./tmp/data/${TESTFILE}
cp ${LABELBLOB} ./tmp/data/${LABELFILE}
rm ${TRAINBLOB}
rm ${DEVBLOB}
rm ${TESTBLOB}
rm ${LABELBLOB}
ls -l ./tmp/data/

if [ ${NUMGPUS} -gt 1 ]
then
  python -m torch.distributed.launch --nproc_per_node=${NUMGPUS} ./src/run_seqcls.py \
  --model_name_or_path=${MODELDIR} \
  --do_train \
  --do_eval \
  --train_file=./tmp/data/${TRAINFILE} \
  --validation_file=./tmp/data/${DEVFILE} \
  --label_file=./tmp/data/${LABELFILE} \
  --max_seq_length=${MAXSEQLEN} \
  --per_device_train_batch_size=${BATCHSIZE} \
  --per_device_eval_batch_size=${BATCHSIZE} \
  --gradient_accumulation_steps ${GRADACCSTEPS} \
  --learning_rate=${LEARNINGRATE} \
  --num_train_epochs=${NUMEPOCHS} \
  --weight_decay=${WEIGHTDECAY} \
  --warmup_steps=${WARMUPSTEPS} \
  --logging_steps=50 \
  --save_step=${EVALSTEPS} \
  --eval_step=${EVALSTEPS} \
  --evaluation_strategy=steps \
  --overwrite_output_dir \
  --output_dir=${OUTPUTDIR} \
  --seed=${RANDOMSEED} \
  --report_to=mlflow \
  --save_total_limit=${SAVELIMIT} \
  --metric_for_best_model=accuracy
else
  python ./src/run_seqcls.py \
  --model_name_or_path=${MODELDIR} \
  --do_train \
  --do_eval \
  --train_file=./tmp/data/${TRAINFILE} \
  --validation_file=./tmp/data/${DEVFILE} \
  --label_file=./tmp/data/${LABELFILE} \
  --max_seq_length=${MAXSEQLEN} \
  --per_device_train_batch_size=${BATCHSIZE} \
  --per_device_eval_batch_size=${BATCHSIZE} \
  --gradient_accumulation_steps=${GRADACCSTEPS} \
  --learning_rate=${LEARNINGRATE} \
  --num_train_epochs=${NUMEPOCHS} \
  --weight_decay=${WEIGHTDECAY} \
  --warmup_steps=${WARMUPSTEPS} \
  --logging_steps=50 \
  --save_step=${EVALSTEPS} \
  --eval_step=${EVALSTEPS} \
  --evaluation_strategy=steps \
  --overwrite_output_dir \
  --output_dir=${OUTPUTDIR} \
  --seed=${RANDOMSEED} \
  --report_to=mlflow \
  --save_total_limit=${SAVELIMIT} \
  --metric_for_best_model=accuracy
fi

python ./src/run_seqcls.py \
--model_name_or_path=${OUTPUTDIR} \
--do_predict \
--train_file=./tmp/data/${TRAINFILE} \
--validation_file=./tmp/data/${DEVFILE} \
--test_file=./tmp/data/${DEVFILE} \
--label_file=./tmp/data/${LABELFILE} \
--max_seq_length=${MAXSEQLEN} \
--per_device_eval_batch_size=${BATCHSIZE} \
--output_dir=${OUTPUTDIR} \
--seed=${RANDOMSEED} \
--report_to=mlflow

python ./src/find_threshold.py \
--eval_file=${OUTPUTDIR}/predict_results.json \
--label_file=${OUTPUTDIR}/predict_labels.txt \
--output_dir=${OUTPUTDIR} \
--search_strategy="p/r" \
--optimized_metric="precision" \
--constraint_metric="recall" \
--min_score=0.2

python ./src/apply_threshold.py \
--eval_file=${OUTPUTDIR}/predict_results.json \
--label_file=${OUTPUTDIR}/predict_labels.txt \
--threshold_file=${OUTPUTDIR}/threshold_optimal.json \
--default_label="default.skip" \
--output_dir=${OUTPUTDIR}

python ./src/eval_result.py --eval_file=${OUTPUTDIR}/optimal_results.json --label_file=${OUTPUTDIR}/optimal_labels.txt --output_dir=${OUTPUTDIR}
mv ${OUTPUTDIR}/optimal_results.json ${OUTPUTDIR}/optimal_results.dev.json
mv ${OUTPUTDIR}/optimal_labels.txt ${OUTPUTDIR}/optimal_labels.dev.txt
mv ${OUTPUTDIR}/classification_report.txt ${OUTPUTDIR}/optimal_classification_report.dev.txt
mv ${OUTPUTDIR}/confusion_matrix.png ${OUTPUTDIR}/optimal_confusion_matrix.dev.png

python ./src/eval_result.py --eval_file=${OUTPUTDIR}/predict_results.json --label_file=${OUTPUTDIR}/predict_labels.txt --output_dir=${OUTPUTDIR}
mv ${OUTPUTDIR}/predict_results.json ${OUTPUTDIR}/predict_results.dev.json
mv ${OUTPUTDIR}/predict_labels.txt ${OUTPUTDIR}/predict_labels.dev.txt
mv ${OUTPUTDIR}/predict_args.json ${OUTPUTDIR}/predict_args.dev.json
mv ${OUTPUTDIR}/classification_report.txt ${OUTPUTDIR}/classification_report.dev.txt
mv ${OUTPUTDIR}/confusion_matrix.png ${OUTPUTDIR}/confusion_matrix.dev.png

python ./src/run_seqcls.py \
--model_name_or_path=${OUTPUTDIR} \
--do_predict \
--train_file=./tmp/data/${TRAINFILE} \
--validation_file=./tmp/data/${DEVFILE} \
--test_file=./tmp/data/${TESTFILE} \
--label_file=./tmp/data/${LABELFILE} \
--max_seq_length=${MAXSEQLEN} \
--per_device_eval_batch_size=${BATCHSIZE} \
--output_dir=${OUTPUTDIR} \
--seed=${RANDOMSEED} \
--report_to=mlflow

python ./src/apply_threshold.py \
--eval_file=${OUTPUTDIR}/predict_results.json \
--label_file=${OUTPUTDIR}/predict_labels.txt \
--threshold_file=${OUTPUTDIR}/threshold_optimal.json \
--default_label="default.skip" \
--output_dir=${OUTPUTDIR}

python ./src/eval_result.py --eval_file=${OUTPUTDIR}/optimal_results.json --label_file=${OUTPUTDIR}/optimal_labels.txt --output_dir=${OUTPUTDIR}
mv ${OUTPUTDIR}/optimal_results.json ${OUTPUTDIR}/optimal_results.test.json
mv ${OUTPUTDIR}/optimal_labels.txt ${OUTPUTDIR}/optimal_labels.test.txt
mv ${OUTPUTDIR}/classification_report.txt ${OUTPUTDIR}/optimal_classification_report.test.txt
mv ${OUTPUTDIR}/confusion_matrix.png ${OUTPUTDIR}/optimal_confusion_matrix.test.png

python ./src/eval_result.py --eval_file=${OUTPUTDIR}/predict_results.json --label_file=${OUTPUTDIR}/predict_labels.txt --output_dir=${OUTPUTDIR}
mv ${OUTPUTDIR}/predict_results.json ${OUTPUTDIR}/predict_results.test.json
mv ${OUTPUTDIR}/predict_labels.txt ${OUTPUTDIR}/predict_labels.test.txt
mv ${OUTPUTDIR}/predict_args.json ${OUTPUTDIR}/predict_args.test.json
mv ${OUTPUTDIR}/classification_report.txt ${OUTPUTDIR}/classification_report.test.txt
mv ${OUTPUTDIR}/confusion_matrix.png ${OUTPUTDIR}/confusion_matrix.test.png

rm -r ${OUTPUTDIR}/checkpoint-*
mkdir -p ./tmp/model/
mv ${OUTPUTDIR} ./tmp/model/pytorch/

python ./src/convert_model.py \
--model_dir=./tmp/model/pytorch/ \
--onnx_dir=./tmp/model/onnx/model.onnx \
--task_name=sentiment-analysis

cp ./tmp/model/pytorch/predict_labels.dev.txt ./tmp/model/onnx/labels.txt
cp ./tmp/model/pytorch/threshold_optimal.json ./tmp/model/onnx/threshold.json

if [ -z ${REGISTEREDNAME} ]
then
  python ./src/register_model.py --model_dir=./tmp/model/pytorch --onnx_dir=./tmp/model/onnx
else
  python ./src/register_model.py --model_dir=./tmp/model/pytorch --onnx_dir=./tmp/model/onnx --registered_name=${REGISTEREDNAME}
fi
