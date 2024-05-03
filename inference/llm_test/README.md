launch triton server
```
cd ./inference/llm/pyt
bash setup.script
```

run grpc testing
```
pip install tritonclient[grpc]

cd ./inference/llm_test

python ./run_grpc.py \
--input_file=test_generate.pyt.json \
--output_dir=./output \
--task_name=generate \
--model_endpoint=host.docker.internal:8001 \
--model_name=generate \
--model_version=1
```

run http testing
```
pip install tritonclient[http]

cd ./inference/llm_test

python ./run_http.py \
--input_file=test_generate.pyt.json \
--output_dir=./output \
--task_name=generate \
--model_endpoint=host.docker.internal:8000 \
--model_name=generate \
--model_version=1
```
