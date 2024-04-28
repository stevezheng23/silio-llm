FROM nvcr.io/nvidia/tritonserver:24.03-py3
LABEL author.name="Mingzhi Zheng"
LABEL author.email="stevezheng23@gmail.com"

RUN pip --no-cache-dir install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY model_repository /model_repository
