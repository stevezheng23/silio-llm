FROM silio.azurecr.io/tritonserver:23.02-py3-cpu.latest
LABEL author.name="Mingzhi Zheng"
LABEL author.email="stevezheng23@gmail.com"

RUN pip3 --no-cache-dir install --pre onnxruntime-genai

COPY model_repository /model_repository
