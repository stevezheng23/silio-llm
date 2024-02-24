FROM silio.azurecr.io/tritonserver:23.02-py3-cpu.latest
LABEL author.name="Mingzhi Zheng"
LABEL author.email="stevezheng23@gmail.com"

RUN pip --no-cache-dir install tokenizer==0.13.2
RUN pip --no-cache-dir install scipy==1.8.0

COPY model_repository /model_repository
