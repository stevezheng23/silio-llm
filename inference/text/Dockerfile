FROM silio.azurecr.io/tritonserver:23.02-py3-cpu.latest
LABEL author.name="Mingzhi Zheng"
LABEL author.email="stevezheng23@gmail.com"

RUN pip3 --no-cache-dir install nltk==3.6.7
RUN pip3 --no-cache-dir install spacy==3.5.0

RUN python3 -m spacy download en_core_web_lg

COPY model_repository /model_repository
