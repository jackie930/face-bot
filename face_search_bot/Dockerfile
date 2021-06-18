ARG REGISTRY_URI
FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-cpu-py3

RUN mkdir -p /opt/program

ENV LANG=en_US.utf8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip install boto3 -i https://opentuna.cn/pypi/web/simple
RUN pip install opencv-python==3.4.11.41 -i https://opentuna.cn/pypi/web/simple
RUN pip install elasticsearch==7.8.0 -i https://opentuna.cn/pypi/web/simple
RUN pip install requests -i https://opentuna.cn/pypi/web/simple
RUN pip install annoy==1.15.2 -i https://opentuna.cn/pypi/web/simple

COPY . /opt/program/

WORKDIR /opt/program

ENTRYPOINT ["python", "-u", "task.py"]
