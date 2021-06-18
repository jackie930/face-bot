ARG REGISTRY_URI
FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-cpu-py3

RUN mkdir -p /opt/program

ENV LANG=en_US.utf8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip install -i https://opentuna.cn/pypi/web/simple boto3 opencv-python==3.4.11.41 elasticsearch==7.8.0 requests

COPY . /opt/program/

WORKDIR /opt/program

CMD ["python", "task.py"]
ENTRYPOINT ["python", "task.py"]
