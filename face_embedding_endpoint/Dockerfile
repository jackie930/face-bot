ARG REGISTRY_URI
FROM ${REGISTRY_URI}/mxnet-inference:1.6.0-cpu-py3

RUN mkdir -p /opt/ml/model

# COPY package/ /opt/ml/code/package/

# COPY serve.py /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install -i https://opentuna.cn/pypi/web/simple/ flask gevent gunicorn insightface

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY deploy /opt/program/deploy
COPY serve.py /opt/program/
COPY predictor.py /opt/program/
COPY wsgi.py /opt/program/
COPY nginx.conf /opt/program/

# Copy pretrained model
COPY model/model-r100-ii /opt/program/model-r100-ii
COPY model/mtcnn-model /opt/program/mtcnn-model

WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]
