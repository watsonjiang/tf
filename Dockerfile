FROM tensorflow/tensorflow:2.2.0-gpu

WORKDIR /root

COPY tflearn-*-py3-none-any.whl .

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install tflearn-*-py3-none-any.whl -i https://mirrors.aliyun.com/pypi/simple/

