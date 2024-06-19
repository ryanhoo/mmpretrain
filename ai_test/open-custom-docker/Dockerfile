FROM python:3.10.12-slim

# 设置非交互式安装，防止在安装过程中出现提示
# ENV DEBIAN_FRONTEND=noninteractive

# 安装 Yolo v8
RUN pip install ultralytics fiftyone

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libusb-1.0-0 curl

WORKDIR /app

# CMD ["python", "train.py"]
CMD ./start.sh

# docker build -t open-custom-docker .

# docker run --gpus all --shm-size=32g -v .:/app open-custom-docker