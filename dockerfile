FROM nvidia/cuda:11.2.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-opencv

RUN mkdir /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["./run.sh"]