FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
CMD ["bash"]
