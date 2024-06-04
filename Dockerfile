FROM huggingface/transformers-pytorch-gpu:4.41.2
#看看什么系统，python位置
WORKDIR /root
COPY ./run.py .
RUN python run.py
