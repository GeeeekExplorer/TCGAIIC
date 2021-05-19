FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3
RUN pip install transformers onnxruntime-gpu==1.4 scikit-learn flask --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY library/cudnn7 /usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH
COPY user_data user_data
COPY code code
WORKDIR /workspace/code
CMD ["sh", "run.sh"]


#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3
#ENV CUDA_HOME=/usr/local/cuda
#ENV LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH
##RUN apt update && apt install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
#RUN pip install transformers scikit-learn flask --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
#COPY code/modeling_bert.py /opt/conda/lib/python3.8/site-packages/transformers/models/bert/
#COPY library/cudnn8 /usr/lib/x86_64-linux-gnu/
#COPY library/TensorRT-7.2.3.4/lib /usr/lib/x86_64-linux-gnu/
#COPY library/TensorRT-7.2.3.4/include /usr/include/x86_64-linux-gnu/
#COPY library/torch2trt-0.2.0 /workspace/library/torch2trt-0.2.0/
#WORKDIR /workspace/library/torch2trt-0.2.0/
#RUN pip install ./tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
#RUN python setup.py install
#WORKDIR /workspace/
#COPY tcdata tcdata
#COPY user_data user_data
#COPY code code
#WORKDIR /workspace/code
#CMD ["sh", "run.sh"]