FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3
RUN pip install transformers onnxruntime-gpu==1.4 scikit-learn flask --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY library/cudnn7 /usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH
COPY user_data user_data
COPY code code
WORKDIR /workspace/code
CMD ["sh", "run.sh"]
