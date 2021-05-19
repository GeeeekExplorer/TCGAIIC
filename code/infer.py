# 采用自己实现服务的方式
# 参考docker：registry.cn-hangzhou.aliyuncs.com/zhyuyang/zhyuyang:0.3
import onnxruntime
# from torch2trt import torch2trt
# from math import exp
from scipy.special import softmax
from flask import Flask, request
from transformers import BertTokenizer, BertForSequenceClassification
from model import GAIICBert
import config
import os

app = Flask(__name__)


# 正式测试，batch_size固定为1
@app.route('/tccapi', methods=['GET', 'POST'])
def tccapi():
    data = request.get_data()
    if data == b'exit':
        print('received exit command, exit now')
        exit(0)

    input_list = request.form.getlist('input')
    index_list = request.form.getlist('index')

    response_batch = {'results': []}
    for i in range(len(index_list)):
        index_str = index_list[i]
        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split('\t')
            sa = elems[0].strip()
            sb = elems[1].strip()
            predict = infer(models, tokenizer, sa, sb)
            response['predict'] = predict
            response['index'] = index_str
            response['ok'] = True
        except Exception:
            response['predict'] = 0
            response['index'] = index_str
            response['ok'] = False
        response_batch['results'].append(response)

    return response_batch


# 需要根据模型类型重写
def infer(models, tokenizer, sa, sb):
    # encoding = tokenizer(sa, sb, return_tensors='pt')
    # with torch.no_grad():
    #     outputs = model(**{k: v.cuda() for k, v in encoding.items()})[0][0]
    # neg, pos = outputs
    # predict = exp(pos) / (exp(neg) + exp(pos))
    #     if encoding['input_ids'].size(1) > 10:
    #         outputs = [models[0](**{k: v.cuda() for k, v in encoding.items()})[0][0].tolist()]
    #     else:
    #         outputs = [model(**{k: v.cuda() for k, v in encoding.items()})[0][0].tolist() for model in models]
    encoding = tokenizer(sa, sb, truncation=True, max_length=32, return_tensors='np')
    outputs = [model.run([], dict(encoding))[0][0].tolist() for model in models]
    predict = softmax(outputs, 1)[:, 1].mean()
    return predict


# 需要根据模型类型重写
def init(tokenizer_path, model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    # model = BertForSequenceClassification.from_pretrained(model_path, return_dict=False).cuda()
    # model = GAIICBert.from_pretrained(model_path).cuda()
    models = [onnxruntime.InferenceSession(f'{config.onnx_path}{i}/model.onnx') for i in range(6)]
    # dummy = torch.ones(1, config.max_seq_length, dtype=torch.int)
    # model = torch2trt(model, [dummy.cuda()])
    return models, tokenizer


if __name__ == '__main__':
    models, tokenizer = init(config.vocab_path, config.model_path)
    app.run(host='0.0.0.0', port=8080)
