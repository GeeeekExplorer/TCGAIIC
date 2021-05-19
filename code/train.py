import os
import glob2
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers.convert_graph_to_onnx import convert
from sklearn.metrics import roc_auc_score
from random import shuffle
from model import GAIICBert
from dataset import GAIICDataset
import config

names = ['hfl/chinese-macbert-base', 'bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'clue/roberta_chinese_base',
         'hfl/chinese-macbert-base', 'bert-base-chinese']
index = len(glob2.glob(config.model_path + '*.txt'))
name = names[index]
index = str(index)
os.mknod(config.model_path + index + '.txt')
config.epochs = 6
config.batch_size = 350
config.learning_rate = 4e-5
os.environ['CUDA_VISIBLE_DEVICES'] = str(int(index) % 4)
tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
tokenizer.save_pretrained(config.model_path + index)
lines1 = open(config.train1_path).readlines()
lines2 = open(config.train2_path).readlines()
lines = lines1 + lines2
fold_size = len(lines) // config.folds
shuffle(lines)
# for i in range(config.folds):
# model = GAIICBert.from_pretrained(config.pretrain_path)
model = BertForSequenceClassification.from_pretrained(config.pretrain_path + index)
args = TrainingArguments(output_dir=config.model_path + index, evaluation_strategy='steps', save_total_limit=1,
                         learning_rate=config.learning_rate, num_train_epochs=config.epochs,
                         per_device_train_batch_size=config.batch_size,
                         per_device_eval_batch_size=config.batch_size,
                         load_best_model_at_end=True, metric_for_best_model='auc')
train_dataset = GAIICDataset(tokenizer, lines[config.eval_size:], 1)
eval_dataset = GAIICDataset(tokenizer, lines[:config.eval_size], 1)
trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                  compute_metrics=lambda ep: {'auc': roc_auc_score(ep.label_ids, ep.predictions[:, 1])})
trainer.train()
trainer.save_model(config.model_path + index)
convert('pt', config.model_path + index, Path(config.onnx_path + index) / 'model.onnx', 11,
        pipeline_name="sentiment-analysis")
