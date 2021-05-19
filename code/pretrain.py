import os
import glob2
from transformers import BertTokenizer, BertModel, BertForMaskedLM, Trainer, TrainingArguments
from dataset import GAIICDataset
from datacollator import GAIICDataCollator
from random import shuffle
import config

names = ['hfl/chinese-macbert-base', 'bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'hfl/chinese-bert-wwm-ext']
index = len(glob2.glob(config.pretrain_path + "*.txt"))
name = names[index]
index = str(index)
os.mknod(config.pretrain_path + index + ".txt")
os.environ['CUDA_VISIBLE_DEVICES'] = index
tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
lines = open(config.all_path).readlines()
# model = BertModel.from_pretrained(name, return_dict=False, mirror='bfsu', cache_dir=config.hf_path + name[name.rfind('/')+1:])
# model.save_pretrained('/mnt/' + name[name.rfind('/') + 1:])
model = BertForMaskedLM.from_pretrained(config.hf_path + name[name.rfind('/') + 1:])
model.resize_token_embeddings(config.vocab_size)
for _ in range(config.folds):
    shuffle(lines)
    args = TrainingArguments(output_dir=config.pretrain_path + index, save_total_limit=1,
                             learning_rate=config.learning_rate, num_train_epochs=config.epochs,
                             per_device_train_batch_size=config.batch_size)
    dataset = GAIICDataset(tokenizer, lines, 0)
    data_collator = GAIICDataCollator(tokenizer)
    trainer = Trainer(model, args, data_collator, dataset)
    trainer.train()
    trainer.save_model(config.pretrain_path + index)
    config.learning_rate -= 1e-5
