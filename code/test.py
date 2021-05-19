from transformers import BertTokenizer, Trainer
from scipy.special import softmax
from model import GAIICBert
from dataset import GAIICDataset
import config


tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
lines = open(config.testB_path).readlines()
test_dataset = GAIICDataset(tokenizer, lines, 2)
results = 0
for i in range(config.folds):
    model = GAIICBert.from_pretrained(config.model_path + str(i))
    trainer = Trainer(model)
    outputs = trainer.predict(test_dataset)
    results += softmax(outputs[0], 1)[:, 1]
results /= config.folds
with open(config.result_path, 'w') as f:
    f.write('\n'.join(map(str, results)))
