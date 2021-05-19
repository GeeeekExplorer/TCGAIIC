from collections import Counter
import config


def read(filepath):
    with open(filepath) as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines if line]
    if 'train' in filepath:
        lines = [line[:line.rfind('\t')] for line in lines]
    return lines


def count(lines):
    counter = {}
    for line in lines:
        for word in line.split():
            counter[word] = counter.get(word, 0) + 1
    return Counter(counter)


paths = [config.train1_path, config.testA_path, config.testB_path, config.train2_path]
files = [read(path) for path in paths]
with open(config.all_path, 'w') as fp:
    for lines in files:
        for line in lines:
            fp.writelines(line + '\n')
counters = [count(lines) for lines in files]
words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[BOS]', '[EOS]']
for i in range(1, 4):
    counter = sum(counters[:i + 1], Counter())
    counter = {k: v for k, v in counter.items() if v >= config.min_freq}
    for word in words:
        if word in counter:
            counter.pop(word)
    words.extend(sorted(counter.keys(), key=int))
with open(config.vocab_path, 'w') as fp:
    fp.writelines('\n'.join(words))
