from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import Dict, List
import torch
import config


class GAIICDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, lines: List[str], mode: int):
        lines = [line.strip() for line in lines if line]
        lines1 = [line.split('\t')[0] for line in lines]
        lines2 = [line.split('\t')[1] for line in lines]
        # examples1 = tokenizer(lines1, padding='max_length', truncation=True, max_length=config.max_seq_length,
        #                       return_tensors='pt', return_special_tokens_mask=(mode == 0))
        # examples2 = tokenizer(lines2, padding='max_length', truncation=True, max_length=config.max_seq_length,
        #                       return_tensors='pt', return_special_tokens_mask=(mode == 0))
        # examples1.pop('token_type_ids')
        # examples2.pop('token_type_ids')
        # position_ids = torch.arange(config.max_seq_length).expand(len(lines), -1)
        # examples1['position_ids'] = examples2['position_ids'] = position_ids
        # examples = {key: torch.cat([examples1[key], examples2[key]], 1) for key in examples1}
        examples = tokenizer(lines1, lines2, padding='max_length', truncation=True, max_length=config.max_seq_length,
                             return_tensors='pt', return_special_tokens_mask=(mode == 0))
        if mode == 1:
            examples['labels'] = torch.tensor([int(line.split('\t')[2]) for line in lines])
        self.examples = [{key: value[i] for key, value in examples.items()} for i in range(len(lines))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
