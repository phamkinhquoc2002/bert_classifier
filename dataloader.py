import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def validation_split(ds: Dataset):
    ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    test = ds['train']
    val = ds['test']
    return test, val

class DataModule(pl.LightningDataModule):
    def __init__(self, path: str, tokenizer: AutoTokenizer, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.tokenizer = tokenizer

    def prepare_data(self):
        dataset = load_dataset(path=self.path)
        self.train_data = dataset['train']
        if dataset['validation']:
            self.test_data = dataset['test']
            self.validation_data = dataset['validation']
        else:
            self.test_data, self.validation_data = validation_split(dataset['test'])

    def tokenize(self, example):
        return self.tokenizer(
            example['text'],
            truncation=True,
            padding="max_length",
            max_length=self._maximum_length
        )
    
    def setup(self):
        self.train_data= self.train_data.map(self.tokenize, batched=True)
        self.test_data=self.train_data.map(self.tokenize, batched=True)
        self.validation_data=self.validation_data.map(self.tokenize, batched=True)

        self.train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.validation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def _maximum_length(self):
        max_length = 0
        for example in self.train_data:
            if len(example['text'].split()) > max_length:
                max_length=len(example['text'].split())
        return max_length

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
    
    def validation_save(self):
        return torch.utils.data.DataLoader(
            self.validation_data, batch_size=self.batch_size, shuffle=False
        )