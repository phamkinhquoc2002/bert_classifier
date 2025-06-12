import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

def validation_split(ds: Dataset):
    print("Inside validation_split function:")
    print(f"Input dataset: {ds}")
    split = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
    print(f"Split dataset: {split}")
    return split['train'], split['test']

class DataModule(pl.LightningDataModule):
    def __init__(self, path: str, tokenizer: str, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token is None:self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self):
        try:
            self.dataset = load_dataset(path=self.path)
            print("\nDataset loaded in prepare_data:")
            print(self.dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e
    
    def label_tokenize(self, examples):
        examples["label"] = [self.classes_map.get(label) for label in examples["label"]]
        return examples
                
    def text_tokenize(self, example):
        return self.tokenizer(
            example['text'],
            truncation=True,
            return_attention_mask=True,
            padding="max_length",
            padding_side="right",
            max_length=1024
        )

    def setup(self, stage: str = None):
        print("\nInside setup function:")
        print(f"self.dataset: {self.dataset}")
        if self.dataset is not None and 'train' in self.dataset and 'test' in self.dataset:
            print("Using existing train and test splits.")
            self.train_data, self.validation_data = self.dataset['train'], self.dataset['test']
            print(f"self.train_data (before map): {self.train_data}")
            print(f"self.validation_data (before map): {self.validation_data}")
        elif self.dataset is not None:
            print("Attempting to split the entire dataset.")
            self.train_data, self.validation_data = validation_split(self.dataset)
        else:
            raise ValueError(f"Dataset loaded from '{self.path}' is None.")
        
        assert type(self.train_data)==Dataset, f"Train Data set after split: {type(self.train_data)}"
        assert type(self.validation_data)==Dataset, f"Validation Data set after split: {type(self.validation_data)}"

        self.classes_map = {label: idx for idx, label in enumerate(sorted(set(self.train_data["label"])))}

        self.train_data = self.train_data.map(self.text_tokenize, batched=True, remove_columns=["text"])
        self.train_data = self.train_data.map(self.label_tokenize, batched=True)

        self.validation_data = self.validation_data.map(self.text_tokenize, batched=True, remove_columns=["text"])
        self.validation_data = self.validation_data.map(self.label_tokenize, batched=True)

        self.train_data = self.train_data.with_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.validation_data = self.validation_data.with_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    def train_dataloader(self):
        print("\nInside train_dataloader function:")
        print(f"self.train_data: {self.train_data}")
        if self.validation_data is None:
            raise ValueError("self.validation_data is None — did you forget to set it?")
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=6, persistent_workers=True
        )

    def val_dataloader(self):
        print("\nInside val_dataloader function:")
        print(f"self.validation_data: {self.validation_data}")
        if self.validation_data is None:
            raise ValueError("self.validation_data is None — did you forget to set it?")
        return DataLoader(
            self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=6, persistent_workers=True
        )