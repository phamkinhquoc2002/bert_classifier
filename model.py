import torch
import pytorch_lightning as pl
from config import ModelConfig, TrainingConfig
from transformers import AutoModelForCausalLM
from torchmetrics.classification import MulticlassAccuracy

class DistilBert(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super(DistilBert, self).__init__()
        self.save_hyperparameters()
        self.training_config = training_config
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_config['model_name'])
        self.out_head = torch.nn.Linear(self.model.lm_head.out_features, model_config['num_classes'])
        self.train_accuracy = MulticlassAccuracy(num_classes=model_config["num_classes"])
        self.val_accuracy = MulticlassAccuracy(num_classes=model_config["num_classes"])

        
    def forward(self, input_ids, attention_mask):
        print(f"input_ids type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', None)}")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h_classes = outputs.logits[:, -1, :]
        logits = self.out_head(h_classes)
        return logits
    
    def training_step(self, batch, bach_idx):
        input_ids, attention_mask, label = batch["input_ids"], batch["attention_mask"], batch["label"]
        print(f"input_ids type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', None)}")
        logits = self.forward(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, label)
        _, preds = torch.max(logits, dim=1)
        train_acc = self.train_accuracy(preds, label)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch["input_ids"], batch["attention_mask"], batch["label"]
        print(f"input_ids type: {type(input_ids)}, shape: {getattr(input_ids, 'shape', None)}")
        logits = self.forward(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, label)
        _, preds = torch.max(logits, dim=1)
        val_acc = self.val_accuracy(preds, label)
        self.log("val_loss", loss,prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        optim = self.training_config["optmizer"]
        if optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=1e-2)
        if optim == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=1e-2)
        if optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=1e-2)  