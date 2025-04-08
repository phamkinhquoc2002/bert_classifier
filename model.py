import torch
import torch.functional as F
import pytorch_lightning as pl
from config import ModelConfig
from transformers import AutoModelForCausalLM

class DistilBert(pl.LightningModule):
    def __init__(self, model_config: ModelConfig):
        super(DistilBert, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name=model_config['model_name'])
        self.out_head = torch.nn.Linear(self.model.lm_head.out_features, model_config['num_classes'])
        