import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from dataloader import DataModule

class SamplesLogger(pl.Callback):
    def __init__(self, data_module: DataModule):
        self.data_module = data_module
    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.data_module.val_dataloader()))
        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        pred = torch.argmax(outputs.logits, dim=1)
        label = val_batch["label"]
        df = pd.DataFrame({"label": label, "prediction": pred.numpy()})

        wrong_df = df[df["label"] != df["prediction"]]

        trainer.logger.experiment(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )