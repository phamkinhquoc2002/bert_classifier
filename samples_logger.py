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
        val_batch = {
            k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
            for k, v in val_batch.items()
        }
        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        pred = torch.argmax(outputs, dim=1)
        label = val_batch["label"]

        df = pd.DataFrame({"label": label.cpu().numpy(), "prediction": pred.cpu().numpy()})
        wrong_df = df[df["label"] != df["prediction"]]

        trainer.logger.log_metrics(metrics={"key": wandb.Table(dataframe=wrong_df, allow_mixed_types=True)}
        )