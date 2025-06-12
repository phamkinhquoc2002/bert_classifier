import os

import torch
import hydra

import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import FSDPStrategy
from pytorch_lightning.loggers import WandbLogger

from dataloader import DataModule
from model import DistilBert
from samples_logger import SamplesLogger

@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg):
    #Configuration Setting
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))
    #Data Loader
    data = DataModule(path=cfg.training.data_path, tokenizer=cfg.model.tokenizer, batch_size=cfg.training.batch_size)
    data.prepare_data()
    data.setup(stage="fit")
    model = DistilBert(model_name=cfg.model.model_name, 
                       num_classes=cfg.training.num_classes,
                       optim=cfg.training.optimizer)
    #Callbacks and Loggers
    checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min", filename="last", save_last=True, 
        )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(project="vietnamese-classification")
    sample_logger = SamplesLogger(data_module=data)
    #Distributed Training Settings
    strategy = cfg.training.distributed_strategy
    if strategy == "FSDP":
        distributed_strategy = FSDPStrategy(
            sharding_strategy= "FULL_SHARD"
        )
    elif strategy == "DDP":
        distributed_strategy = "ddp"
    elif strategy == "single":
        distributed_strategy = "auto"
    #Trainer Configuration
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=(-1 if torch.cuda.is_available() else 0),
        max_epochs=cfg.training.epochs,
        strategy=distributed_strategy,
        logger=wandb_logger,
        deterministic=cfg.training.deterministic,
        log_every_n_steps=cfg.training.log_every_n_steps, 
        callbacks=[checkpoint_callback, early_stopping_callback, sample_logger]
    )
    #Checkpoint Configuration
    resume_ckpt_path=None
    if cfg.training.resume_from_checkpoint:
        resume_ckpt_path = os.path.abspath(cfg.training.resume_from_checkpoint)
        if not os.path.exists(resume_ckpt_path):
            print(f"Warning: Checkpoint path {resume_ckpt_path} does not exist. Starting fresh training.")
            resume_ckpt_path = None
    elif checkpoint_callback.last_model_path and os.path.exists(checkpoint_callback.last_model_path):
        resume_ckpt_path = checkpoint_callback.last_model_path
    #Start the training
    trainer.fit(model, data, ckpt_path=resume_ckpt_path)

if __name__ == "__main__":
    main()