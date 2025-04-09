import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

from dataloader import DataModule
from model import DistilBert

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", 
                        type=str,
                        help="Bert Model Name")
    parser.add_argument("--path",
                        type=str,
                        help="Data Path")
    parser.add_argument("--num-classes",
                        type=int,
                        help="Total number of classification classes")
    parser.add_argument("--batch-size",
                        type=int,
                        help="Total number of batch size per accelerator")
    parser.add_argument("--optimizer",
                        type=str,
                        help="Optimizer to choose")
    parser.add_argument("--epochs",
                        type=int,
                        help="The total number of epochs")
    parser.add_argument("--distributed-strategy",
                        type=str,
                        help="The distributed Strategy that you want to use",
                        choices=["single", "FSDP", "DDP"],
                        default="single")
    return parser.parse_args()

def main():
    args = parse_args()
    data = DataModule(path=args.path, tokenizer=args.model_name, batch_size=8)
    data.prepare_data()
    data.setup()
    model = DistilBert(
        model_config={
            "model_name": args.model_name,
            "num_classes": args.num_classes
            },
        training_config={
            "optmizer": args.optimizer,
            "distributed_strategy": args.distributed_strategy
        }
            )
    
    checkpoint_callback = ModelCheckpoint(
            dirpath="./models", monitor="val_loss", mode="min"
        )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    strategy = args.distributed_strategy
    if strategy == "FSDP":
        distributed_strategy = FSDPStrategy(
            sharding_strategy= "FULL_SHARD"
        )
    elif strategy == "DDP":
        distributed_strategy = "ddp"
    elif strategy == "single":
        distributed_strategy = "auto"

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=(-1 if torch.cuda.is_available() else 0),
        max_epochs=args.epochs,
        strategy=distributed_strategy,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    train_data_loader = data.train_dataloader()
    val_data_loader = data.val_dataloader()

    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

if __name__ == "__main__":
    main()