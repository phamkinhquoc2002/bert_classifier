import os
import hydra
import torch

from dataloader import DataModule
from model import DistilBert

@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def onnx_converter(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    data = DataModule(path=cfg.training.data_path, tokenizer=cfg.model.tokenizer, batch_size=cfg.training.batch_size)
    data.prepare_data()
    data.setup(stage="fit")
    model = DistilBert.load_from_checkpoint("./models/last.ckpt")

    raw_input = next(iter(data.val_dataloader()))
    input_sample = {
        "input_ids": raw_input["input_ids"][0].unsqueeze(0),
        "attention_mask": raw_input["attention_mask"][0].unsqueeze(0)
    }
    torch.onnx.export(
        model,
        (
            input_sample["input_ids"],
            input_sample["attention_mask"],
        ),
        "./production_models/last_model.onnx",
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["outputs"],
    )

if __name__ == "__main__":
    onnx_converter()