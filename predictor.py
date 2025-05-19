import torch
from model import DistilBert
from dataloader import DataModule

class Predictor():
    def __init__(self, model_path: str, data_module: DataModule):
        self.model_path = model_path
        self.model=DistilBert.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.model.freeze()
        self.data_module = data_module
        self.labels = {label : idx for idx, label in enumerate(sorted(set(self.data_module.train_data["label"])))}
        self.soft_max = torch.nn.Softmax(dim=0)

    def inference(self, text: str):
        text = {"text": text}
        tokenized_text = self.data_module.text_tokenize(text)
        logits = self.model(
            torch.tensor(tokenized_text["input_ids"]),
            torch.tensor(tokenized_text["attention_mask"])
        )
        scores = self.soft_max(logits[0]).to_list()
        for label, score in zip(self.labels, scores):
            predictions = {label:score}

        return max(predictions, predictions.get)
