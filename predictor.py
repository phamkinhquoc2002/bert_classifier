import torch
import onnxruntime as ort
from dataloader import DataModule
import numpy as np

class Predictor():
    def __init__(self, model_path: str, data_module: DataModule):
        self.ort_session = ort.InferenceSession(model_path)
        self.data_module = data_module
        self.data_module.prepare_data()
        self.data_module.setup(stage="fit")
        self.labels ={v: k for k, v in self.data_module.classes_map.items()}
        self.soft_max = torch.nn.Softmax(dim=1)

    def inference(self, text: str):
        """Make predictions using the ONNX model."""
        text = {"text": [text]}  # Wrap text in a list for batch processing
        tokenized_text = self.data_module.text_tokenize(text)
        
        # Run inference
        ort_inputs = {
            "input_ids": np.array(tokenized_text["input_ids"]),
            "attention_mask": np.array(tokenized_text["attention_mask"]),
        }
        
        # Run inference
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # Convert output to torch tensor and apply softmax
        scores = self.soft_max(torch.tensor(ort_outs[0]))
        
        # Get predicted class and probabilities
        predicted_idx = torch.argmax(scores, dim=1).item()
        predicted_label = self.labels[predicted_idx]
        probabilities = scores[0].tolist()
        
        return {
            "predicted_label": predicted_label,
            "probabilities": {
                self.labels[i]: prob for i, prob in enumerate(probabilities)
            }
        }