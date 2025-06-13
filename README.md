# 🧠 BERT Classifier with MLOps Integration

This repository contains a BERT-based text classification system built with Torch Lightning and integrated into an MLOps workflow. It is designed for modular, scalable, and reproducible training, evaluation, and deployment pipelines.

## 📦 Project Structure
```
├── Dockerfile
├── LICENSE
├── README.md
├── app.py
├── configs
│   ├── config.yaml
│   ├── model
│   └── training
├── dataloader.py
├── model.py
├── models
├── onx_converter.py
├── predictor.py
├── requirements.txt
├── samples_logger.py
├── train.py
```

## 🛠️ Guide

### 1. Clone the repo
```bash
git clone https://github.com/phamkinhquoc2002/bert_classifier.git
cd bert_classifier
```
### 2. Start the training by defining configurations in HYDRA.
To train the model:
```bash
python train.py
```
Metrics will be logged into wandb.
### 3. DVC Configuration.
```bash
dvc init
dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
dvc add models/last.ckpt
```
Metrics will be logged into wandb.
### 3. Build the Docker Image
```bash
docker build --build-arg GDRIVE_FOLDER_ID=$GDRIVE_FOLDER_ID  -t inference:latest .
```
### 4. Run the Docker Image for inference
Set the GDRIVE_FOLDER_ID in your terminal.

```bash
docker run -d --name inference -e DVC_REMOTE_URI=gdrive://${GDRIVE_FOLDER_ID} -v ./creds.json:/run/secrets/gdrive_creds.json:ro -p 8000:8000 inference:latest
```
