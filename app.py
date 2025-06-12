from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import Predictor
from dataloader import DataModule

app = FastAPI(title="Vietnamese Text Classification API")

# Create a model for the prediction request
class PredictionRequest(BaseModel):
    text: str

# Initialize the predictor when the app starts
@app.on_event("startup")
async def startup_event():
    app.state.predictor = Predictor(
        model_path="./production_models/last_model.onnx",
        data_module=DataModule(
            path="hiuman/vietnamese_classification",
            tokenizer="distilbert-base-uncased",
            batch_size=8
        )
    )

@app.get("/")
def home():
    """Root endpoint"""
    return {"status": "ok", "message": "Vietnamese Text Classification API"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict endpoint"""
    try:
        result = app.state.predictor.inference(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)