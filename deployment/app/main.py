from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
from src.predict import Translator

app = FastAPI(title="Neural Machine Translation API")

# Biến toàn cục để giữ instance của mô hình (tránh load lại nhiều lần)
translator = None

class TranslationRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global translator
    # Giả định bạn đã lưu từ điển en_vocab và vi_vocab bằng pickle sau khi EDA hoặc Train
    with open("weights/en_vocab.pkl", "rb") as f:
        en_vocab = pickle.load(f)
    with open("weights/vi_vocab.pkl", "rb") as f:
        vi_vocab = pickle.load(f)
        
    translator = Translator(
        model_path="weights/model_epoch_10.pth",
        en_vocab=en_vocab,
        vi_vocab=vi_vocab,
        device="cpu" # FastAPI thường chạy trên CPU để tiết kiệm chi phí
    )

@app.post("/translate")
def translate(request: TranslationRequest):
    output_text = translator.translate(request.text)
    return {"original": request.text, "translated": output_text}