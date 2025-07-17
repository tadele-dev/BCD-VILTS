from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoTokenizer
import io

app = FastAPI()

model = torch.jit.load("traced_model.pt")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict(image: UploadFile = File(...), text: TextInput = None):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    tokens = tokenizer(text.text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()

    return {"prediction": prediction}
