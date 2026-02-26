import io
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "Model1.pth")

print("Loading model from:", MODEL_PATH)


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
class_names = checkpoint["class_names"]


model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_percentage = confidence.item() * 100

    return predicted_class, confidence_percentage



@app.get("/")
def home():
    return {"message": "Disease Detection API is Running!"}


@app.post("/predict")
async def predict(device_id: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction, confidence = predict_image(image_bytes)

    return {
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
