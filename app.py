import io
import os
import random
import torch
import boto3
import torch.nn as nn
import mysql.connector
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
from pydantic import BaseModel
from passlib.context import CryptContext
from torchvision import models, transforms
from PIL import Image
from twilio.rest import Client

app = FastAPI()

# =========================
# TWILIO CONFIG 🔥
# =========================
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PASSWORD HASHING
# =========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

# =========================
# OTP STORE
# =========================
otp_store = {}

# =========================
# DATABASE
# =========================
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
cursor = db.cursor(dictionary=True)

# =========================
# MODEL LOADING FROM S3 🚀
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOCAL_PATH = os.path.join(BASE_DIR, "Model2.pth")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Download model if not exists
if not os.path.exists(MODEL_LOCAL_PATH):
    print("Downloading model from S3...")
    s3.download_file(
        os.getenv("AWS_BUCKET_NAME"),
        os.getenv("MODEL_KEY"),
        MODEL_LOCAL_PATH
    )
    print("Model downloaded!")

# Load model
checkpoint = torch.load(MODEL_LOCAL_PATH, map_location="cpu")
class_names = checkpoint["class_names"]

model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
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

    return class_names[predicted.item()], confidence.item() * 100

# =========================
# IMAGE DIR
# =========================
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# =========================
# MODELS
# =========================
class SignupModel(BaseModel):
    name: str
    email: str
    password: str
    phone: str
    gender: str
    device_id: str
    otp: str

class OTPRequest(BaseModel):
    phone: str

class OTPVerify(BaseModel):
    phone: str
    otp: str

class LoginModel(BaseModel):
    email: str
    password: str

class ResetPassword(BaseModel):
    phone: str
    new_password: str
    otp: str

class EditProfile(BaseModel):
    email: str
    name: str
    phone: str
    gender: str

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

# =========================
# SEND OTP (REAL SMS)
# =========================
@app.post("/send-otp")
def send_otp(data: OTPRequest):
    otp = str(random.randint(1000, 9999))
    otp_store[data.phone] = otp

    try:
        twilio_client.messages.create(
            body=f"Your OTP is {otp}",
            from_=TWILIO_PHONE,
            to=data.phone
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "OTP sent successfully"}

# =========================
# VERIFY OTP
# =========================
@app.post("/verify-otp")
def verify_otp(data: OTPVerify):
    if otp_store.get(data.phone) != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    return {"message": "OTP verified"}

# =========================
# SIGNUP (WITH OTP)
# =========================
@app.post("/signup")
def signup(user: SignupModel):
    if otp_store.get(user.phone) != user.otp:
        raise HTTPException(status_code=400, detail="OTP not verified")

    hashed = hash_password(user.password)

    try:
        cursor.execute(
            "INSERT INTO users (name,email,password,phone,gender,device_id) VALUES (%s,%s,%s,%s,%s,%s)",
            (user.name, user.email, hashed, user.phone, user.gender, user.device_id)
        )
        db.commit()
    except:
        raise HTTPException(status_code=400, detail="User already exists")

    del otp_store[user.phone]

    return {"message": "User registered successfully"}

# =========================
# LOGIN
# =========================
@app.post("/login")
def login(user: LoginModel):
    cursor.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid password")

    return {"message": "Login successful", "device_id": db_user["device_id"]}

# =========================
# RESET PASSWORD (WITH OTP)
# =========================
@app.post("/reset-password")
def reset_password(data: ResetPassword):
    if otp_store.get(data.phone) != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed = hash_password(data.new_password)

    cursor.execute(
        "UPDATE users SET password=%s WHERE phone=%s",
        (hashed, data.phone)
    )
    db.commit()

    del otp_store[data.phone]

    return {"message": "Password updated"}

# =========================
# PROFILE
# =========================
@app.get("/profile/{email}")
def profile(email: str):
    cursor.execute(
        "SELECT name,email,phone,gender,device_id FROM users WHERE email=%s",
        (email,)
    )
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user

# =========================
# EDIT PROFILE
# =========================
@app.put("/edit-profile")
def edit_profile(data: EditProfile):
    cursor.execute(
        "UPDATE users SET name=%s, phone=%s, gender=%s WHERE email=%s",
        (data.name, data.phone, data.gender, data.email)
    )
    db.commit()

    return {"message": "Profile updated"}

# =========================
# DELETE USER
# =========================
@app.delete("/delete-user/{email}")
def delete_user(email: str):
    cursor.execute("DELETE FROM users WHERE email=%s", (email,))
    db.commit()
    return {"message": "User deleted"}

# =========================
# PREDICTION
# =========================
@app.post("/predict")
async def predict(device_id: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()

    prediction, confidence = predict_image(image_bytes)

    filename = f"{device_id}_{file.filename}"

    # ✅ FIX: Handle missing content type
    content_type = file.content_type or "image/jpeg"

    # Upload to S3
    s3.put_object(
        Bucket=os.getenv("AWS_BUCKET_NAME"),
        Key=f"images/{filename}",
        Body=image_bytes,
        ContentType=content_type
    )

    # Generate URL
    image_url = f"https://{os.getenv('AWS_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/images/{filename}"

    # Save in DB
    cursor.execute(
        "INSERT INTO predictions (device_id,image_url,prediction,confidence) VALUES (%s,%s,%s,%s)",
        (device_id, image_url, prediction, confidence)
    )
    db.commit()

    return {
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "image_url": image_url
    }

@app.get("/predictions/{device_id}")
def get_predictions(device_id: str):
    cursor.execute(
        "SELECT * FROM predictions WHERE device_id=%s ORDER BY created_at DESC",
        (device_id,)
    )
    return {"predictions": cursor.fetchall()}

@app.get("/latest/{device_id}")
def latest(device_id: str):
    cursor.execute(
        "SELECT * FROM predictions WHERE device_id=%s ORDER BY created_at DESC LIMIT 1",
        (device_id,)
    )
    return cursor.fetchone()

@app.delete("/prediction/{prediction_id}")
def delete_prediction(prediction_id: int):
    cursor.execute("DELETE FROM predictions WHERE prediction_id=%s", (prediction_id,))
    db.commit()
    return {"message": "Deleted"}