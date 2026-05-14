# Plantrix

Plantrix is a plant-leaf image classification backend built with FastAPI, PyTorch, MySQL, AWS S3, and Twilio. It supports user signup/login with OTP verification, stores prediction history in a database, uploads submitted images to S3, and serves predictions from a trained EfficientNet model.

Live website: https://plantcare-ai-base.netlify.app/

## Deployment note

The model is hosted on AWS S3 and served through an EC2 instance. The instance is currently turned off to avoid AWS charges on a free account, so live prediction is not available at the moment. To restore prediction, the EC2 instance must be started and the backend on the website must be updated with the instance’s current public IP address.

## What’s in this project

- `app.py` - FastAPI backend and API routes
- `Model_Train.ipynb` - notebook used to train the image classification model
- `requirement.txt` - Python dependencies

## Core capabilities

- OTP-based signup and password reset flow via Twilio SMS
- Email/password login with bcrypt password hashing
- Profile lookup, update, and user deletion
- Leaf image prediction using a PyTorch model
- Image upload to AWS S3
- Prediction history stored in MySQL

## Model overview

The training notebook uses `torchvision.datasets.ImageFolder` on a `Dataset` directory and trains an EfficientNet-B0 classifier. The notebook shows a 70/20/10 train/validation/test split, batch size 64, and a two-phase training schedule:

- Phase 1: train classifier only for 5 epochs
- Phase 2: fine-tune the model for 10 epochs

Observed classes in the notebook:

- Bacterial Blight
- Cercospora
- Healthy Coffee Leaf
- Healthy Sugarcane Leaf
- Mosaic
- RedRot
- Rust Coffee Leaf
- Rust Sugarcane Leaf
- Yellow

At runtime, `app.py` loads `Model2.pth` from the project root. If it is missing locally, the app downloads it from S3 using the configured bucket and key.

## API endpoints

- `GET /` - health check
- `POST /send-otp` - generate and send an OTP to a phone number
- `POST /verify-otp` - verify an OTP
- `POST /signup` - create a new user after OTP verification
- `POST /login` - authenticate a user and return the stored device ID
- `POST /reset-password` - reset a password after OTP verification
- `GET /profile/{email}` - fetch a user profile
- `PUT /edit-profile` - update name, phone, and gender
- `DELETE /delete-user/{email}` - delete a user
- `POST /predict` - classify an uploaded image and store the result
- `GET /predictions/{device_id}` - list prediction history for a device
- `GET /latest/{device_id}` - fetch the most recent prediction
- `DELETE /prediction/{prediction_id}` - delete a prediction record

## Requirements

- Python 3.10 or newer
- MySQL database
- AWS account with S3 access
- Twilio account and verified SMS sender

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirement.txt
```

3. Create a `.env` file in the project root with the required values.

## Environment Variables

Create a `.env` file in the root directory and add the following variables:

```env
# Database
DB_HOST=your_mysql_host
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_mysql_database

# Twilio
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE=your_twilio_phone_number

# AWS
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_s3_bucket_name

# ML Model
MODEL_KEY=path/to/Model2.pth/in/s3
```

## Database expectations

The backend expects at least these tables:

- `users` with columns: `name`, `email`, `password`, `phone`, `gender`, `device_id`
- `predictions` with columns: `prediction_id`, `device_id`, `image_url`, `prediction`, `confidence`, `created_at`

## Running the backend

Start the API with Uvicorn:

```bash
uvicorn app:app --reload
```

By default, the service runs at `http://127.0.0.1:8000`.

## Uploading an image for prediction

The prediction endpoint expects multipart form data:

- `device_id` as a form field
- `file` as the uploaded image

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "device_id=device-123" \
  -F "file=@sample.jpg"
```

## Training notebook notes

`Model_Train.ipynb` contains the model training workflow used to produce the deployed classifier. It uses image augmentation, class visualization, train/validation/test loaders, and a two-stage optimization approach with Adam and mixed precision.

If you retrain the model, make sure the exported checkpoint includes:

- `model_state_dict`
- `class_names`

so `app.py` can load the model and map predictions to labels correctly.

## Notes

- CORS is currently open to all origins in the backend.
- OTPs are stored in memory, so they reset when the app restarts.
- The app assumes the database schema already exists.
