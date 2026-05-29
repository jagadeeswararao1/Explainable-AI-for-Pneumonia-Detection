# XAI Dashboard (PyTorch EfficientNet-B0)

This is a premium, modern dashboard built with React (Vite) and Django. It provides an interface to upload images, generate predictions from your PyTorch model, and extract Explainable AI maps using Grad-CAM++ and LIME.

## Quick Start Setup

### 1. Model Configuration

Before running the server, you must provide your trained model:

1. Look in `backend/api/ml_utils.py`.
2. Update `MODEL_PATH = 'model.pth'` to the exact location of your saved `.pth` or `.pt` PyTorch model file.
3. Update the `CLASSES` list to correctly map to the index classes (e.g. `CLASSES = ['cats', 'dogs', 'birds']`). This must be identical to `train_dataset.classes` from your notebook.

### 2. Backend (Django) Setup

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Virtual Environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Django Development Server:
   ```bash
   python manage.py runserver
   ```
   *The server should now be running on `http://127.0.0.1:8000/`*

### 3. Frontend (React) Setup

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install NodeJS dependencies:
   ```bash
   npm install
   ```
3. Start the Vite Development Server:
   ```bash
   npm run dev
   ```
   *Open the URL shown in your terminal (usually `http://localhost:5173`) in your browser.*

## Tech Stack Overview

- **Backend**: Django REST Framework
- **Deep Learning**: PyTorch (`torch`, `torchvision`, `grad-cam`, `lime`)
- **Frontend**: React (Vite framework)
- **UI Design**: Modern Vanilla CSS with dark mode dynamics, animated gradients, and glassmorphism cards.

## Troubleshooting

- **CORS Errors**: If your frontend cannot communicate with the backend, ensure `django-cors-headers` is installed and `CORS_ALLOW_ALL_ORIGINS = True` is set in `backend/core/settings.py`.
- **Model Loading Errors**: If the weights fail to load, ensure the EfficientNet-B0 architecture defined in `ml_utils.py` exactly matches the model saved in your notebook.
- **XAI Memory Limits**: LIME is computationally heavy. If you run out of memory, decrease the `num_samples` parameter in `ml_utils.py` under the `generate_lime()` function.
