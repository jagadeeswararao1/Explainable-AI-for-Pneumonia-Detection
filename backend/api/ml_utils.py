import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import io
import base64
import cv2

# Ensure you have installed: pip install grad-cam lime scikit-image
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAMPlusPlus = None

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    lime_image = None

# ---------------------------------------------------------
# CONFIGURATION (✅ FIXED)
# ---------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')

CLASSES = [
    'Atelectasis',
    'Covid-19',
    'Effusion',
    'Emphysema',
    'Infiltration',
    'Normal',
    'Pneumonia-Bacterial',
    'Pneumonia-Viral',
    'Tuberculosis'
]

NUM_CLASSES = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# MODEL LOADING (✅ STRICT + CORRECT)
# ---------------------------------------------------------
def load_model():
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model


model = load_model()


# ---------------------------------------------------------
# TRANSFORM (match training)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# IMAGE → BASE64
# ---------------------------------------------------------
def image_to_base64(img_array):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if img_array.dtype in [np.float32, np.float64]:
        img_array = (img_array * 255).astype(np.uint8)

    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')


# ---------------------------------------------------------
# GRADCAM (EfficientNet FIX)
# ---------------------------------------------------------
def generate_grad_cam(input_tensor, original_img_np):
    if GradCAMPlusPlus is None:
        return ""

    # Correct target layer for EfficientNet
    target_layers = [model.features[-1][0]]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        img_normalized = original_img_np.astype(np.float32) / 255.0
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

        return image_to_base64(visualization)


# ---------------------------------------------------------
# LIME
# ---------------------------------------------------------
def generate_lime(original_img_np):
    if lime_image is None:
        return ""

    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        batch = torch.stack([
            transform(Image.fromarray(img)) for img in images
        ]).to(device)

        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    explanation = explainer.explain_instance(
        original_img_np,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    img_boundary = mark_boundaries(temp / 255.0, mask)
    return image_to_base64((img_boundary * 255).astype(np.uint8))

def generate_saliency(input_tensor, original_img_np):

    input_tensor = input_tensor.clone()
    input_tensor.requires_grad_()

    # Forward
    output = model(input_tensor)
    score, _ = torch.max(output, 1)

    # Backward
    model.zero_grad()
    score.backward()

    # Get gradients
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    # Convert to grayscale
    saliency = np.max(saliency, axis=0)

    # Normalize properly
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)

    # Convert to 0–255
    saliency = (saliency * 255).astype(np.uint8)

    # Resize to original image size
    saliency = cv2.resize(saliency, (original_img_np.shape[1], original_img_np.shape[0]))

    # 🔥 APPLY HEATMAP (THIS FIXES BLACK IMAGE)
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_np, 0.6, saliency, 0.4, 0)
    return image_to_base64(overlay)



def generate_combined_xai(input_tensor, original_img_np):

    # -------- GradCAM --------
    target_layers = [model.features[-1][0]]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        gradcam = cam(input_tensor=input_tensor)[0]

    gradcam = (gradcam - gradcam.min()) / (gradcam.max() + 1e-8)
    gradcam = (gradcam * 255).astype(np.uint8)
    gradcam = cv2.resize(gradcam, (original_img_np.shape[1], original_img_np.shape[0]))
    gradcam = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)

    # -------- Saliency --------
    input_tensor_sal = input_tensor.clone()
    input_tensor_sal.requires_grad_()

    output = model(input_tensor_sal)
    score, _ = torch.max(output, 1)

    model.zero_grad()
    score.backward()

    saliency = input_tensor_sal.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)

    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    saliency = (saliency * 255).astype(np.uint8)

    saliency = cv2.resize(saliency, (original_img_np.shape[1], original_img_np.shape[0]))
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

    # -------- COMBINE --------
    combined = cv2.addWeighted(gradcam, 0.6, saliency, 0.4, 0)

    # -------- Overlay on original --------
    final_overlay = cv2.addWeighted(original_img_np, 0.5, combined, 0.5, 0)

    return image_to_base64(final_overlay)
# ---------------------------------------------------------
# MAIN INFERENCE
# ---------------------------------------------------------
def run_inference_and_xai(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_resized = image.resize((224, 224))
    original_img_np = np.array(image_resized)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---------------- Prediction ----------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted_idx = torch.max(probs, 1)

    class_idx = predicted_idx.item()
    conf_score = confidence.item()

    class_name = CLASSES[class_idx]

    # ---------------- XAI ----------------
    gradcam_b64 = generate_grad_cam(input_tensor, original_img_np)
    lime_b64 = generate_lime(original_img_np)
    saliency_b64 = generate_saliency(input_tensor.clone(), original_img_np)
    combined_b64 = generate_combined_xai(input_tensor, original_img_np)

    # ---------------- Original ----------------
    original_b64 = image_to_base64(original_img_np)



    return {
        "prediction": class_name,
        "confidence": round(conf_score * 100, 2),
        "original_image": f"data:image/jpeg;base64,{original_b64}",
        "gradcam_image": f"data:image/jpeg;base64,{gradcam_b64}",
        "lime_image": f"data:image/jpeg;base64,{lime_b64}",
        "saliency_map": f"data:image/jpeg;base64,{saliency_b64}",
        "combined_xai": f"data:image/jpeg;base64,{combined_b64}"
    }