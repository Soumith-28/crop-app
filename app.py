import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------
# Page Settings
# -------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and the model will predict the disease.")

# -------------------------
# Load Class Names
# -------------------------
CLASS_NAMES = [
    'Black_Gram_Anthracnose', 'Black_Gram_Healthy', 'Black_Gram_Leaf_Crinckle', 'Black_Gram_Powdery_Mildew', 'Black_Gram_Yellow_Mosaic', 'Cotton_Bacterial_Blight', 'Cotton_Curl_Virus', 'Cotton_Healthy_Leaf', 'Cotton_Hopper_Jassids', 'Cotton_Leaf_Redding', 'Rice___bacterial_blight', 'Rice___blast', 'Rice___brown_spot', 'Rice___leaf_scald', 'Rice___tungro', 'Sugarcane_Healthy', 'Sugarcane_Mosaic', 'Sugarcane_Redrot', 'Sugarcane_Rust', 'Sugarcane_Yellow_Mosaic', 'Wheat_Black_Point', 'Wheat_Blast', 'Wheat_Foot_Rot', 'Wheat_Healthy', 'Wheat_Leaf_Blight'   # CHANGE THIS to your classes
]

NUM_CLASSES = len(CLASS_NAMES)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("plant_disease_resnet50.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# File Upload UI
# -------------------------
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("predict"):
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        pred_class = CLASS_NAMES[predicted.item()]
        st.success(f"🌱 Predicted Disease: **{pred_class}**")
