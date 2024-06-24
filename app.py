# Import necessary libraries
import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, ViTForImageClassification, BertTokenizer, BertModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import os
import base64
import numpy as np
import random
from io import BytesIO  # Import BytesIO
# Set page configuration
st.set_page_config(
    page_title="Fashion Design Recommendation",
    page_icon=":dress:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to encode local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to the local image
background_image_path = r'D:/AI Virtual anaylst/myenvA/Scripts/image.jpg'  # Ensure this is the correct path
background_image_base64 = get_base64_image(background_image_path)

# Load pre-trained models
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return tokenizer, bert_model, feature_extractor, model

tokenizer, bert_model, feature_extractor, model = load_models()

# Load image features from the pickle file
@st.cache_data
def load_image_features():
    file_path = r'D:/AI Virtual anaylst/myenvA/Scripts/image_features.pkl'  # Ensure this is the correct path
    if not os.path.exists(file_path):
        st.error(f"Pickle file not found: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        image_features_df = pickle.load(f)
    return image_features_df

image_features_df = load_image_features()
if image_features_df is None:
    st.stop()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize images to a smaller resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

# Streamlit app styling
st.markdown(
    f"""
    <style>
    .main {{
        background-image: url("data:image/jpeg;base64,{background_image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.8; /* Make background more transparent */
        padding: 20px;
        border-radius: 10px;
    }}
    .title {{
        font-family: 'Arial Black', sans-serif;
        color: black;
        text-shadow: 0 0 5px #ADD8E6, 0 0 10px #ADD8E6, 0 0 15px #ADD8E6, 0 0 20px #ADD8E6, 0 0 25px #ADD8E6;
        font-style: italic;
    }}
    .header {{
        font-family: 'Arial', sans-serif;
        color: black;
        text-shadow: 0 0 5px #ADD8E6, 0 0 10px #ADD8E6, 0 0 15px #ADD8E6, 0 0 20px #ADD8E6, 0 0 25px #ADD8E6;
        font-style: italic;
        text-align: center;
    }}
    .description {{
        font-family: 'Arial', sans-serif;
        color: #F6F5EE; /* Neon green color */
        text-shadow: 0 0 5px #F6F5EE, 0 0 10px #F6F5EE, 0 0 15px #F6F5EE, 0 0 20px #F6F5EE, 0 0 25px #F6F5EE;
        font-style: italic;
    }}
    .image-container {{
        display: flex;
        justify-content: center;
        gap: 10px;
    }}
    .image-item {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #fff;
        padding: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .image-item img {{
        width: 150px;
        height: 150px;
        object-fit: cover;
    }}
    .caption {{
        font-family: 'Arial', sans-serif;
        color: #333;
        text-align: center;
        margin-top: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Fashion Design Recommendation</h1>', unsafe_allow_html=True)

st.markdown('<h3 class="header">Enter a description of the fashion item to see recommended designs:</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    description = st.text_input("", placeholder="Type your description here...", label_visibility="collapsed")

with col2:
    search_button = st.button("Search")

def display_local_images(num_images=5):
    images_folder = r'D:/AI Virtual anaylst/myenvA/Scripts/images'  # Ensure this is the correct path
    image_files = os.listdir(images_folder)
    selected_images = random.sample(image_files, num_images)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    for image_name in selected_images:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            # Open image and resize
            img = Image.open(image_path)
            img = img.resize((150, 150))  # Resize image to a smaller resolution
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'''
                <div class="image-item">
                    <img src="data:image/jpeg;base64,{img_base64}" alt="{image_name}">
                    <div class="caption">{image_name}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.warning(f"Image not found: {image_name}")
    st.markdown('</div>', unsafe_allow_html=True)

if search_button:
    if description:
        # Process the description to get related images (for demonstration, use local images)
        display_local_images()
    else:
        st.info("Please enter a description to see recommended designs.")
        display_local_images()
