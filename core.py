import os
import torch
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer
import pickle

MODEL_NAME = "Xenova/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization constants (required for CLIP models)
# These are the per-channel mean and std calculated from ImageNet dataset
IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)  # RGB means
IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)   # RGB stds

def load_model_and_processor():
    """Load the quantized ONNX models for vision and text"""
    
    # Download the quantized ONNX models
    vision_model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="onnx/vision_model_quantized.onnx"
    )
    
    text_model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="onnx/text_model_quantized.onnx"
    )
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create ONNX runtime sessions
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
    
    vision_session = ort.InferenceSession(vision_model_path, providers=providers)
    text_session = ort.InferenceSession(text_model_path, providers=providers)
    
    return {
        'vision_session': vision_session,
        'text_session': text_session,
        'tokenizer': tokenizer
    }, None  # Return None for processor to maintain compatibility

def check_required_files(filenames):
    missing = [f for f in filenames if not os.path.isfile(f)]
    return missing

def load_labels():
    with open("data/labels_EN.txt", encoding="utf-8") as f:
        labels_en_raw = f.read().splitlines()
    with open("data/labels_DE.txt", encoding="utf-8") as f:
        labels_de = f.read().splitlines()
    with open("data/answers_DE.txt", encoding="utf-8") as f:
        answers_de = f.read().splitlines()
    labels_en = [f"An image containing {x}" for x in labels_en_raw]
    return labels_en_raw, labels_en, labels_de, answers_de

def embed_text(texts, model, processor, chunk=128):
    """Generate text embeddings using ONNX text model"""
    text_session = model['text_session']
    tokenizer = model['tokenizer']
    
    parts = [texts[i:i+chunk] for i in range(0, len(texts), chunk)]
    out = []
    
    for text_chunk in parts:
        # Tokenize the text - only get input_ids
        inputs = tokenizer(
            text_chunk,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="np"
        )
        
        # Run inference with only input_ids
        text_features = text_session.run(
            ["text_embeds"],
            {"input_ids": inputs["input_ids"]}
        )[0]
        
        # Normalize
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        out.append(torch.from_numpy(text_features))
    
    feats = torch.cat(out)
    # Final normalization
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def classify(image, model, processor, text_feats):
    """Classify image using ONNX vision model"""
    vision_session = model['vision_session']
    
    # Preprocess image
    image = ImageOps.exif_transpose(image).convert("RGB")
    # image = image.resize((224, 224))
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization: (pixel - mean) / std
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    
    # Reshape to (1, 3, 224, 224) - NCHW format
    image_array = image_array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    
    # Run inference
    image_features = vision_session.run(
        ["image_embeds"],
        {"pixel_values": image_array}
    )[0]
    
    # Normalize
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    image_features = torch.from_numpy(image_features)
    
    # Compute similarity
    similarity_scores = (image_features @ text_feats.T)[0]
    return similarity_scores