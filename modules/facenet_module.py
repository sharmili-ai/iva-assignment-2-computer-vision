"""
Module 4: FaceNet Embedding Generation & Metric Comparison for IVA Assignment 2.
Implements deep face embedding extraction (Inception-ResNet / FaceNet),
Euclidean distance & Cosine similarity computation, and interactive Triplet Loss education.
"""

import os
import cv2
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

# Check for PyTorch & facenet-pytorch or DeepFace Facenet
FACENET_AVAILABLE = False
_facenet_model = None

try:
    import torch
    from torchvision import transforms
    from facenet_pytorch import InceptionResnetV1, MTCNN
    FACENET_AVAILABLE = True
except Exception:
    FACENET_AVAILABLE = False


@st.cache_resource
def load_facenet_pytorch_model():
    """
    Loads pre-trained InceptionResnetV1 (FaceNet) model using PyTorch.
    Cached using st.cache_resource for optimal performance.
    """
    if not FACENET_AVAILABLE:
        return None, None
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained on VGGFace2 or Casia-Webface
        mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device, post_process=True)
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return mtcnn, model
    except Exception as e:
        print(f"FaceNet PyTorch loading notice: {e}")
        return None, None


def extract_facenet_embedding(
    image_rgb: np.ndarray,
    target_dim: int = 512
) -> Dict[str, Any]:
    """
    Extracts L2-normalized face embedding vector from an input RGB image.
    
    Args:
        image_rgb: Input RGB image
        target_dim: Target embedding vector dimension (128 or 512)
        
    Returns:
        Dictionary containing embedding vector, face crop, dimensions, and execution mode.
    """
    if image_rgb is None:
        return {"success": False, "error": "Image is missing."}

    # 1. Try PyTorch facenet-pytorch InceptionResnetV1
    mtcnn, model = load_facenet_pytorch_model()
    
    if mtcnn is not None and model is not None:
        try:
            device = next(model.parameters()).device
            pil_img = Image.fromarray(image_rgb)
            
            # Detect and align face
            face_tensor = mtcnn(pil_img)
            
            if face_tensor is not None:
                # Add batch dim
                face_tensor = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    raw_emb = model(face_tensor).cpu().numpy().flatten()
                    
                # L2 Normalize embedding
                norm = np.linalg.norm(raw_emb)
                emb = raw_emb / (norm + 1e-10)
                
                # Face crop for visual preview
                boxes, _ = mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    bx = [int(v) for v in boxes[0]]
                    x1, y1, x2, y2 = max(0, bx[0]), max(0, bx[1]), min(image_rgb.shape[1], bx[2]), min(image_rgb.shape[0], bx[3])
                    face_crop = image_rgb[y1:y2, x1:x2].copy()
                else:
                    face_crop = image_rgb
                    
                return {
                    "success": True,
                    "is_fallback": False,
                    "embedding": emb,
                    "dim": len(emb),
                    "face_crop": face_crop,
                    "model_name": "InceptionResnetV1 (Pretrained VGGFace2)",
                    "framework": "PyTorch / facenet-pytorch"
                }
        except Exception as e:
            print(f"PyTorch extraction error, falling back: {e}")

    # 2. Try DeepFace represent with 'Facenet' or 'Facenet512'
    try:
        from deepface import DeepFace
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        emb_obj = DeepFace.represent(img_bgr, model_name="Facenet", enforce_detection=False)
        if emb_obj and len(emb_obj) > 0:
            raw_emb = np.array(emb_obj[0]["embedding"])
            norm = np.linalg.norm(raw_emb)
            emb = raw_emb / (norm + 1e-10)
            return {
                "success": True,
                "is_fallback": False,
                "embedding": emb,
                "dim": len(emb),
                "face_crop": image_rgb,
                "model_name": "FaceNet (DeepFace Engine)",
                "framework": "DeepFace"
            }
    except Exception:
        pass

    # 3. Graceful Fallback Mode: Deterministic Feature Projection
    return _fallback_face_embedding(image_rgb, target_dim=128)


def _fallback_face_embedding(image_rgb: np.ndarray, target_dim: int = 128) -> Dict[str, Any]:
    """
    Generates deterministic spatial-frequency L2-normalized feature representation
    for offline academic demonstration when heavy deep learning weights cannot be retrieved.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    
    # 2D Discrete Cosine Transform (DCT) for spatial-frequency energy compaction
    dct = cv2.dct(resized)
    
    # Flatten top low-frequency coefficients
    feat = dct[:16, :16].flatten()[:target_dim]
    
    # Add subtle color histogram component
    h_hist = cv2.calcHist([image_rgb], [0], None, [16], [0, 256]).flatten()
    feat[:len(h_hist)] += h_hist * 0.1
    
    # L2-normalization
    norm = np.linalg.norm(feat)
    embedding = feat / (norm + 1e-10)
    
    # Extract Haar face crop if possible
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = image_rgb[y:y+h, x:x+w].copy()
    else:
        face_crop = image_rgb
        
    return {
        "success": True,
        "is_fallback": True,
        "fallback_reason": "Neural network weights offline; operating in educational simulation mode.",
        "embedding": embedding,
        "dim": len(embedding),
        "face_crop": face_crop,
        "model_name": "Frequency-Domain DCT Projection (Simulation)",
        "framework": "OpenCV / NumPy"
    }


def compare_face_embeddings(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    distance_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Computes Euclidean distance and Cosine similarity between two L2-normalized face embeddings.
    
    Args:
        emb_a: Embedding vector A
        emb_b: Embedding vector B
        distance_threshold: Decision threshold for Euclidean distance
        
    Returns:
        Dictionary with metrics and matching verdict.
    """
    a = np.asarray(emb_a).flatten()
    b = np.asarray(emb_b).flatten()
    
    # Ensure L2 normalization
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    
    # Euclidean Distance: ||a - b||_2
    euclidean_dist = float(np.linalg.norm(a - b))
    
    # Cosine Similarity: (a . b) / (||a|| ||b||) = dot(a, b) since normalized
    cosine_sim = float(np.dot(a, b))
    cosine_sim = max(-1.0, min(1.0, cosine_sim)) # Clamp
    
    # Cosine Distance: 1 - Cosine Similarity
    cosine_dist = float(1.0 - cosine_sim)
    
    # Verification Decision
    is_same_person = bool(euclidean_dist < distance_threshold)
    
    # Confidence estimation
    # For normalized vectors, Euclidean distance ranges [0, 2]
    similarity_percentage = max(0.0, min(100.0, (1.0 - (euclidean_dist / 2.0)) * 100.0))
    
    return {
        "euclidean_distance": round(euclidean_dist, 4),
        "cosine_similarity": round(cosine_sim, 4),
        "cosine_distance": round(cosine_dist, 4),
        "threshold": distance_threshold,
        "is_same_identity": is_same_person,
        "match_confidence": round(similarity_percentage, 1),
        "interpretation": (
            "✅ Strong Identity Match (Same Person)" if is_same_person
            else "❌ Different Identities (Distinct Persons)"
        )
    }
