"""
Image Utilities for IVA Assignment 2 Lab.
Handles robust image loading, format conversion, resizing, and sample image management.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images")

def load_image_from_upload(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Image.Image], Optional[str]]:
    """
    Loads an image from a Streamlit UploadedFile object.
    
    Returns:
        tuple: (rgb_array, bgr_array, pil_image, error_message)
    """
    if uploaded_file is None:
        return None, None, None, "No file provided."
    
    try:
        # Read bytes
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0) # Reset pointer
        
        # Decode using OpenCV
        np_arr = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if bgr is None:
            # Fallback to PIL
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
                rgb = np.array(pil_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return rgb, bgr, pil_img, None
            except Exception as e:
                return None, None, None, f"Could not decode image: {str(e)}"
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return rgb, bgr, pil_img, None
        
    except Exception as e:
        return None, None, None, f"Error processing uploaded image: {str(e)}"

def load_image_from_path(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Image.Image], Optional[str]]:
    """
    Loads an image from a local file path.
    """
    if not os.path.exists(file_path):
        return None, None, None, f"File not found: {file_path}"
    
    try:
        bgr = cv2.imread(file_path)
        if bgr is None:
            return None, None, None, f"Failed to read image at {file_path}"
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return rgb, bgr, pil_img, None
    except Exception as e:
        return None, None, None, f"Error reading image path: {str(e)}"

def resize_for_display(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resizes image maintaining aspect ratio so the longest side does not exceed max_dim.
    """
    if image is None:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def get_sample_images_catalog() -> Dict[str, Dict[str, Any]]:
    """
    Returns pre-configured sample datasets for quick testing in the UI.
    """
    return {
        "Template Matching": {
            "Circuit Board & Microchip": {
                "main": os.path.join(SAMPLE_DIR, "scene_circuit.jpg"),
                "template": os.path.join(SAMPLE_DIR, "template_chip.jpg"),
                "desc": "Detecting microchip 'IVA' on PCB grid scene"
            },
            "Office Station & Target Badge": {
                "main": os.path.join(SAMPLE_DIR, "scene_office.jpg"),
                "template": os.path.join(SAMPLE_DIR, "template_target.jpg"),
                "desc": "Finding target badge on lab station board"
            }
        },
        "Viola-Jones Faces": {
            "Single Subject (Person A)": {
                "path": os.path.join(SAMPLE_DIR, "face_person_a_1.jpg"),
                "desc": "Single front-facing portrait"
            },
            "Single Subject with Glasses (Person B)": {
                "path": os.path.join(SAMPLE_DIR, "face_person_b_1.jpg"),
                "desc": "Portrait with spectacles & facial features"
            },
            "Laboratory Group Photo (3 Persons)": {
                "path": os.path.join(SAMPLE_DIR, "group_faces.jpg"),
                "desc": "Multiple faces in a group composition"
            }
        },
        "DeepFace / FaceNet Pairs": {
            "Same Person Pair (Person A - Smile vs Serious)": {
                "img1": os.path.join(SAMPLE_DIR, "face_person_a_1.jpg"),
                "img2": os.path.join(SAMPLE_DIR, "face_person_a_2.jpg"),
                "same": True,
                "desc": "Positive pair: Same identity under different facial expression"
            },
            "Different Person Pair (Person A vs Person B)": {
                "img1": os.path.join(SAMPLE_DIR, "face_person_a_1.jpg"),
                "img2": os.path.join(SAMPLE_DIR, "face_person_b_1.jpg"),
                "same": False,
                "desc": "Negative pair: Two distinctly different individuals"
            },
            "Different Person Pair (Person A vs Person C)": {
                "img1": os.path.join(SAMPLE_DIR, "face_person_a_1.jpg"),
                "img2": os.path.join(SAMPLE_DIR, "face_person_c_1.jpg"),
                "same": False,
                "desc": "Negative pair: Different skin tone and facial structure"
            }
        }
    }
