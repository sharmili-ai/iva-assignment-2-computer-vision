# IVA Assignment 2 – Computer Vision & Face Analysis Lab

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facial%20Analytics-brightgreen.svg)](https://github.com/serengil/deepface)

> **Interactive Demonstration of Template Matching, Viola–Jones, DeepFace, and FaceNet**  
> Developed for the **Image and Video Analytics (IVA)** Academic Course.

---

## 🌐 Live Demo

[Open the IVA Assignment 2 Web Application](YOUR_STREAMLIT_URL)

---

## 📌 Project Overview

This web application is an educational, interactive computer vision laboratory designed to demonstrate and compare foundational classical computer vision algorithms against state-of-the-art deep learning architectures. It enables students and faculty members to adjust hyperparameters in real time, visualize intermediate representations (such as correlation response heatmaps, integral image mappings, and embedding profiles), and observe how algorithmic evolution has transformed visual analytics.

### 🎓 Academic Metadata
- **Student:** Sharmili
- **Course:** Image and Video Analytics (IVA)
- **Assignment:** Lab Assignment 2 – Computer Vision & Face Analysis

---

## 🚀 Key Features

1. **Interactive Template Matching Module:**
   - Evaluates all 6 standard OpenCV matching methods (`TM_CCOEFF_NORMED`, `TM_CCOEFF`, `TM_CCORR_NORMED`, `TM_CCORR`, `TM_SQDIFF_NORMED`, `TM_SQDIFF`).
   - Dynamic 3-panel visualization showing detected match, query template, and correlation response heatmap $R(x,y)$ with peak marker.
   - Non-Maximum Suppression (NMS) multi-match threshold slider for discovering multiple instances.

2. **Interactive Viola–Jones Face Detection Module:**
   - Real-time Haar Cascade frontal face and eye detection using OpenCV.
   - Real-time parameter tuning: `scaleFactor` (pyramid scale), `minNeighbors` (false positive rejection), and `minSize`.
   - Visual breakdown of Grayscale conversions, Log-normalized **Integral Image ($II(x,y)$)** representation, and segmented individual face crops.

3. **Interactive DeepFace Facial Analytics Module:**
   - **Mode A (Single Image Analysis):** Automatic face localization with demographic and affective estimations (Age, Gender with confidence %, Emotion probability distribution, and Race).
   - **Mode B (Pairwise Verification):** Compares two facial images and returns an identity match verdict, computed distance, and decision threshold.
   - Graceful offline fallback with clear heuristic labels if external network weights are unavailable.

4. **Interactive FaceNet Metric Learning Module:**
   - Generates compact 128-d or 512-d $L_2$-normalized feature embeddings via Inception-ResNet.
   - Computes **Euclidean Distance ($\|a-b\|_2$)** and **Cosine Similarity ($\cos\theta$)** between identity pairs.
   - Interactive Triplet Loss visualizer explaining Anchor, Positive, Negative, and Margin $\alpha$.
   - Interactive bar chart showing the first 30 dimensions and dimension-wise residual differences $\Delta_i$.

5. **Comprehensive Algorithmic Comparison Matrix:**
   - Multi-criteria radar chart comparing Speed, Pose Invariance, Expression Invariance, Identity Verification, and Low-Power CPU Execution.
   - Interactive selection wizard recommending which algorithm to use based on project requirements.

6. **Academic References & Viva Voce Exam Bank:**
   - Formal IEEE-style citations of seminal papers (Viola & Jones 2001, FaceNet 2015, DeepFace 2014).
   - 8 in-depth Viva Voce oral examination questions with model academic answers.

---

## 🎛️ Algorithms Covered

| Feature | Template Matching | Viola–Jones | DeepFace | FaceNet |
| :--- | :--- | :--- | :--- | :--- |
| **Main Purpose** | Template localization | Face detection | Facial analysis & verification | Face embeddings & recognition |
| **Approach** | Classical image matching | Classical CV (Haar Cascade) | Deep learning (CNN) | Deep metric learning (Triplet Loss) |
| **Input** | Main Image + Template | Image / Video stream | Face / Portrait image | Face image pairs |
| **Output** | Match location $(x,y)$ | Face bounding boxes | Facial predictions / verification | 128-d / 512-d embeddings & similarity |
| **Training** | No training required | Pre-trained cascade (AdaBoost) | Pre-trained deep models | Pre-trained Inception-ResNet |
| **Main Strength** | Simple sub-image localization | Fast CPU real-time detection | Rich multi-task facial analytics | Compact identity vector space |
| **Limitation** | Scale & rotation sensitive | Struggles with pose & occlusions | Computationally heavier (FLOPs) | Requires metric thresholding |

---

## 📁 Project Structure

```text
iva_assignment_2/
│
├── app.py                      # Main Streamlit Dashboard application
├── requirements.txt            # Python dependencies (Streamlit Cloud ready)
├── README.md                   # Comprehensive project documentation
│
├── modules/
│   ├── __init__.py
│   ├── template_matching.py    # OpenCV template matching with 6 methods
│   ├── viola_jones.py          # Haar cascade face detection & integral image
│   ├── deepface_module.py      # DeepFace analysis & verification with fallback
│   └── facenet_module.py       # FaceNet embeddings & Triplet Loss computation
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Image loading, preprocessing, and catalog
│   └── visualization.py        # Heatmap, embedding bar charts & radar visualizers
│
├── sample_images/              # Built-in offline academic sample images
│   ├── scene_circuit.jpg
│   ├── template_chip.jpg
│   ├── scene_office.jpg
│   ├── template_target.jpg
│   ├── face_person_a_1.jpg
│   ├── face_person_a_2.jpg
│   ├── face_person_b_1.jpg
│   ├── face_person_c_1.jpg
│   └── group_faces.jpg
│
└── static/
    └── style.css               # Modern academic glassmorphic lab styling
```

---

## 🛠️ Technology Stack

- **Python 3.10+ / 3.11 / 3.12**
- **Streamlit** – Web dashboard and interactive widgets
- **OpenCV (`opencv-python-headless`)** – Classical computer vision, correlation filtering, and Haar Cascades
- **NumPy & SciPy** – Array manipulation and distance metrics
- **Matplotlib** – Heatmaps, radar charts, and embedding profiles
- **Pillow (PIL)** – Image transformations and conversions
- **Pandas** – Dataframes for metrics and comparison tables
- **PyTorch & facenet-pytorch** – DeepFace / FaceNet convolutional inference

---

## 💻 Installation & Local Execution

### 1. Clone or Navigate to the Project Directory
```bash
cd iva_assignment_2
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This project is configured for 1-click deployment on **Streamlit Community Cloud**:

1. Push this directory to your **GitHub** repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository.
4. Set Main file path to `app.py` (or `iva_assignment_2/app.py`).
5. Click **Deploy**!

> **Cloud Note:** The repository utilizes `opencv-python-headless` in `requirements.txt` to ensure compatibility with headless Linux cloud instances without requiring external system GUI packages.

---

## 📚 Mathematical Formulations

### 1. Normalized Correlation Coefficient (Template Matching)
$$R(x,y) = \frac{\sum_{x',y'} \left(T'(x',y') \cdot I'(x+x', y+y')\right)}{\sqrt{\sum_{x',y'} T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x', y+y')^2}}$$

### 2. Integral Image (Viola–Jones)
$$II(x,y) = \sum_{x' \le x, y' \le y} I(x', y') \implies \text{Sum}(ABCD) = D - B - C + A$$

### 3. FaceNet Triplet Loss Objective
$$\mathcal{L}(A, P, N) = \max \left( 0, \|f(A) - f(P)\|^2_2 - \|f(A) - f(N)\|^2_2 + \alpha \right)$$

---

## 🔮 Future Enhancements

- [ ] Support for real-time live webcam video feeds using `streamlit-webrtc`.
- [ ] Integration of 3D facial landmark meshes (MediaPipe Face Mesh).
- [ ] Support for modern transformer-based vision architectures (e.g., Vision Transformers / ViT-Face).
- [ ] Vector database integration (e.g. ChromaDB / FAISS) for 1:N real-time gallery face search.

---

## 👩‍💻 Author & Academic Purpose

- **Author:** Sharmili
- **Course:** Image and Video Analytics (IVA)
- **Institution:** College Assignment Demonstration
- **License:** Open Academic Educational Resource
