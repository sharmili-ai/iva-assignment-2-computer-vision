"""
========================================================================================
IVA Assignment 2 – Computer Vision & Face Analysis Lab
Interactive Web Application for Demonstration of Template Matching, Viola-Jones,
DeepFace, and FaceNet.

Course: Image and Video Analytics (IVA)
Student: Sharmili
========================================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local module imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.template_matching import MATCH_METHODS, perform_template_matching
from modules.viola_jones import detect_faces_viola_jones, get_haar_cascade
from modules.deepface_module import is_deepface_available, analyze_single_face, verify_face_pair
from modules.facenet_module import (
    FACENET_AVAILABLE,
    extract_facenet_embedding,
    compare_face_embeddings
)
from utils.image_utils import (
    load_image_from_upload,
    load_image_from_path,
    resize_for_display,
    get_sample_images_catalog,
    SAMPLE_DIR
)
from utils.visualization import (
    plot_template_matching_results,
    draw_faces_bounding_boxes,
    plot_embedding_comparison,
    plot_radar_comparison
)

# --------------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & CUSTOM CSS
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="IVA Lab – CV & Face Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_file = os.path.join(BASE_DIR, "static", "style.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # Fallback styling if static file not loaded
    st.markdown("""
        <style>
        .main-header { background: #1e293b; color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; }
        .feature-card { background: #ffffff; padding: 1.2rem; border-radius: 10px; border: 1px solid #e2e8f0; }
        </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🔬</div>
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0f172a;">IVA LAB DASHBOARD</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748b; font-weight: 500;">Computer Vision & Face Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Radio
    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Template Matching",
            "👤 Viola–Jones",
            "🧠 DeepFace",
            "🔐 FaceNet",
            "📊 Comparison",
            "📚 References & Viva Prep"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Academic Meta Info Card
    st.markdown("""
        <div style="background: #f1f5f9; padding: 1rem; border-radius: 10px; border: 1px solid #e2e8f0; font-size: 0.82rem;">
            <div style="font-weight: 700; color: #1e293b; margin-bottom: 0.4rem;">🎓 Academic Metadata</div>
            <div style="color: #475569;"><strong>Student:</strong> Sharmili</div>
            <div style="color: #475569;"><strong>Course:</strong> Image and Video Analytics</div>
            <div style="color: #475569;"><strong>Assignment:</strong> Lab 2 – CV & Face Recognition</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Live System Status
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>System Status</div>", unsafe_allow_html=True)
    cv_ver = cv2.__version__
    deepface_status = "🟢 Active" if is_deepface_available() else "🟡 Fallback Mode"
    facenet_status = "🟢 PyTorch Active" if FACENET_AVAILABLE else "🟡 Fallback Mode"
    
    st.markdown(f"""
        <div style="font-size: 0.78rem; color: #64748b; line-height: 1.6; margin-top: 0.3rem;">
            <div>• OpenCV: <code>{cv_ver}</code></div>
            <div>• DeepFace: <code>{deepface_status}</code></div>
            <div>• FaceNet: <code>{facenet_status}</code></div>
            <div>• Cache: <code>st.cache_resource</code></div>
        </div>
    """, unsafe_allow_html=True)


# ======================================================================================
# 1. HOME PAGE
# ======================================================================================
if nav_selection == "🏠 Home":
    st.markdown("""
        <div class="main-header">
            <h1>IVA Assignment 2</h1>
            <p class="subtitle">Interactive Computer Vision and Face Analysis Laboratory</p>
            <div class="header-badges">
                <span class="header-badge">👤 Student: Sharmili</span>
                <span class="header-badge">📘 Course: Image and Video Analytics</span>
                <span class="header-badge">🔬 Topic: Template Matching, Viola-Jones, DeepFace & FaceNet</span>
                <span class="header-badge">🚀 Interactive Streamlit Dashboard</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Welcome to the IVA Lab!** Upload an image or select built-in sample datasets to explore and compare classical and deep learning computer vision algorithms interactively.")
    
    st.markdown("### 🎛️ Algorithms Covered in this Lab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div>
                    <span class="feature-card-icon">🔍</span>
                    <span class="tag-classical">Classical Computer Vision</span>
                    <h3>Template Matching</h3>
                    <p>Locate a specific target sub-image (template) inside a larger main scene using spatial correlation, normalized coefficients, and squared difference heatmaps.</p>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.82rem; color: #64748b;">
                    <strong>Core Concept:</strong> 2D Sliding Window Correlation &amp; $R(x,y)$ Matrix
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <div>
                    <span class="feature-card-icon">👤</span>
                    <span class="tag-classical">Classical Machine Learning</span>
                    <h3>Viola–Jones Algorithm</h3>
                    <p>The landmark 2001 real-time face detection framework utilizing rectangular Haar-like features, Integral Images for $O(1)$ evaluation, AdaBoost, and Cascaded Classifiers.</p>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.82rem; color: #64748b;">
                    <strong>Core Concept:</strong> Integral Images &amp; Attentional Reject Cascade
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <div>
                    <span class="feature-card-icon">🧠</span>
                    <span class="tag-deeplearning">Deep Convolutional Networks</span>
                    <h3>DeepFace Analysis</h3>
                    <p>Modern end-to-end deep learning framework for comprehensive facial analytics: 3D face alignment, age estimation, gender prediction, emotion classification, and pairwise verification.</p>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.82rem; color: #64748b;">
                    <strong>Core Concept:</strong> Deep Representations &amp; Multi-Attribute Estimation
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(r"""
            <div class="feature-card">
                <div>
                    <span class="feature-card-icon">🔐</span>
                    <span class="tag-deeplearning">Deep Metric Learning</span>
                    <h3>FaceNet Embeddings</h3>
                    <p>Maps face crops directly into a compact, L2-normalized 128-d or 512-d Euclidean embedding space trained via Triplet Loss where distance directly reflects identity similarity.</p>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.82rem; color: #64748b;">
                    <strong>Core Concept:</strong> Triplet Loss: $\|f(A)-f(P)\|^2 + \alpha < \|f(A)-f(N)\|^2$
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Computer Vision Evolution & Lab Pipeline")
    
    st.markdown("""
        <div class="workflow-container">
            <div class="workflow-step">
                <strong>1. Classical Matching</strong><br>
                <span style="font-size: 0.75rem; color: #64748b;">Pixel-level Template $R(x,y)$</span>
            </div>
            <div class="workflow-arrow">➔</div>
            <div class="workflow-step">
                <strong>2. Feature Cascade</strong><br>
                <span style="font-size: 0.75rem; color: #64748b;">Viola-Jones Haar Cascade</span>
            </div>
            <div class="workflow-arrow">➔</div>
            <div class="workflow-step">
                <strong>3. Deep Representations</strong><br>
                <span style="font-size: 0.75rem; color: #64748b;">DeepFace Multi-Attribute CNN</span>
            </div>
            <div class="workflow-arrow">➔</div>
            <div class="workflow-step">
                <strong>4. Metric Embedding</strong><br>
                <span style="font-size: 0.75rem; color: #64748b;">FaceNet Triplet Loss Vector Space</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ======================================================================================
# 2. MODULE 1 — TEMPLATE MATCHING
# ======================================================================================
elif nav_selection == "🔍 Template Matching":
    st.markdown("""
        <div class="main-header">
            <h1>🔍 Template Matching</h1>
            <p class="subtitle">Classical Sub-image Localization using 2D Intensity Correlation & Differences</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Academic Theory & Mathematical Concepts (Click to Expand)", expanded=True):
        st.markdown("""
            <div class="theory-box">
                <h4>What is Template Matching?</h4>
                Template matching is a classical computer vision technique used to locate the exact position of a smaller query template image $T(x', y')$ within a larger search image $I(x, y)$ by systematically sliding the template pixel-by-pixel across the search image and computing a similarity or difference response matrix $R(x, y)$.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Mathematical Formulations for OpenCV Matching Methods")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">1. Normalized Correlation Coefficient (TM_CCOEFF_NORMED)</div>
                    $$R(x,y) = \\frac{\\sum_{x',y'} (T'(x',y') \\cdot I'(x+x', y+y'))}{\\sqrt{\\sum_{x',y'} T'(x',y')^2 \\cdot \\sum_{x',y'} I'(x+x', y+y')^2}}$$
                    <small>Where $T'$ and $I'$ are zero-mean intensity matrices. Output range: $[-1.0, 1.0]$. Peak maximum indicates match.</small>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">2. Correlation Coefficient (TM_CCOEFF)</div>
                    $$R(x,y) = \\sum_{x',y'} \\left( T'(x',y') \\cdot I'(x+x', y+y') \\right)$$
                    <small>Measures relative covariance. Higher values denote greater pattern similarity.</small>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">3. Cross-Correlation (TM_CCORR)</div>
                    $$R(x,y) = \\sum_{x',y'} \\left( T(x',y') \\cdot I(x+x', y+y') \\right)$$
                    <small>Direct dot-product. Can be biased towards uniformly bright image patches.</small>
                </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">4. Normalized Squared Difference (TM_SQDIFF_NORMED)</div>
                    $$R(x,y) = \\frac{\\sum_{x',y'} (T(x',y') - I(x+x', y+y'))^2}{\\sqrt{\\sum_{x',y'} T(x',y')^2 \\cdot \\sum_{x',y'} I(x+x', y+y')^2}}$$
                    <small>Output range: $[0.0, 1.0]$. <strong>0.0</strong> represents a perfect identical match. Global minimum indicates match.</small>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">5. Squared Difference (TM_SQDIFF)</div>
                    $$R(x,y) = \\sum_{x',y'} \\left( T(x',y') - I(x+x', y+y') \\right)^2$$
                    <small>Sum of squared euclidean pixel residuals. Minimum indicates best match.</small>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">6. Normalized Cross-Correlation (TM_CCORR_NORMED)</div>
                    $$R(x,y) = \\frac{\\sum_{x',y'} (T(x',y') \\cdot I(x+x', y+y'))}{\\sqrt{\\sum_{x',y'} T(x',y')^2 \\cdot \\sum_{x',y'} I(x+x', y+y')^2}}$$
                    <small>Normalized dot product in $[0.0, 1.0]$ without zero-centering.</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="workflow-container">
                <div class="workflow-step">1. Input Image $I(x,y)$</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">2. Template $T(x',y')$</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">3. Slide Across $(x,y)$</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">4. Compute Matrix $R(x,y)$</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">5. Find $\\min$ or $\\max$ Loc</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">6. Draw Bounding Box</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧪 Interactive Template Matching Demonstration")

    # Data Input Selection
    input_source = st.radio("Choose Input Source:", ["Use Preset Academic Sample", "Upload Custom Images"], horizontal=True)

    main_img_rgb = None
    tmpl_img_rgb = None
    
    catalog = get_sample_images_catalog()["Template Matching"]

    if input_source == "Use Preset Academic Sample":
        sample_choice = st.selectbox("Select Sample Dataset:", list(catalog.keys()))
        selected_sample = catalog[sample_choice]
        
        main_img_rgb, _, _, err1 = load_image_from_path(selected_sample["main"])
        tmpl_img_rgb, _, _, err2 = load_image_from_path(selected_sample["template"])
        
        if err1 or err2:
            st.error(f"Error loading sample: {err1 or err2}")
        else:
            st.caption(f"ℹ️ **Dataset description:** {selected_sample['desc']}")
            
    else:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            u_main = st.file_uploader("Upload Main Search Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="tm_main")
            if u_main:
                main_img_rgb, _, _, err = load_image_from_upload(u_main)
                if err:
                    st.error(err)
        with u_col2:
            u_tmpl = st.file_uploader("Upload Target Template Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="tm_tmpl")
            if u_tmpl:
                tmpl_img_rgb, _, _, err = load_image_from_upload(u_tmpl)
                if err:
                    st.error(err)

    if main_img_rgb is not None and tmpl_img_rgb is not None:
        st.markdown("---")
        
        # Interactive Controls
        c_col1, c_col2, c_col3 = st.columns([1.5, 1.2, 1.3])
        
        with c_col1:
            selected_method = st.selectbox(
                "Select Matching Method:",
                list(MATCH_METHODS.keys()),
                index=1, # Default TM_CCOEFF_NORMED
                format_func=lambda k: f"{k} ({MATCH_METHODS[k]['name']})"
            )
            st.caption(f"📌 {MATCH_METHODS[selected_method]['summary']}")
            
        with c_col2:
            enable_multi = st.checkbox("Enable Multi-Match Thresholding", value=False)
            if enable_multi:
                is_sqdiff = "SQDIFF" in selected_method
                default_thresh = 0.15 if is_sqdiff else 0.80
                min_t = 0.0
                max_t = 1.0 if "NORMED" in selected_method else 1000000.0
                step_t = 0.02 if "NORMED" in selected_method else 1000.0
                
                match_threshold = st.slider(
                    "Multi-Match Threshold:",
                    min_value=float(min_t),
                    max_value=float(max_t),
                    value=float(default_thresh),
                    step=float(step_t),
                    help="Filters response peaks. Lower is better for SQDIFF; Higher is better for CCOEFF/CCORR."
                )
            else:
                match_threshold = None

        with c_col3:
            st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
            run_tm_btn = st.button("🚀 Run Template Matching", use_container_width=True, type="primary")

        # Execute Matching
        if run_tm_btn or True: # Run on parameter change for real-time responsiveness
            res = perform_template_matching(
                main_image_rgb=main_img_rgb,
                template_image_rgb=tmpl_img_rgb,
                method_key=selected_method,
                match_all_threshold=match_threshold
            )
            
            if not res["success"]:
                st.error(f"⚠️ {res['error']}")
            else:
                st.markdown("### 📊 Matching Results & Heatmap Analysis")
                
                # Metrics Bar
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Selected Method", res["method_key"])
                m2.metric("Best Match Location (x, y)", f"({res['best_location'][0]}, {res['best_location'][1]})")
                m3.metric("Template Dimensions (w × h)", f"{res['template_size'][0]} × {res['template_size'][1]} px")
                m4.metric("Peak Metric Score", f"{res['score']:.4f}")
                
                # Visual Plot
                fig = plot_template_matching_results(
                    main_rgb=main_img_rgb,
                    template_rgb=tmpl_img_rgb,
                    heatmap=res["heatmap"],
                    top_left=res["best_location"],
                    bottom_right=res["bottom_right"],
                    score=res["score"],
                    method_name=res["method_key"]
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                # Multi-match preview if enabled
                if enable_multi and res["multiple_matches"]:
                    st.markdown(f"#### 🎯 Multi-Instance Detection (Found {len(res['multiple_matches'])} candidate regions)")
                    df_multi = pd.DataFrame([
                        {
                            "Instance": f"#{idx+1}",
                            "Top-Left (X, Y)": f"({m['top_left'][0]}, {m['top_left'][1]})",
                            "Bottom-Right (X, Y)": f"({m['bottom_right'][0]}, {m['bottom_right'][1]})",
                            "Response Score": f"{m['score']:.4f}"
                        }
                        for idx, m in enumerate(res["multiple_matches"])
                    ])
                    st.dataframe(df_multi, use_container_width=True)

        st.markdown("---")
        st.markdown("""
            <div class="lab-card">
                <div class="lab-card-header">💡 Academic Discussion: Strengths, Limitations & Real-World Uses</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; font-size: 0.9rem; color: #334155;">
                    <div>
                        <strong>✅ Advantages:</strong>
                        <ul>
                            <li><strong>Zero Training Required:</strong> Operates instantaneously on raw pixel intensity matrices without dataset collection or backpropagation.</li>
                            <li><strong>Mathematical Simplicity:</strong> Deterministic sliding correlation window with exact global optimum guarantee.</li>
                            <li><strong>Sub-pixel Accuracy:</strong> Works exceptionally well for rigid, static industrial fiducial markers.</li>
                        </ul>
                    </div>
                    <div>
                        <strong>⚠️ Limitations:</strong>
                        <ul>
                            <li><strong>Scale & Rotation Invariance:</strong> Highly sensitive to object scaling or rotation; requires multi-scale pyramids to compensate.</li>
                            <li><strong>Illumination Sensitivity:</strong> Unnormalized methods fail drastically under non-uniform illumination shifts.</li>
                            <li><strong>Computational Complexity:</strong> $O((W-w+1)(H-h+1) \cdot w \cdot h)$ spatial complexity.</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.88rem; color: #475569;">
                    <strong>🏭 Real-World Applications:</strong> PCB component alignment, watermarking detection, optical character recognition (OCR) glyph localization, and robotic pick-and-place fiducial tracking.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please provide both the main search image and the query template to run the demonstration.")


# ======================================================================================
# 3. MODULE 2 — VIOLA-JONES FACE DETECTION
# ======================================================================================
elif nav_selection == "👤 Viola–Jones":
    st.markdown("""
        <div class="main-header">
            <h1>👤 Viola–Jones Face Detection</h1>
            <p class="subtitle">Classical Real-Time Object Detection using Haar Cascades, Integral Images & AdaBoost</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Academic Theory & Pipeline Breakdown (Click to Expand)", expanded=True):
        st.markdown("""
            <div class="theory-box">
                <h4>What is the Viola–Jones Algorithm?</h4>
                Proposed by <strong>Paul Viola and Michael Jones in 2001</strong>, this seminal framework was the first real-time object detection algorithm capable of processing face video frames at 15+ FPS on standard CPUs. It represents the pinnacle of classical computer vision feature engineering.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🏛️ The Four Foundational Pillars of Viola–Jones")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">1. Haar-like Features</div>
                    Rectangular scalar features computed as the difference between sums of pixels under white and black rectangular regions:
                    $$\\text{Feature Value} = \\sum \\text{Pixels}_{\\text{White}} - \\sum \\text{Pixels}_{\\text{Black}}$$
                    <small>Captures natural facial structural contrasts: e.g., the eye region is systematically darker than the upper cheeks; the nose bridge is brighter than nasal flanks.</small>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">2. Integral Image (Summed-Area Table)</div>
                    An intermediate spatial lookup matrix $II(x,y)$ where each entry contains the sum of all pixels above and to the left:
                    $$II(x,y) = \\sum_{x' \\le x, y' \\le y} I(x', y')$$
                    $$\\text{Sum of Rectangle } ABCD = D - B - C + A$$
                    <small>Enables computing any arbitrary rectangular pixel sum in <strong>constant $O(1)$ time</strong> regardless of size!</small>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">3. AdaBoost Feature Selection</div>
                    From an exhaustive set of over <strong>160,000+ candidate Haar features</strong> in a $24 \\times 24$ sub-window, AdaBoost iteratively selects a tiny subset (approx. 6,000) of the most discriminative weak classifiers $h_j(x)$ and weights them:
                    $$H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)\\right)$$
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">4. Attentional Cascade Classifier</div>
                    Chains multiple stages of classifiers in an early-reject cascade.
                    <ul>
                        <li>Stage 1 (2 features): Eliminates ~50% of non-face background windows instantly.</li>
                        <li>Subsequent stages: Apply deeper classifiers only to survivor windows.</li>
                    </ul>
                    <small>Achieves extraordinary runtime speed by rejecting 99% of background with minimal compute.</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="workflow-container">
                <div class="workflow-step">Haar-like Features</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">Integral Image $O(1)$</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">AdaBoost Training</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">Cascade Classifier</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">Multi-Scale Face Detection</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧪 Interactive Viola–Jones Detection Lab")

    vj_source = st.radio("Select Input Source:", ["Use Preset Academic Sample", "Upload Custom Image"], horizontal=True, key="vj_src")
    
    vj_img_rgb = None
    vj_catalog = get_sample_images_catalog()["Viola-Jones Faces"]
    
    if vj_source == "Use Preset Academic Sample":
        vj_choice = st.selectbox("Select Test Image:", list(vj_catalog.keys()))
        vj_img_rgb, _, _, err = load_image_from_path(vj_catalog[vj_choice]["path"])
        if err:
            st.error(err)
        else:
            st.caption(f"ℹ️ **Subject description:** {vj_catalog[vj_choice]['desc']}")
    else:
        vj_up = st.file_uploader("Upload Portrait or Group Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="vj_upload")
        if vj_up:
            vj_img_rgb, _, _, err = load_image_from_upload(vj_up)
            if err:
                st.error(err)

    if vj_img_rgb is not None:
        st.markdown("---")
        st.markdown("#### 🎚️ Detection Hyperparameters Tuning")
        
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            scale_factor = st.slider(
                "Scale Factor (scaleFactor):",
                min_value=1.05,
                max_value=1.50,
                value=1.10,
                step=0.05,
                help="Specifies how much image size is reduced at each image pyramid scale. Lower = finer granularity & slower; Higher = faster but may skip faces."
            )
        with p2:
            min_neighbors = st.slider(
                "Min Neighbors (minNeighbors):",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="How many neighboring candidate rectangles should retain to pass detection. Higher = fewer false positives."
            )
        with p3:
            min_size_val = st.slider(
                "Min Face Size (px):",
                min_value=20,
                max_value=120,
                value=30,
                step=10,
                help="Minimum bounding box dimension (width and height in pixels)."
            )
        with p4:
            enable_eyes = st.checkbox("Include Eye Cascade", value=True, help="Runs secondary eye detector within detected face ROIs")
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            run_vj_btn = st.button("🚀 Detect Faces", type="primary", use_container_width=True)

        # Run Detection
        t_start = time.time()
        vj_res = detect_faces_viola_jones(
            image_rgb=vj_img_rgb,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=(min_size_val, min_size_val),
            detect_eyes=enable_eyes
        )
        t_elapsed = (time.time() - t_start) * 1000.0

        if not vj_res["success"]:
            st.error(f"⚠️ {vj_res['error']}")
        else:
            st.markdown("### 📊 Detection Results & Visual Representations")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faces Detected", f"{vj_res['face_count']} Face(s)")
            m2.metric("Inference Time", f"{t_elapsed:.1f} ms")
            m3.metric("Pyramid Scale Factor", f"{scale_factor}")
            m4.metric("Min Neighbors Filter", f"{min_neighbors}")
            
            # Tabbed Visualizations
            tab1, tab2, tab3 = st.tabs(["📸 Detected Faces (Annotated)", "🧮 Grayscale & Integral Image", "🖼️ Extracted Face Crops"])
            
            with tab1:
                st.image(
                    vj_res["annotated_image"],
                    caption=f"Viola-Jones Detection Output ({vj_res['face_count']} face(s) localized with bounding boxes)",
                    use_container_width=True
                )
                if vj_res['face_count'] == 0:
                    st.warning("⚠️ No faces detected with the current hyperparameters. Try lowering 'Scale Factor' to 1.05 or reducing 'Min Neighbors' to 2.")

            with tab2:
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.image(vj_res["gray_image"], caption="1. Grayscale Converted Channel", use_container_width=True)
                with col_g2:
                    st.image(vj_res["integral_preview"], caption="2. Integral Image $II(x,y)$ Log-Intensity Representation", use_container_width=True)

            with tab3:
                if vj_res["face_crops"]:
                    st.markdown("#### Segmented Face Regions:")
                    crop_cols = st.columns(min(4, len(vj_res["face_crops"])))
                    for idx, crop_info in enumerate(vj_res["face_crops"]):
                        with crop_cols[idx % len(crop_cols)]:
                            st.image(crop_info["crop"], caption=f"Face #{crop_info['id']} | Box: {crop_info['box'][2]}x{crop_info['box'][3]}", use_container_width=True)
                else:
                    st.info("No face crops available. Adjust hyperparameters to detect candidate faces.")

        st.markdown("---")
        st.markdown("""
            <div class="lab-card">
                <div class="lab-card-header">💡 Why is Viola–Jones Considered a Classical CV Milestone?</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    Unlike modern deep learning methods that require billions of floating-point operations (FLOPs) and dedicated GPUs, Viola–Jones relies on <strong>handcrafted geometric intensity heuristics</strong> and the mathematical elegance of the <strong>Integral Image</strong>. This allowed real-time detection on 2001-era hardware (Pentium III processors).
                    However, it is strictly limited to near-frontal poses, uniform illumination, and unoccluded facial appearances.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please provide an image to perform Viola–Jones face detection.")


# ======================================================================================
# 4. MODULE 3 — DEEPFACE
# ======================================================================================
elif nav_selection == "🧠 DeepFace":
    st.markdown("""
        <div class="main-header">
            <h1>🧠 DeepFace Facial Analytics</h1>
            <p class="subtitle">Deep Learning Framework for Age, Gender, Emotion Estimation & Face Verification</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Academic Theory & Deep Learning Architecture (Click to Expand)", expanded=True):
        st.markdown("""
            <div class="theory-box">
                <h4>What is DeepFace?</h4>
                DeepFace is a modern deep-learning facial analysis framework pioneered by researchers at Facebook (Taigman et al., 2014). It applies deep Convolutional Neural Networks (CNNs) to facial imagery to solve complex recognition and attribute estimation tasks with human-level accuracy.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="workflow-container">
                <div class="workflow-step">1. Input Face Image</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">2. Face Localization</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">3. 3D Piecewise Alignment</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">4. Deep CNN Backbone</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">5. Representation Vector</div>
                <div class="workflow-arrow">➔</div>
                <div class="workflow-step">6. Multi-Attribute Output</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="lab-card">
                <div class="lab-card-header">⚖️ Classical Haar Cascades vs. DeepFace Representations</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    While Viola–Jones uses rigid, handcrafted binary rectangular masks, DeepFace utilizes millions of trainable parameters across hierarchical convolutional layers. Early layers extract edges and Gabor-like filters; middle layers learn facial components (noses, lips, eyes); deep layers synthesize holistic identity and affective semantic representations invariant to illumination, pose, and expression.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧪 DeepFace Demonstration Suite")
    
    deepface_mode = st.radio("Choose Analysis Mode:", ["Mode A: Single Image Facial Analytics", "Mode B: Pairwise Face Verification"], horizontal=True)

    # ----------------------------------------------------------------------------------
    # MODE A: SINGLE IMAGE ANALYSIS
    # ----------------------------------------------------------------------------------
    if deepface_mode == "Mode A: Single Image Facial Analytics":
        st.markdown("#### Mode A: Multi-Attribute Estimation (Age, Gender, Emotion, Ethnicity)")
        
        df_src = st.radio("Input Source:", ["Use Preset Academic Sample", "Upload Custom Image"], horizontal=True, key="df_src_a")
        
        df_img_rgb = None
        df_catalog = get_sample_images_catalog()["Viola-Jones Faces"]
        
        if df_src == "Use Preset Academic Sample":
            df_choice = st.selectbox("Select Subject:", list(df_catalog.keys()), key="df_sel_a")
            df_img_rgb, _, _, err = load_image_from_path(df_catalog[df_choice]["path"])
            if err:
                st.error(err)
        else:
            df_up = st.file_uploader("Upload Face Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="df_up_a")
            if df_up:
                df_img_rgb, _, _, err = load_image_from_upload(df_up)
                if err:
                    st.error(err)

        if df_img_rgb is not None:
            c_run1, c_run2 = st.columns([2, 1])
            with c_run1:
                backend = st.selectbox("Detector Backend:", ["opencv", "ssd", "mtcnn", "retinaface"], index=0, help="Underlying face localization engine")
            with c_run2:
                st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
                run_df_a = st.button("🚀 Analyze Face Attributes", type="primary", use_container_width=True)

            if run_df_a or True:
                with st.spinner("Analyzing facial representations..."):
                    res_a = analyze_single_face(df_img_rgb, detector_backend=backend)
                    
                if not res_a["success"]:
                    st.error(f"{res_a['error']}")
                else:
                    if res_a.get("is_fallback"):
                        st.warning(f"ℹ️ {res_a.get('fallback_reason', 'Running in educational demonstration mode.')}")

                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1.1, 1.9])
                    
                    with res_col1:
                        st.image(res_a["face_crop"], caption="Segmented Face Crop", use_container_width=True)
                        st.markdown(f"""
                            <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem;">
                                <div><strong>Backend:</strong> <code>{res_a['model_backend']}</code></div>
                                <div><strong>ROI Region:</strong> <code>{res_a['region']}</code></div>
                            </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        st.markdown("#### 🎯 Model Predictions & Estimations")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Estimated Age", f"~{res_a['age']} yrs")
                        m2.metric("Predicted Gender", f"{res_a['dominant_gender']}", f"{res_a['gender_conf']:.1f}% conf")
                        m3.metric("Dominant Emotion", f"{res_a['dominant_emotion'].capitalize()}")
                        
                        # Emotion Distribution Chart
                        st.markdown("##### 🎭 Emotion Probability Distribution:")
                        emo_data = res_a["emotion_dist"]
                        df_emo = pd.DataFrame({
                            "Emotion": [k.capitalize() for k in emo_data.keys()],
                            "Confidence (%)": [float(v) for v in emo_data.values()]
                        }).sort_values(by="Confidence (%)", ascending=True)
                        
                        fig_emo, ax_emo = plt.subplots(figsize=(6, 2.8), dpi=110)
                        colors = ['#3b82f6' if e.lower() == res_a['dominant_emotion'].lower() else '#94a3b8' for e in df_emo["Emotion"]]
                        ax_emo.barh(df_emo["Emotion"], df_emo["Confidence (%)"], color=colors)
                        ax_emo.set_xlabel("Probability Score (%)", fontsize=8)
                        ax_emo.tick_params(labelsize=8)
                        plt.tight_layout()
                        st.pyplot(fig_emo, use_container_width=True)
                        plt.close(fig_emo)

                    st.caption("⚠️ **Academic Note:** All demographic and affective attributes are probabilistic statistical estimations generated by convolutional representations and should not be treated as absolute ground truth.")
        else:
            st.warning("⚠️ Please provide an image to perform facial attribute analysis.")

    # ----------------------------------------------------------------------------------
    # MODE B: FACE VERIFICATION
    # ----------------------------------------------------------------------------------
    else:
        st.markdown("#### Mode B: Pairwise Face Identity Verification")
        st.caption("Compares two face images to determine whether they represent the same individual based on deep feature distance.")
        
        pair_catalog = get_sample_images_catalog()["DeepFace / FaceNet Pairs"]
        pair_src = st.radio("Verification Pairs Source:", ["Use Preset Verification Pairs", "Upload Custom Image Pair"], horizontal=True, key="df_pair_src")
        
        img1_rgb, img2_rgb = None, None
        expected_same = None
        
        if pair_src == "Use Preset Verification Pairs":
            pair_choice = st.selectbox("Select Test Pair:", list(pair_catalog.keys()))
            pair_data = pair_catalog[pair_choice]
            expected_same = pair_data["same"]
            
            img1_rgb, _, _, err1 = load_image_from_path(pair_data["img1"])
            img2_rgb, _, _, err2 = load_image_from_path(pair_data["img2"])
            if err1 or err2:
                st.error(f"Error: {err1 or err2}")
            else:
                st.caption(f"ℹ️ **Ground Truth:** {pair_data['desc']}")
        else:
            c_up1, c_up2 = st.columns(2)
            with c_up1:
                u1 = st.file_uploader("Upload Image 1 (Subject A)", type=["jpg", "jpeg", "png"], key="df_u1")
                if u1:
                    img1_rgb, _, _, err = load_image_from_upload(u1)
                    if err:
                        st.error(err)
            with c_up2:
                u2 = st.file_uploader("Upload Image 2 (Subject B)", type=["jpg", "jpeg", "png"], key="df_u2")
                if u2:
                    img2_rgb, _, _, err = load_image_from_upload(u2)
                    if err:
                        st.error(err)

        if img1_rgb is not None and img2_rgb is not None:
            st.markdown("---")
            
            # Preview pair
            prev1, prev2 = st.columns(2)
            with prev1:
                st.image(img1_rgb, caption="Subject Image 1", use_container_width=True)
            with prev2:
                st.image(img2_rgb, caption="Subject Image 2", use_container_width=True)

            # Verification controls
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                model_sel = st.selectbox("Deep Model Backbone:", ["VGG-Face", "Facenet", "OpenFace", "DeepFace"], index=0)
            with vc2:
                dist_metric = st.selectbox("Distance Metric:", ["cosine", "euclidean", "euclidean_l2"], index=0)
            with vc3:
                st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
                run_verif_btn = st.button("🚀 Verify Identity Pair", type="primary", use_container_width=True)

            if run_verif_btn or True:
                with st.spinner("Computing deep representations and feature distance..."):
                    v_res = verify_face_pair(img1_rgb, img2_rgb, model_name=model_sel, distance_metric=dist_metric)
                    
                if not v_res["success"]:
                    st.error(f"{v_res['error']}")
                else:
                    if v_res.get("is_fallback"):
                        st.warning(f"ℹ️ {v_res.get('fallback_reason', 'Running in educational fallback mode.')}")
                        
                    st.markdown("---")
                    
                    # Verdict Card
                    is_match = v_res["verified"]
                    card_class = "verify-match" if is_match else "verify-no-match"
                    verdict_icon = "✅" if is_match else "❌"
                    verdict_text = "VERIFIED: SAME PERSON" if is_match else "NOT VERIFIED: DIFFERENT PERSONS"
                    
                    st.markdown(f"""
                        <div class="{card_class}">
                            <div style="font-size: 2.2rem; margin-bottom: 0.3rem;">{verdict_icon}</div>
                            <h3 style="margin: 0 0 0.4rem 0;">{verdict_text}</h3>
                            <p style="margin: 0; font-size: 0.95rem;">
                                Computed Distance: <strong>{v_res['distance']:.4f}</strong> | Decision Threshold: <strong>{v_res['threshold']:.4f}</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    vm1, vm2, vm3, vm4 = st.columns(4)
                    vm1.metric("Verification Status", "Match" if is_match else "No Match")
                    vm2.metric("Computed Distance", f"{v_res['distance']:.4f}")
                    vm3.metric("Decision Threshold", f"{v_res['threshold']:.4f}")
                    vm4.metric("Model & Metric", f"{v_res['model']} ({dist_metric})")
        else:
            st.warning("⚠️ Please provide both Image 1 and Image 2 to run identity verification.")


# ======================================================================================
# 5. MODULE 4 — FACENET
# ======================================================================================
elif nav_selection == "🔐 FaceNet":
    st.markdown("""
        <div class="main-header">
            <h1>🔐 FaceNet Embeddings & Metric Learning</h1>
            <p class="subtitle">Direct Mapping from Face Imagery to 128-d/512-d Euclidean Vector Space via Triplet Loss</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Academic Theory & Triplet Loss Formulation (Click to Expand)", expanded=True):
        st.markdown("""
            <div class="theory-box">
                <h4>What is FaceNet?</h4>
                Introduced by Google researchers <strong>Schroff, Kalenichenko, and Philbin in 2015</strong>, FaceNet fundamentally revolutionized face recognition by abandoning classification layer bottlenecks. Instead, it trains a deep convolutional network (Inception-ResNet) to directly map facial images into a compact <strong>$d$-dimensional Euclidean embedding space</strong> ($d = 128$ or $512$) where distances directly correspond to facial similarity.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📐 Mathematical Concept: The Triplet Loss Objective")
        
        t_col1, t_col2 = st.columns([1.2, 1.0])
        with t_col1:
            st.markdown("""
                <div class="formula-box">
                    <div class="formula-title">Triplet Loss Mathematical Formulation</div>
                    $$\\mathcal{L}(A, P, N) = \\max \\left( 0, \\|f(A) - f(P)\\|^2_2 - \\|f(A) - f(N)\\|^2_2 + \\alpha \\right)$$
                    Where:
                    <ul>
                        <li>$f(x) \\in \\mathbb{R}^d$ is the $L_2$-normalized embedding vector ($\\|f(x)\\|_2 = 1$).</li>
                        <li><strong>Anchor ($A$):</strong> Reference image of Person X.</li>
                        <li><strong>Positive ($P$):</strong> Another image of the <em>same</em> Person X.</li>
                        <li><strong>Negative ($N$):</strong> Image of a <em>different</em> Person Y.</li>
                        <li><strong>Margin $\\alpha$:</strong> Enforced geometric separation buffer.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with t_col2:
            st.markdown("""
                <div class="lab-card">
                    <div class="lab-card-header">🎯 Triplet Mining Strategies</div>
                    <div style="font-size: 0.85rem; color: #334155; line-height: 1.5;">
                        To guarantee steady gradient descent and prevent trivial convergence, FaceNet employs active triplet mining:
                        <ul>
                            <li><strong>Hard Positive:</strong> $\\operatorname{argmax}_P \\|f(A) - f(P)\\|^2$</li>
                            <li><strong>Hard Negative:</strong> $\\operatorname{argmin}_N \\|f(A) - f(N)\\|^2$</li>
                            <li><strong>Semi-Hard Negative:</strong> $\\|f(A)-f(P)\\|^2 < \\|f(A)-f(N)\\|^2 < \\|f(A)-f(P)\\|^2 + \\alpha$</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Interactive Triplet Loss visualizer
        st.markdown("##### 🎛️ Interactive Triplet Loss Margin Explorer")
        sim_col1, sim_col2 = st.columns([1.2, 1.8])
        with sim_col1:
            alpha_val = st.slider("Enforced Margin (α):", 0.1, 1.0, 0.3, 0.05)
            d_ap = st.slider("Anchor-Positive Distance ||f(A)-f(P)||²:", 0.05, 1.0, 0.25, 0.05)
            d_an = st.slider("Anchor-Negative Distance ||f(A)-f(N)||²:", 0.1, 2.0, 0.70, 0.05)
            
            computed_loss = max(0.0, d_ap - d_an + alpha_val)
            st.metric("Computed Triplet Loss $\\mathcal{L}$", f"{computed_loss:.3f}", "Optimal (0.0)" if computed_loss == 0 else "Active Gradient")

        with sim_col2:
            fig_trip, ax_trip = plt.subplots(figsize=(6, 2.5), dpi=110)
            ax_trip.barh(["Anchor-Positive", "Anchor-Negative (Required > Pos+α)", "Anchor-Negative (Actual)"], [d_ap, d_ap + alpha_val, d_an], color=["#3b82f6", "#f59e0b", "#10b981" if d_an >= d_ap + alpha_val else "#ef4444"])
            ax_trip.set_xlabel("Euclidean Distance Squared", fontsize=8)
            ax_trip.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_trip, use_container_width=True)
            plt.close(fig_trip)

    st.markdown("### 🧪 Interactive FaceNet Embedding & Distance Laboratory")

    fn_pair_catalog = get_sample_images_catalog()["DeepFace / FaceNet Pairs"]
    fn_src = st.radio("Select Input Data Source:", ["Use Preset Evaluation Pairs", "Upload Custom Image Pair"], horizontal=True, key="fn_src_radio")
    
    img_a, img_b = None, None
    expected_same = None
    
    if fn_src == "Use Preset Evaluation Pairs":
        fn_choice = st.selectbox("Select Test Pair:", list(fn_pair_catalog.keys()), key="fn_sel_box")
        p_info = fn_pair_catalog[fn_choice]
        expected_same = p_info["same"]
        
        img_a, _, _, err1 = load_image_from_path(p_info["img1"])
        img_b, _, _, err2 = load_image_from_path(p_info["img2"])
        if err1 or err2:
            st.error(f"Error: {err1 or err2}")
        else:
            st.caption(f"ℹ️ **Ground Truth Info:** {p_info['desc']}")
    else:
        fn_u1, fn_u2 = st.columns(2)
        with fn_u1:
            u_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="fn_ua")
            if u_a:
                img_a, _, _, err = load_image_from_upload(u_a)
                if err:
                    st.error(err)
        with fn_u2:
            u_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="fn_ub")
            if u_b:
                img_b, _, _, err = load_image_from_upload(u_b)
                if err:
                    st.error(err)

    if img_a is not None and img_b is not None:
        st.markdown("---")
        
        # Display side-by-side
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.image(img_a, caption="Subject Image A", use_container_width=True)
        with p_c2:
            st.image(img_b, caption="Subject Image B", use_container_width=True)

        # Controls
        fn_c1, fn_c2 = st.columns([1.5, 1.5])
        with fn_c1:
            thresh_slider = st.slider("Euclidean Decision Threshold:", 0.40, 1.20, 0.85, 0.05, help="Distances strictly below this threshold are classified as the same individual.")
        with fn_c2:
            st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
            run_fn_btn = st.button("🚀 Extract Embeddings & Compute Distance", type="primary", use_container_width=True)

        if run_fn_btn or True:
            with st.spinner("Generating FaceNet feature embeddings..."):
                emb_res_a = extract_facenet_embedding(img_a)
                emb_res_b = extract_facenet_embedding(img_b)
                
            if not emb_res_a["success"] or not emb_res_b["success"]:
                st.error(f"⚠️ {emb_res_a.get('error') or emb_res_b.get('error')}")
            else:
                is_fallback = emb_res_a.get("is_fallback") or emb_res_b.get("is_fallback")
                if is_fallback:
                    st.warning("ℹ️ **Demonstration Notice:** Running feature projection simulation mode. All vector calculations and metric comparisons are live and mathematically accurate.")

                # Compute metric distance
                cmp_res = compare_face_embeddings(
                    emb_res_a["embedding"],
                    emb_res_b["embedding"],
                    distance_threshold=thresh_slider
                )
                
                st.markdown("### 📊 Embedding Metric Comparison & Verdict")
                
                # Verdict Box
                is_same = cmp_res["is_same_identity"]
                card_class = "verify-match" if is_same else "verify-no-match"
                v_icon = "✅" if is_same else "❌"
                
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">{v_icon}</div>
                        <h3 style="margin: 0 0 0.3rem 0;">{cmp_res['interpretation']}</h3>
                        <p style="margin: 0; font-size: 0.95rem;">
                            Euclidean Distance: <strong>{cmp_res['euclidean_distance']}</strong> (Threshold: {cmp_res['threshold']}) | Cosine Similarity: <strong>{cmp_res['cosine_similarity']}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Embedding Dimension", f"{emb_res_a['dim']}-D Vector")
                m2.metric("Euclidean Distance", f"{cmp_res['euclidean_distance']}")
                m3.metric("Cosine Similarity", f"{cmp_res['cosine_similarity']}")
                m4.metric("Similarity Index", f"{cmp_res['match_confidence']}%")
                
                # Visual Plot of Embedding Dimensions
                fig_emb = plot_embedding_comparison(
                    emb_a=emb_res_a["embedding"],
                    emb_b=emb_res_b["embedding"],
                    metric_dist=cmp_res["euclidean_distance"],
                    cosine_sim=cmp_res["cosine_similarity"],
                    same_verdict=is_same,
                    threshold=thresh_slider
                )
                st.pyplot(fig_emb, use_container_width=True)
                plt.close(fig_emb)
                
                # Raw Embedding Vector Values Accordion
                with st.expander("🔍 Inspect Raw Feature Vector Activations (First 20 Dimensions)"):
                    raw_df = pd.DataFrame({
                        "Dimension Index": [f"Dim #{i}" for i in range(20)],
                        "Embedding Vector A": [f"{v:.6f}" for v in emb_res_a["embedding"][:20]],
                        "Embedding Vector B": [f"{v:.6f}" for v in emb_res_b["embedding"][:20]],
                        "Absolute Difference |Δ|": [f"{abs(a-b):.6f}" for a, b in zip(emb_res_a["embedding"][:20], emb_res_b["embedding"][:20])]
                    })
                    st.dataframe(raw_df, use_container_width=True)

        st.markdown("---")
        st.markdown("""
            <div class="lab-card">
                <div class="lab-card-header">💡 Why FaceNet Embeddings Outperform Classical Face Verification</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    Classical methods (such as template matching or raw pixel correlation) compare images in the unconstrained pixel space $\\mathbb{R}^{H \\times W \\times 3}$, where slight head turns, lighting changes, or facial expressions cause immense Euclidean error.
                    FaceNet learns a manifold projection into a compact sphere $\\mathcal{S}^{d-1}$ where intraclass variance (same person under different lighting/poses) is minimized and interclass variance (different people) is maximized.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please provide two face images to extract embeddings and perform metric comparisons.")


# ======================================================================================
# 6. COMPARISON PAGE
# ======================================================================================
elif nav_selection == "📊 Comparison":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Comprehensive Algorithmic Comparison</h1>
            <p class="subtitle">Multi-Criteria Evaluation: Classical Computer Vision vs. Deep Learning Approaches</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Academic Comparison Matrix")
    
    comp_data = [
        {
            "Algorithm": "🔍 Template Matching",
            "Main Purpose": "Template localization",
            "Approach": "Classical image matching",
            "Input": "Image + template",
            "Output": "Match location $(x,y)$",
            "Training": "No training required",
            "Main Strength": "Simple object/template localization",
            "Limitation": "Sensitive to scale/rotation"
        },
        {
            "Algorithm": "👤 Viola–Jones",
            "Main Purpose": "Face detection",
            "Approach": "Classical CV (Haar Cascade)",
            "Input": "Image / video stream",
            "Output": "Face bounding boxes $(x,y,w,h)$",
            "Training": "Pre-trained cascade (AdaBoost)",
            "Main Strength": "Fast CPU real-time face detection",
            "Limitation": "Limited to frontal & unoccluded faces"
        },
        {
            "Algorithm": "🧠 DeepFace",
            "Main Purpose": "Facial analysis & verification",
            "Approach": "Deep learning (CNN)",
            "Input": "Face / image",
            "Output": "Facial predictions / verification",
            "Training": "Pre-trained deep models",
            "Main Strength": "Rich multi-task facial analysis",
            "Limitation": "Computationally heavier (FLOPs)"
        },
        {
            "Algorithm": "🔐 FaceNet",
            "Main Purpose": "Face embeddings / recognition",
            "Approach": "Deep metric learning (Triplet Loss)",
            "Input": "Face images",
            "Output": "128-d/512-d embeddings & similarity",
            "Training": "Pre-trained Inception-ResNet",
            "Main Strength": "Compact identity representation",
            "Limitation": "Requires vector database & thresholding"
        }
    ]
    
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(
        df_comp.set_index("Algorithm"),
        use_container_width=True,
        height=200
    )

    st.markdown("---")
    
    r_col1, r_col2 = st.columns([1.2, 1.0])
    
    with r_col1:
        st.markdown("### 🕸️ Multi-Criteria Radar Benchmark")
        fig_radar = plot_radar_comparison()
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)
        
    with r_col2:
        st.markdown("### 🎯 Algorithm Selection Guide (Which should you use?)")
        
        st.markdown("""
            <div class="lab-card">
                <div style="font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">Use Template Matching when:</div>
                <div style="font-size: 0.88rem; color: #475569;">
                    • Locating rigid electronic parts on a PCB.<br>
                    • Watermark or logo verification on standard documents.<br>
                    • Scale and orientation are fixed and predictable.
                </div>
            </div>
            
            <div class="lab-card">
                <div style="font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">Use Viola–Jones when:</div>
                <div style="font-size: 0.88rem; color: #475569;">
                    • Deploying on ultra-low-power microcontrollers (e.g. Raspberry Pi, IoT).<br>
                    • Low-latency CPU video streams where only frontal face presence is needed.<br>
                    • Hardware has zero GPU capability.
                </div>
            </div>
            
            <div class="lab-card">
                <div style="font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">Use DeepFace / FaceNet when:</div>
                <div style="font-size: 0.88rem; color: #475569;">
                    • Building large-scale 1:N face identification systems (e.g. airport e-gates).<br>
                    • Robustness across age progression, cosmetics, and wild lighting is essential.<br>
                    • Deep affective computing & emotion/age analytics are needed.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ======================================================================================
# 7. REFERENCES & VIVA PREPARATION
# ======================================================================================
elif nav_selection == "📚 References & Viva Prep":
    st.markdown("""
        <div class="main-header">
            <h1>📚 Academic References & Viva Voce Preparation</h1>
            <p class="subtitle">Seminal Research Publications, Citations & Viva Examination Question Bank</p>
        </div>
    """, unsafe_allow_html=True)

    tab_ref1, tab_ref2 = st.tabs(["📄 Seminal Research Citations", "🎓 Viva Voce Oral Exam Q&A Bank"])

    with tab_ref1:
        st.markdown("### 🏛️ Foundational Literature")
        
        st.markdown("""
            <div class="lab-card">
                <div class="lab-card-header">1. Viola–Jones Classical Face Detection (2001)</div>
                <p style="font-size: 0.9rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Viola, P., & Jones, M. (2001).</strong> <em>"Rapid object detection using a boosted cascade of simple features."</em> In Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2001), Vol. 1, pp. I-511. IEEE.
                </p>
                <div style="font-size: 0.82rem; color: #64748b;">Key Contribution: Integral image for constant-time rectangular feature extraction and attentional cascade classifier.</div>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">2. FaceNet: Unified Face Recognition & Clustering (2015)</div>
                <p style="font-size: 0.9rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Schroff, F., Kalenichenko, D., & Philbin, J. (2015).</strong> <em>"FaceNet: A unified embedding for face recognition and clustering."</em> In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015), pp. 815-823.
                </p>
                <div style="font-size: 0.82rem; color: #64748b;">Key Contribution: Direct Euclidean metric embedding trained via Triplet Loss with online hard-negative mining.</div>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">3. DeepFace: Closing the Gap to Human-Level Performance (2014)</div>
                <p style="font-size: 0.9rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. (2014).</strong> <em>"DeepFace: Closing the gap to human-level performance in face verification."</em> In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2014), pp. 1701-1708.
                </p>
                <div style="font-size: 0.82rem; color: #64748b;">Key Contribution: 3D piecewise affine face alignment with locally connected deep neural network layers.</div>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">4. OpenCV Structural Analysis & Template Matching Documentation</div>
                <p style="font-size: 0.9rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Bradski, G. (2000).</strong> <em>"The OpenCV Library."</em> Dr. Dobb's Journal of Software Tools. Official OpenCV documentation on <code>cv2.matchTemplate</code> and <code>cv2.CascadeClassifier</code>.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with tab_ref2:
        st.markdown("### 🎓 Viva Voce Oral Examination Questions & Model Answers")
        
        viva_qa = [
            (
                "Q1: Why does Template Matching fail when an object is rotated or zoomed?",
                r"Template matching operates on raw pixel intensity matrices using a fixed-dimension sliding window. It does not extract affine-invariant or scale-invariant features. If the object rotates or changes scale, the pixel values no longer align with the template, resulting in low correlation response."
            ),
            (
                "Q2: What is the primary computational benefit of the Integral Image in Viola–Jones?",
                r"An Integral Image allows calculating the sum of pixel intensities within ANY arbitrary rectangular region in exactly 4 array lookups ($D - B - C + A$), which is $O(1)$ constant time complexity, completely independent of the rectangle's size."
            ),
            (
                "Q3: How does AdaBoost select features in Viola–Jones?",
                r"AdaBoost evaluates thousands of candidate Haar-like features against the training set. In each round, it selects the single feature that minimizes weighted classification error, increases weights on misclassified examples, and linearly combines the selected weak classifiers into a strong classifier."
            ),
            (
                "Q4: Why are early stages of the Viola–Jones cascade classifier so shallow (e.g. 2 features)?",
                r"The cascade is an attentional reject cascade. Since >99% of random image patches in a scene do not contain faces, a shallow 2-feature stage can reject over 50% of negative windows with minimal CPU cycles, allowing the detector to focus computation only on candidate face regions."
            ),
            (
                "Q5: What is the fundamental difference between classification-based face recognition and metric embeddings (FaceNet)?",
                r"Classification-based networks output probabilities over a fixed set of $N$ known training identities via a Softmax layer. FaceNet directly maps faces into an open-set Euclidean vector space ($\mathbb{R}^{128}$ or $\mathbb{R}^{512}$). To recognize a new person, one simply stores their embedding without retraining the neural network."
            ),
            (
                "Q6: What is Triplet Loss and why is the margin α necessary?",
                r"Triplet Loss enforces $\|f(A) - f(P)\|^2 + \alpha < \|f(A) - f(N)\|^2$. The margin $\alpha$ prevents the trivial degenerate solution where the network collapses all embeddings to zero ($f(x) = 0$), forcing a minimum geometric separation between distinct identities."
            ),
            (
                "Q7: Why are embeddings L2-normalized in FaceNet?",
                r"L2-normalization ($\|f(x)\|_2 = 1$) constrains all embeddings onto the surface of a hypersphere $\mathcal{S}^{d-1}$. This establishes an exact mathematical equivalence between Euclidean distance and Cosine distance ($\|a-b\|^2 = 2 - 2(a \cdot b)$) and prevents unbounded vector magnitudes."
            ),
            (
                "Q8: What is the difference between TM_CCOEFF_NORMED and TM_SQDIFF_NORMED in OpenCV?",
                r"TM_CCOEFF_NORMED subtracts the mean intensity from the template and image, producing a normalized cross-correlation in $[-1, 1]$ where 1 is the best match. TM_SQDIFF_NORMED computes normalized squared difference in $[0, 1]$ where 0 is a perfect match (global minimum)."
            )
        ]
        
        for q, a in viva_qa:
            with st.expander(f"📌 {q}"):
                st.markdown(f"**Model Answer:** {a}")


# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown("""
    <div class="lab-footer">
        <strong>IVA Assignment 2 – Computer Vision & Face Analysis Lab</strong><br>
        Developed for Academic Demonstration & Viva Evaluation | Student: Sharmili | Image and Video Analytics (IVA)
    </div>
""", unsafe_allow_html=True)
