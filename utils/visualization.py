"""
Visualization Utilities for IVA Assignment 2 Lab.
Provides Matplotlib and UI visualizers for heatmaps, bounding boxes, embeddings, and radar charts.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

# Set modern matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_template_matching_results(
    main_rgb: np.ndarray,
    template_rgb: np.ndarray,
    heatmap: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    score: float,
    method_name: str
) -> plt.Figure:
    """
    Creates a comprehensive 3-panel figure for template matching:
    1. Main Image with Bounding Box
    2. Template Image
    3. Matching Correlation Heatmap with peak indicator
    """
    fig = plt.figure(figsize=(14, 4.8), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.7, 1.3], wspace=0.25)
    
    # Panel 1: Detected Result
    ax1 = fig.add_subplot(gs[0, 0])
    res_img = main_rgb.copy()
    cv2.rectangle(res_img, top_left, bottom_right, (0, 230, 115), 3) # bright green
    # Add label tag
    label = f"Match: {score:.3f}"
    y_pos = max(20, top_left[1] - 8)
    cv2.putText(res_img, label, (top_left[0], y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 115), 2)
    
    ax1.imshow(res_img)
    ax1.set_title("📍 Detected Match Location", fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')
    
    # Panel 2: Template
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(template_rgb)
    ax2.set_title("🔍 Query Template", fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')
    
    # Panel 3: Correlation Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    # Normalize heatmap for display if needed
    cmap = 'viridis' if 'SQDIFF' not in method_name else 'viridis_r'
    im = ax3.imshow(heatmap, cmap=cmap)
    
    # Mark the match location on heatmap
    ax3.plot(top_left[0], top_left[1], 'r+', markersize=14, markeredgewidth=2.5, label='Peak Match')
    ax3.set_title(f"🔥 Correlation Response $R(x,y)$ ({method_name})", fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    ax3.legend(loc='upper right', frameon=True, fontsize=8)
    
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig

def draw_faces_bounding_boxes(
    rgb_image: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 200, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draws rectangular bounding boxes and indices for detected faces.
    """
    output = rgb_image.copy()
    for idx, (x, y, w, h) in enumerate(faces, 1):
        # Draw outer rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label background
        label = f"Face #{idx} ({w}x{h})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_top = max(0, y - lh - 8)
        cv2.rectangle(output, (x, bg_top), (x + lw + 8, y), color, -1)
        
        # Text in contrasting dark color
        cv2.putText(output, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 20, 30), 1, cv2.LINE_AA)
        
    return output

def plot_embedding_comparison(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    metric_dist: float,
    cosine_sim: float,
    same_verdict: bool,
    threshold: float = 0.68
) -> plt.Figure:
    """
    Plots a multi-panel visual embedding representation:
    1. First 25 dimensions bar comparison
    2. Vector difference profile
    3. Similarity gauge / distance metric
    """
    emb_a = np.asarray(emb_a).flatten()
    emb_b = np.asarray(emb_b).flatten()
    dim = len(emb_a)
    
    num_preview = min(30, dim)
    dims_x = np.arange(num_preview)
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), dpi=120, gridspec_kw={'height_ratios': [1.2, 0.8]})
    
    # Subplot 1: Side-by-side bar chart of first dimensions
    width = 0.38
    axes[0].bar(dims_x - width/2, emb_a[:num_preview], width=width, label="Image A Embedding", color="#2563eb", alpha=0.85)
    axes[0].bar(dims_x + width/2, emb_b[:num_preview], width=width, label="Image B Embedding", color="#e11d48", alpha=0.85)
    axes[0].set_title(f"FaceNet Feature Embeddings Profile (Preview: First {num_preview} of {dim} Dimensions)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Vector Dimension Index", fontsize=9)
    axes[0].set_ylabel("Activation Value", fontsize=9)
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[0].tick_params(labelsize=8)
    
    # Subplot 2: Difference vector (emb_a - emb_b)
    diff = emb_a[:num_preview] - emb_b[:num_preview]
    colors = ['#10b981' if abs(d) < 0.1 else '#f59e0b' if abs(d) < 0.25 else '#ef4444' for d in diff]
    axes[1].bar(dims_x, diff, color=colors, width=0.5, alpha=0.9)
    axes[1].axhline(0, color="#64748b", linestyle="--", linewidth=0.8)
    axes[1].set_title("Dimension-wise Residual $\\Delta_i = f(A)_i - f(B)_i$ (Smaller residual = Higher similarity)", fontsize=10, fontweight='bold')
    axes[1].set_xlabel("Vector Dimension Index", fontsize=9)
    axes[1].set_ylabel("Difference", fontsize=9)
    axes[1].tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig

def plot_radar_comparison() -> plt.Figure:
    """
    Produces a comparative radar chart evaluating the four methods across 5 core dimensions.
    """
    categories = ['Inference Speed', 'Pose/Angle Robustness', 'Expression Invariance', 'Identity Verification', 'Lightweight/No GPU']
    N = len(categories)
    
    # Normalized scores out of 5
    tm_scores = [4, 1, 1, 1, 5]
    vj_scores = [5, 2, 2, 1, 5]
    df_scores = [3, 4.5, 4.5, 4.5, 2.5]
    fn_scores = [3.5, 5, 5, 5, 2.5]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=110)
    
    def add_radar_trace(values, label, color):
        val = values + values[:1]
        ax.plot(angles, val, linewidth=2, linestyle='solid', label=label, color=color)
        ax.fill(angles, val, color=color, alpha=0.15)
        
    add_radar_trace(tm_scores, "Template Matching (Classical)", "#f59e0b")
    add_radar_trace(vj_scores, "Viola-Jones (Haar Cascade)", "#06b6d4")
    add_radar_trace(df_scores, "DeepFace (Deep CNN)", "#8b5cf6")
    add_radar_trace(fn_scores, "FaceNet (Metric Embeddings)", "#10b981")
    
    plt.xticks(angles[:-1], categories, size=9, fontweight='bold', color='#1e293b')
    ax.set_rlabel_position(20)
    plt.yticks([1, 2, 3, 4, 5], ["1 (Low)", "2", "3", "4", "5 (High)"], color="#64748b", size=7)
    plt.ylim(0, 5.5)
    plt.title("Multi-Criteria Algorithmic Comparison", size=12, fontweight='bold', pad=20, color='#0f172a')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=8.5)
    
    plt.tight_layout()
    return fig
