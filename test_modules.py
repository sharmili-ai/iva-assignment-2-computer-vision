"""
Comprehensive Automated Test Suite for IVA Assignment 2 Modules.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.template_matching import MATCH_METHODS, perform_template_matching
from modules.viola_jones import detect_faces_viola_jones, get_haar_cascade
from modules.deepface_module import analyze_single_face, verify_face_pair
from modules.facenet_module import extract_facenet_embedding, compare_face_embeddings
from utils.image_utils import load_image_from_path, get_sample_images_catalog, resize_for_display
from utils.visualization import (
    plot_template_matching_results,
    draw_faces_bounding_boxes,
    plot_embedding_comparison,
    plot_radar_comparison
)

def run_all_tests():
    print("==================================================")
    print("STARTING IVA LAB 2 AUTOMATED QUALITY ASSURANCE")
    print("==================================================")
    
    catalog = get_sample_images_catalog()
    
    # 1. Test Sample Images Catalog
    print("\n[TEST 1] Testing Sample Images Catalog...")
    for cat_name, items in catalog.items():
        print(f"  Checking category '{cat_name}' with {len(items)} items...")
        for item_name, details in items.items():
            for key in ["main", "template", "path", "img1", "img2"]:
                if key in details:
                    path = details[key]
                    assert os.path.exists(path), f"Missing sample image at {path}"
    print("  [PASS] All sample images exist and are accessible.")

    # 2. Test Template Matching
    print("\n[TEST 2] Testing Template Matching with all 6 methods...")
    tm_sample = catalog["Template Matching"]["Circuit Board & Microchip"]
    main_img, _, _, _ = load_image_from_path(tm_sample["main"])
    tmpl_img, _, _, _ = load_image_from_path(tm_sample["template"])
    
    for method_key in MATCH_METHODS.keys():
        res = perform_template_matching(main_img, tmpl_img, method_key=method_key)
        assert res["success"] is True, f"Failed matching for method {method_key}: {res.get('error')}"
        assert res["best_location"] is not None
        assert res["heatmap"] is not None
        print(f"  [PASS] Method {method_key:<18} -> Score: {res['score']:.4f} at {res['best_location']}")
        
    # Test Template Matching Edge Case: Template larger than image
    oversized_tmpl = np.zeros((600, 700, 3), dtype=np.uint8)
    res_err = perform_template_matching(main_img, oversized_tmpl)
    assert res_err["success"] is False, "Oversized template check failed to return error!"
    print("  [PASS] Oversized template error handling passed.")

    # 3. Test Viola-Jones Face Detection
    print("\n[TEST 3] Testing Viola-Jones Face Detection...")
    vj_sample = catalog["Viola-Jones Faces"]["Single Subject (Person A)"]
    face_img, _, _, _ = load_image_from_path(vj_sample["path"])
    vj_res = detect_faces_viola_jones(face_img, scale_factor=1.1, min_neighbors=3, min_size=(30, 30), detect_eyes=True)
    assert vj_res["success"] is True, f"Viola-Jones failed: {vj_res.get('error')}"
    assert vj_res["face_count"] >= 1, f"Expected at least 1 face, found {vj_res['face_count']}"
    print(f"  [PASS] Detected {vj_res['face_count']} face(s) in {vj_sample['path']}. Integral image shape: {vj_res['integral_preview'].shape}")

    # 4. Test DeepFace Module (Analysis & Verification)
    print("\n[TEST 4] Testing DeepFace Module...")
    df_res_a = analyze_single_face(face_img)
    assert df_res_a["success"] is True, f"DeepFace analysis failed: {df_res_a.get('error')}"
    print(f"  [PASS] Single face analysis -> Age: {df_res_a['age']}, Gender: {df_res_a['dominant_gender']}, Emotion: {df_res_a['dominant_emotion']}")
    
    pair_sample = catalog["DeepFace / FaceNet Pairs"]["Same Person Pair (Person A - Smile vs Serious)"]
    img1, _, _, _ = load_image_from_path(pair_sample["img1"])
    img2, _, _, _ = load_image_from_path(pair_sample["img2"])
    
    verif_res = verify_face_pair(img1, img2)
    assert verif_res["success"] is True, f"DeepFace verification failed: {verif_res.get('error')}"
    print(f"  [PASS] Face verification -> Verified: {verif_res['verified']}, Distance: {verif_res['distance']}, Threshold: {verif_res['threshold']}")

    # 5. Test FaceNet Module
    print("\n[TEST 5] Testing FaceNet Embeddings...")
    emb_a = extract_facenet_embedding(img1)
    emb_b = extract_facenet_embedding(img2)
    assert emb_a["success"] is True and emb_b["success"] is True, "Embedding extraction failed"
    print(f"  [PASS] Extracted {emb_a['dim']}-D embedding vector. L2 norm: {np.linalg.norm(emb_a['embedding']):.4f}")
    
    cmp_emb = compare_face_embeddings(emb_a["embedding"], emb_b["embedding"])
    print(f"  [PASS] Embedding distance comparison -> Euclidean: {cmp_emb['euclidean_distance']}, Cosine Sim: {cmp_emb['cosine_similarity']}")

    # 6. Test Visualizations
    print("\n[TEST 6] Testing Matplotlib & UI Visualizations...")
    fig1 = plot_template_matching_results(main_img, tmpl_img, res["heatmap"], res["best_location"], res["bottom_right"], res["score"], "TM_CCOEFF_NORMED")
    assert fig1 is not None
    
    fig2 = plot_embedding_comparison(emb_a["embedding"], emb_b["embedding"], cmp_emb["euclidean_distance"], cmp_emb["cosine_similarity"], cmp_emb["is_same_identity"])
    assert fig2 is not None
    
    fig3 = plot_radar_comparison()
    assert fig3 is not None
    print("  [PASS] All Matplotlib figures generated successfully without errors.")

    print("\n==================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
