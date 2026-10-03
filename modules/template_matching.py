"""
Module 1: Template Matching for IVA Assignment 2.
Implements OpenCV Template Matching with 6 mathematical methods, multi-match thresholding,
and interactive visualization.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, List, Optional

# OpenCV Matching Methods dictionary
MATCH_METHODS = {
    "TM_CCOEFF": {
        "cv_code": cv2.TM_CCOEFF,
        "name": "Correlation Coefficient",
        "type": "max",
        "formula": r"R(x,y) = \sum_{x',y'} \left( T'(x',y') \cdot I'(x+x', y+y') \right)",
        "summary": "Measures relative correlation after subtracting mean intensity. Good for moderate lighting differences."
    },
    "TM_CCOEFF_NORMED": {
        "cv_code": cv2.TM_CCOEFF_NORMED,
        "name": "Normalized Correlation Coefficient (Recommended)",
        "type": "max",
        "formula": r"R(x,y) = \frac{\sum_{x',y'} (T'(x',y') \cdot I'(x+x', y+y'))}{\sqrt{\sum_{x',y'} T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x', y+y')^2}}",
        "summary": "Standard normalized coefficient [-1.0, 1.0]. 1.0 = perfect match, -1.0 = inverse. Highly robust."
    },
    "TM_CCORR": {
        "cv_code": cv2.TM_CCORR,
        "name": "Cross-Correlation",
        "type": "max",
        "formula": r"R(x,y) = \sum_{x',y'} \left( T(x',y') \cdot I(x+x', y+y') \right)",
        "summary": "Multiplies raw pixel values. Bright image regions yield large responses even without true structural match."
    },
    "TM_CCORR_NORMED": {
        "cv_code": cv2.TM_CCORR_NORMED,
        "name": "Normalized Cross-Correlation",
        "type": "max",
        "formula": r"R(x,y) = \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x', y+y'))}{\sqrt{\sum_{x',y'} T(x',y')^2 \cdot \sum_{x',y'} I(x+x', y+y')^2}}",
        "summary": "Normalized [0.0, 1.0] cross-correlation. Reduces sensitivity to raw intensity variations."
    },
    "TM_SQDIFF": {
        "cv_code": cv2.TM_SQDIFF,
        "name": "Squared Difference",
        "type": "min",
        "formula": r"R(x,y) = \sum_{x',y'} \left( T(x',y') - I(x+x', y+y') \right)^2",
        "summary": "Sum of squared differences. 0 = exact match. Global minimum indicates location."
    },
    "TM_SQDIFF_NORMED": {
        "cv_code": cv2.TM_SQDIFF_NORMED,
        "name": "Normalized Squared Difference",
        "type": "min",
        "formula": r"R(x,y) = \frac{\sum_{x',y'} (T(x',y') - I(x+x', y+y'))^2}{\sqrt{\sum_{x',y'} T(x',y')^2 \cdot \sum_{x',y'} I(x+x', y+y')^2}}",
        "summary": "Normalized [0.0, 1.0] squared difference. 0.0 = perfect match, 1.0 = total mismatch."
    }
}

def perform_template_matching(
    main_image_rgb: np.ndarray,
    template_image_rgb: np.ndarray,
    method_key: str = "TM_CCOEFF_NORMED",
    match_all_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes Template Matching using OpenCV.
    
    Args:
        main_image_rgb: Main scene image (RGB)
        template_image_rgb: Target template image (RGB)
        method_key: Selected method key from MATCH_METHODS
        match_all_threshold: Optional threshold for multi-instance detection
        
    Returns:
        Dictionary with match details, coordinates, score, and heatmap.
    """
    if main_image_rgb is None or template_image_rgb is None:
        return {"success": False, "error": "Main image or template is missing."}
    
    # Check dimensions
    img_h, img_w = main_image_rgb.shape[:2]
    tmpl_h, tmpl_w = template_image_rgb.shape[:2]
    
    if tmpl_h > img_h or tmpl_w > img_w:
        return {
            "success": False,
            "error": f"Template dimensions ({tmpl_w}x{tmpl_h}) cannot exceed main image dimensions ({img_w}x{img_h}). Please resize or select a smaller template."
        }
    
    if method_key not in MATCH_METHODS:
        method_key = "TM_CCOEFF_NORMED"
        
    method_info = MATCH_METHODS[method_key]
    cv_method = method_info["cv_code"]
    is_min_type = (method_info["type"] == "min")
    
    # Convert both to grayscale for robust single-channel correlation matching
    img_gray = cv2.cvtColor(main_image_rgb, cv2.COLOR_RGB2GRAY)
    tmpl_gray = cv2.cvtColor(template_image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute correlation map
    res_map = cv2.matchTemplate(img_gray, tmpl_gray, cv_method)
    
    # Find global min/max
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_map)
    
    if is_min_type:
        best_loc = min_loc
        best_score = float(min_val)
    else:
        best_loc = max_loc
        best_score = float(max_val)
        
    top_left = best_loc
    bottom_right = (top_left[0] + tmpl_w, top_left[1] + tmpl_h)
    
    # Find multiple matches if threshold is specified
    multiple_matches = []
    if match_all_threshold is not None:
        if is_min_type:
            # For SQDIFF methods, lower is better
            locs = np.where(res_map <= match_all_threshold)
        else:
            # For CCOEFF/CCORR, higher is better
            locs = np.where(res_map >= match_all_threshold)
            
        # Extract matching points with Non-Maximum Suppression (simple distance filter)
        points = list(zip(*locs[::-1])) # (x, y)
        filtered_points = []
        for pt in points:
            # Check if too close to already selected points
            too_close = False
            for fpt in filtered_points:
                if abs(pt[0] - fpt[0]) < (tmpl_w // 2) and abs(pt[1] - fpt[1]) < (tmpl_h // 2):
                    too_close = True
                    break
            if not too_close:
                filtered_points.append(pt)
                multiple_matches.append({
                    "top_left": pt,
                    "bottom_right": (pt[0] + tmpl_w, pt[1] + tmpl_h),
                    "score": float(res_map[pt[1], pt[0]])
                })
    
    # Render bounding box on image
    result_img = main_image_rgb.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(
        result_img,
        f"Best ({best_score:.3f})",
        (top_left[0], max(18, top_left[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )
    
    return {
        "success": True,
        "method_key": method_key,
        "method_info": method_info,
        "heatmap": res_map,
        "best_location": top_left,
        "bottom_right": bottom_right,
        "template_size": (tmpl_w, tmpl_h),
        "score": best_score,
        "min_val": float(min_val),
        "max_val": float(max_val),
        "result_image": result_img,
        "multiple_matches": multiple_matches
    }
