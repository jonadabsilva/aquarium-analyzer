"""
Aquarium Water Test Strip Analyzer - Final Robust Version
For SJ Wave 10-in-1 and Ammonia test strips

Features:
- Automatic color detection from strip photos
- Manual color picker fallback
- Comprehensive error handling
- Works on Streamlit Cloud (no OpenCV dependency)
- Saltwater/Freshwater support with specific gravity
- Detailed recommendations and nitrogen cycle assessment
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import traceback

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🐠 Aquarium Analyzer",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS - Mobile-optimized, clean interface
# =============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        border-left: 4px solid #1976D2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-ok { 
        color: #2E7D32; 
        font-weight: bold; 
        background: #E8F5E9;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .status-warning { 
        color: #F57C00; 
        font-weight: bold;
        background: #FFF3E0;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .status-danger { 
        color: #C62828; 
        font-weight: bold;
        background: #FFEBEE;
        padding: 4px 12px;
        border-radius: 20px;
    }
    
    .color-swatch {
        display: inline-block;
        width: 28px;
        height: 28px;
        border-radius: 6px;
        border: 2px solid #333;
        vertical-align: middle;
        margin-right: 12px;
    }
    
    .instruction-box {
        background: rgba(25, 118, 210, 0.1);
        border-left: 4px solid #1976D2;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
    }
    
    .priority-action {
        background: #FFEBEE;
        border-left: 4px solid #C62828;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of analyzing a single parameter"""
    name: str
    display_name: str
    value: float
    unit: str
    status: str  # 'ok', 'warning', 'danger'
    detected_color: Tuple[int, int, int]
    confidence: float
    recommendations: List[str]


# =============================================================================
# REFERENCE DATA - Calibrated for SJ Wave Test Strips
# =============================================================================

# RGB color references for each parameter value
# Format: {value: (R, G, B)}

REFERENCE_COLORS = {
    "Iron": {
        0: (245, 240, 225),
        5: (255, 210, 190),
        10: (255, 185, 160),
        25: (255, 150, 120),
        50: (255, 120, 90),
        100: (220, 90, 70)
    },
    "Copper": {
        0: (250, 248, 245),
        10: (240, 235, 240),
        30: (200, 180, 200),
        100: (160, 130, 180),
        200: (130, 100, 160),
        300: (100, 75, 140)
    },
    "Nitrate": {
        0: (255, 255, 210),
        10: (250, 240, 150),
        25: (230, 200, 80),
        50: (200, 170, 60),
        100: (170, 130, 50),
        250: (150, 90, 50)
    },
    "Nitrite": {
        0: (255, 245, 250),
        1: (250, 210, 230),
        5: (240, 160, 200),
        10: (210, 110, 170)
    },
    "Chlorine": {
        0: (255, 252, 252),
        0.8: (240, 220, 245),
        1.5: (220, 180, 230),
        3: (180, 140, 210)
    },
    "Hardness": {
        0: (170, 210, 195),
        25: (155, 195, 175),
        75: (145, 165, 180),
        150: (150, 145, 185),
        300: (160, 135, 200)
    },
    "Alkalinity": {
        0: (150, 210, 190),
        40: (130, 190, 165),
        80: (100, 185, 145),
        120: (75, 180, 120),
        180: (55, 160, 100),
        300: (45, 140, 85)
    },
    "Carbonate": {
        0: (150, 210, 190),
        40: (130, 190, 165),
        80: (100, 185, 145),
        120: (75, 180, 120),
        180: (55, 160, 100),
        300: (45, 140, 85)
    },
    "pH": {
        6.4: (255, 245, 100),
        6.8: (240, 225, 75),
        7.2: (200, 205, 55),
        7.6: (160, 205, 75),
        8.0: (100, 200, 95),
        8.4: (55, 180, 140),
        9.0: (190, 100, 180)
    },
}

AMMONIA_COLORS = {
    0: (230, 230, 145),
    0.25: (200, 225, 145),
    0.5: (170, 215, 155),
    1: (140, 205, 165),
    3: (100, 185, 165),
    6: (80, 160, 150)
}

# Parameter configuration: thresholds and ideal ranges
PARAM_CONFIG = {
    "Iron": {
        "unit": "ppm",
        "display": "Iron (Fe)",
        "saltwater": {"ideal": (0, 0), "ok": (0, 5), "warn": 10, "danger": 25},
        "freshwater": {"ideal": (0, 0), "ok": (0, 5), "warn": 10, "danger": 25},
    },
    "Copper": {
        "unit": "ppm",
        "display": "Copper (Cu)",
        "saltwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 1, "danger": 10},
        "freshwater": {"ideal": (0, 0), "ok": (0, 10), "warn": 30, "danger": 100},
    },
    "Nitrate": {
        "unit": "ppm",
        "display": "Nitrate (NO₃)",
        "saltwater": {"ideal": (0, 10), "ok": (0, 25), "warn": 50, "danger": 80},
        "freshwater": {"ideal": (0, 20), "ok": (0, 40), "warn": 60, "danger": 100},
    },
    "Nitrite": {
        "unit": "ppm",
        "display": "Nitrite (NO₂)",
        "saltwater": {"ideal": (0, 0), "ok": (0, 0.25), "warn": 0.5, "danger": 1},
        "freshwater": {"ideal": (0, 0), "ok": (0, 0.5), "warn": 1, "danger": 5},
    },
    "Chlorine": {
        "unit": "ppm",
        "display": "Chlorine (Cl₂)",
        "saltwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 0.5, "danger": 1},
        "freshwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 0.5, "danger": 1},
    },
    "Hardness": {
        "unit": "ppm",
        "display": "Hardness (GH)",
        "saltwater": {"ideal": (150, 300), "ok": (75, 300), "warn_low": 50, "warn_high": None},
        "freshwater": {"ideal": (75, 150), "ok": (50, 200), "warn_low": 25, "warn_high": 250},
    },
    "Alkalinity": {
        "unit": "ppm",
        "display": "Alkalinity (KH)",
        "saltwater": {"ideal": (120, 180), "ok": (100, 200), "warn_low": 80, "warn_high": None},
        "freshwater": {"ideal": (80, 120), "ok": (40, 180), "warn_low": 40, "warn_high": 200},
    },
    "Carbonate": {
        "unit": "ppm",
        "display": "Carbonate (KH)",
        "saltwater": {"ideal": (120, 180), "ok": (100, 200), "warn_low": 80, "warn_high": None},
        "freshwater": {"ideal": (80, 120), "ok": (40, 180), "warn_low": 40, "warn_high": 200},
    },
    "pH": {
        "unit": "",
        "display": "pH",
        "saltwater": {"ideal": (8.0, 8.4), "ok": (7.8, 8.5), "warn_low": 7.6, "warn_high": 8.6},
        "freshwater": {"ideal": (6.8, 7.6), "ok": (6.4, 8.0), "warn_low": 6.0, "warn_high": 8.2},
    },
    "Ammonia": {
        "unit": "ppm",
        "display": "Ammonia (NH₃/NH₄⁺)",
        "saltwater": {"ideal": (0, 0), "ok": (0, 0.1), "warn": 0.25, "danger": 0.5},
        "freshwater": {"ideal": (0, 0), "ok": (0, 0.25), "warn": 0.5, "danger": 1},
    },
}

# Row order on the 10-in-1 strip (top to bottom)
STRIP_PARAMS_ORDER = [
    "Iron", "Copper", "Nitrate", "Nitrite", "Chlorine",
    "Hardness", "Alkalinity", "Carbonate", "pH"
]


# =============================================================================
# COLOR ANALYSIS FUNCTIONS
# =============================================================================

def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptually-weighted color distance.
    Uses weights that account for human color perception.
    """
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    
    # Weighted Euclidean distance (green channel weighted more)
    rmean = (r1 + r2) / 2
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    
    return np.sqrt(
        (2 + rmean/256) * dr**2 +
        4 * dg**2 +
        (2 + (255 - rmean)/256) * db**2
    )


def match_color_to_value(
    detected_rgb: Tuple[int, int, int],
    color_chart: Dict[float, Tuple[int, int, int]]
) -> Tuple[float, float]:
    """
    Find the best matching value from a color chart with interpolation.
    Returns (interpolated_value, confidence_percentage)
    """
    if not color_chart:
        return 0, 0
    
    values = sorted(color_chart.keys())
    distances = [(v, color_distance(detected_rgb, color_chart[v])) for v in values]
    distances.sort(key=lambda x: x[1])
    
    v1, d1 = distances[0]
    
    # Calculate confidence (lower distance = higher confidence)
    # Max possible distance is ~765 for RGB
    confidence = max(20, min(100, 100 - d1 * 0.15))
    
    # Interpolate between two closest matches for better accuracy
    if len(distances) >= 2 and d1 < 200:
        v2, d2 = distances[1]
        if d1 + d2 > 0:
            weight = d2 / (d1 + d2)
            value = v1 * weight + v2 * (1 - weight)
            return round(value, 2), confidence
    
    return v1, confidence


def get_status(param_name: str, value: float, water_type: str) -> str:
    """Determine status (ok/warning/danger) for a parameter value"""
    config = PARAM_CONFIG.get(param_name, {})
    ranges = config.get(water_type.lower(), config.get("freshwater", {}))
    
    if not ranges:
        return "ok"
    
    ideal = ranges.get("ideal", (None, None))
    ok = ranges.get("ok", (None, None))
    warn = ranges.get("warn")
    danger = ranges.get("danger")
    warn_low = ranges.get("warn_low")
    warn_high = ranges.get("warn_high")
    
    # Check danger threshold
    if danger is not None and value >= danger:
        return "danger"
    
    # Check warning threshold
    if warn is not None and value >= warn:
        return "warning"
    
    # Check low/high warnings (for parameters like pH, hardness)
    if warn_low is not None and value < warn_low:
        return "warning"
    if warn_high is not None and value > warn_high:
        return "warning"
    
    # Check if in OK range
    if ok and ok[0] is not None:
        if ok[0] <= value <= ok[1]:
            return "ok"
    
    # Check if in ideal range
    if ideal and ideal[0] is not None:
        if ideal[0] <= value <= ideal[1]:
            return "ok"
        elif value < ideal[0]:
            return "warning"  # Low
        else:
            return "warning"  # High
    
    return "ok"


def get_recommendations(param_name: str, value: float, status: str, water_type: str) -> List[str]:
    """Generate actionable recommendations based on parameter and value"""
    recs = []
    wt = water_type.lower()
    
    if param_name == "Ammonia":
        if status == "danger":
            recs = [
                "🚨 EMERGENCY: 50% water change immediately!",
                "Add ammonia detoxifier (Prime, AmGuard)",
                "Stop feeding for 24-48 hours",
                "Check filter - clean in tank water only",
                "Add beneficial bacteria supplement"
            ]
        elif status == "warning":
            recs = [
                "Perform 25-30% water change",
                "Reduce feeding",
                "Check for dead organisms or decaying matter",
                "Test again in 24 hours"
            ]
    
    elif param_name == "Nitrite":
        if status == "danger":
            recs = [
                "🚨 URGENT: 50% water change needed",
                "Add Prime to detoxify nitrite",
                "Stop or heavily reduce feeding",
                "Add beneficial bacteria"
            ]
        elif status == "warning":
            recs = [
                "25-30% water change recommended",
                "Tank may still be cycling",
                "Add bacteria supplement",
                "Reduce feeding frequency"
            ]
    
    elif param_name == "Nitrate":
        if status == "danger":
            recs = [
                "Large water change needed (40-50%)",
                "Check for hidden detritus",
                "Reduce feeding amount",
                "Consider more frequent water changes"
            ]
        elif status == "warning":
            recs = [
                "Schedule 25-30% water change",
                "Add macroalgae/refugium" if wt == "saltwater" else "Add live plants",
                "Review feeding schedule"
            ]
    
    elif param_name == "Copper":
        if value > 0:
            recs = [
                "🚨 Copper detected - toxic to invertebrates!",
                "Run activated carbon immediately",
                "Identify copper source (medications, pipes)",
                "Do NOT add invertebrates until copper is 0"
            ]
    
    elif param_name == "Chlorine":
        if value > 0:
            recs = [
                "🚨 Chlorine detected!",
                "Add dechlorinator immediately",
                "Always treat tap water before adding"
            ]
    
    elif param_name == "pH":
        if wt == "saltwater":
            if value < 8.0:
                recs = [
                    "pH low for saltwater",
                    "Check and boost alkalinity",
                    "Increase surface agitation",
                    "Consider pH buffer"
                ]
            elif value > 8.5:
                recs = [
                    "pH elevated",
                    "Check dosing equipment",
                    "Verify lighting schedule"
                ]
        else:
            if value < 6.5:
                recs = [
                    "pH low - stressful for most fish",
                    "Add crushed coral or limestone",
                    "Check KH levels"
                ]
            elif value > 8.0:
                recs = [
                    "pH elevated for freshwater",
                    "Add driftwood or peat",
                    "Use RO water for changes"
                ]
    
    elif param_name == "Alkalinity" or param_name == "Carbonate":
        if wt == "saltwater" and value < 100:
            recs = [
                "Alkalinity too low for reef keeping",
                "Add alkalinity supplement",
                "Check dosing system"
            ]
    
    elif param_name == "Hardness":
        if wt == "freshwater":
            if value < 50:
                recs = ["Consider adding mineral supplement"]
            elif value > 250:
                recs = ["Use RO water to reduce hardness"]
    
    return recs


# =============================================================================
# IMAGE PROCESSING FUNCTIONS (No OpenCV - Pillow/NumPy only)
# =============================================================================

def safe_load_image(uploaded_file) -> Optional[np.ndarray]:
    """Safely load and validate an uploaded image"""
    try:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate size
        if image.size[0] < 50 or image.size[1] < 50:
            return None
        
        # Limit max size for performance
        max_dim = 2000
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return np.array(image)
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def enhance_image(img_array: np.ndarray) -> np.ndarray:
    """Enhance image for better color detection"""
    try:
        image = Image.fromarray(img_array)
        
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Slight sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return np.array(image)
    except:
        return img_array


def find_card_bounds(img_array: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the white color chart card boundaries in the image.
    Returns (left, right, top, bottom)
    """
    try:
        height, width = img_array.shape[:2]
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Find bright regions (the white card)
        threshold = 150
        bright_mask = gray > threshold
        
        # Find bounds
        rows_with_bright = np.where(np.any(bright_mask, axis=1))[0]
        cols_with_bright = np.where(np.any(bright_mask, axis=0))[0]
        
        if len(rows_with_bright) > 10 and len(cols_with_bright) > 10:
            # Use percentiles to exclude outliers
            top = int(np.percentile(rows_with_bright, 2))
            bottom = int(np.percentile(rows_with_bright, 98))
            left = int(np.percentile(cols_with_bright, 2))
            right = int(np.percentile(cols_with_bright, 98))
            
            # Validate bounds
            if right - left > 100 and bottom - top > 100:
                return left, right, top, bottom
        
        # Fallback: use image margins
        margin = 0.05
        return (
            int(width * margin),
            int(width * (1 - margin)),
            int(height * margin),
            int(height * (1 - margin))
        )
    
    except Exception:
        height, width = img_array.shape[:2]
        return 0, width, 0, height


def sample_region_color(
    img_array: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[int, int, int]:
    """Sample the median color from a rectangular region"""
    try:
        # Clamp coordinates
        h, w = img_array.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return (128, 128, 128)
        
        region = img_array[y1:y2, x1:x2]
        
        if region.size == 0:
            return (128, 128, 128)
        
        # Use median for robustness against outliers
        r = int(np.median(region[:, :, 0]))
        g = int(np.median(region[:, :, 1]))
        b = int(np.median(region[:, :, 2]))
        
        return (r, g, b)
    
    except Exception:
        return (128, 128, 128)


def analyze_10in1_strip_auto(image: np.ndarray) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Automatically analyze a 10-in-1 test strip placed on its color chart.
    Returns (results_list, error_message)
    """
    try:
        height, width = image.shape[:2]
        
        if height < 100 or width < 100:
            return None, "Image too small"
        
        # Enhance image
        enhanced = enhance_image(image)
        
        # Find card boundaries
        card_left, card_right, card_top, card_bottom = find_card_bounds(enhanced)
        card_width = card_right - card_left
        card_height = card_bottom - card_top
        
        if card_width < 100 or card_height < 100:
            return None, "Could not detect color chart card"
        
        # Chart layout (SJ Wave structure):
        # - Strip placement area: leftmost 1-5% of card
        # - Rows: 10 rows after ~6% header, before ~4% footer
        
        rows_start = card_top + int(card_height * 0.06)
        rows_end = card_bottom - int(card_height * 0.04)
        row_height = (rows_end - rows_start) / 10
        
        results = []
        
        for i, param in enumerate(STRIP_PARAMS_ORDER):
            # Calculate row center
            y_center = rows_start + int((i + 0.5) * row_height)
            y1 = max(0, y_center - int(row_height * 0.25))
            y2 = min(height, y_center + int(row_height * 0.25))
            
            # Sample from multiple x positions, take most saturated
            best_color = None
            best_saturation = -1
            
            # Try different x ranges (strip placement varies)
            x_ranges = [
                (card_left + int(card_width * 0.01), card_left + int(card_width * 0.04)),
                (card_left + int(card_width * 0.02), card_left + int(card_width * 0.06)),
                (card_left + int(card_width * 0.00), card_left + int(card_width * 0.05)),
            ]
            
            for x1, x2 in x_ranges:
                color = sample_region_color(enhanced, x1, y1, x2, y2)
                
                # Calculate saturation
                max_c = max(color)
                min_c = min(color)
                sat = (max_c - min_c) / max_c if max_c > 0 else 0
                
                if sat > best_saturation:
                    best_saturation = sat
                    best_color = color
            
            if best_color is None:
                best_color = (200, 200, 200)
            
            # Match to reference colors
            if param in REFERENCE_COLORS:
                value, confidence = match_color_to_value(best_color, REFERENCE_COLORS[param])
            else:
                value, confidence = 0, 50
            
            results.append({
                "name": param,
                "color": best_color,
                "value": value,
                "confidence": confidence
            })
        
        return results, None
    
    except Exception as e:
        return None, f"Analysis error: {str(e)}"


def analyze_ammonia_strip_auto(image: np.ndarray) -> Dict:
    """
    Automatically analyze an ammonia test strip.
    Looks for yellow-green-teal colored pad.
    """
    try:
        height, width = image.shape[:2]
        enhanced = enhance_image(image)
        
        # Convert to float for calculations
        r = enhanced[:, :, 0].astype(float)
        g = enhanced[:, :, 1].astype(float)
        b = enhanced[:, :, 2].astype(float)
        
        # Score pixels for ammonia colors (yellow to teal-green)
        # High green, moderate red (for yellow) or low red (for teal), low-moderate blue
        ammonia_score = g - 0.4 * np.abs(r - g) - 0.3 * b
        
        # Exclude very dark or very bright pixels
        brightness = (r + g + b) / 3
        ammonia_score = np.where((brightness > 60) & (brightness < 240), ammonia_score, -1000)
        
        # Simple smoothing using mean of region
        pad_size = max(10, min(height, width) // 20)
        best_score = -1000
        best_x, best_y = width // 2, height // 2
        
        # Sample grid
        step = max(5, pad_size // 2)
        for y in range(pad_size, height - pad_size, step):
            for x in range(pad_size, width - pad_size, step):
                region_score = np.mean(ammonia_score[y-pad_size//2:y+pad_size//2,
                                                      x-pad_size//2:x+pad_size//2])
                if region_score > best_score:
                    best_score = region_score
                    best_x, best_y = x, y
        
        # Sample color at best location
        color = sample_region_color(
            enhanced,
            best_x - pad_size, best_y - pad_size,
            best_x + pad_size, best_y + pad_size
        )
        
        value, confidence = match_color_to_value(color, AMMONIA_COLORS)
        
        return {
            "name": "Ammonia",
            "color": color,
            "value": value,
            "confidence": confidence
        }
    
    except Exception as e:
        return {
            "name": "Ammonia",
            "color": (200, 200, 150),
            "value": 0,
            "confidence": 30
        }


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    results: List[AnalysisResult],
    water_type: str,
    specific_gravity: Optional[float] = None
) -> str:
    """Generate downloadable text report"""
    lines = [
        "=" * 50,
        "AQUARIUM WATER QUALITY REPORT",
        "=" * 50,
        "",
        f"Water Type: {water_type}",
    ]
    
    if specific_gravity:
        lines.append(f"Specific Gravity: {specific_gravity:.3f}")
    
    lines.extend(["", "-" * 50, "RESULTS", "-" * 50, ""])
    
    for r in results:
        icon = "✅" if r.status == "ok" else ("⚠️" if r.status == "warning" else "🚨")
        lines.append(f"{icon} {r.display_name}: {r.value:.2f} {r.unit}")
        lines.append(f"   Status: {r.status.upper()}")
        lines.append(f"   Confidence: {r.confidence:.0f}%")
        
        if r.recommendations:
            lines.append("   Recommendations:")
            for rec in r.recommendations:
                lines.append(f"     • {rec}")
        lines.append("")
    
    # Nitrogen cycle assessment
    nh3 = next((r.value for r in results if r.name == "Ammonia"), None)
    no2 = next((r.value for r in results if r.name == "Nitrite"), None)
    no3 = next((r.value for r in results if r.name == "Nitrate"), None)
    
    if nh3 is not None or no2 is not None:
        lines.extend(["-" * 50, "NITROGEN CYCLE ASSESSMENT", "-" * 50, ""])
        
        nh3 = nh3 or 0
        no2 = no2 or 0
        no3 = no3 or 0
        
        if nh3 == 0 and no2 == 0 and no3 > 0:
            lines.append("✅ Fully cycled - beneficial bacteria working properly")
        elif nh3 > 0 and no2 == 0 and no3 == 0:
            lines.append("⚠️ Early cycling or ammonia source issue")
        elif nh3 > 0 and no2 > 0:
            lines.append("🚨 Active cycling - frequent water changes needed")
        elif nh3 == 0 and no2 > 0:
            lines.append("⚠️ Mid-cycle - nitrite-converting bacteria establishing")
        else:
            lines.append("ℹ️ Cycle status unclear - continue monitoring")
    
    lines.extend(["", "-" * 50, "END OF REPORT", "-" * 50])
    
    return "\n".join(lines)


# =============================================================================
# STREAMLIT UI
# =============================================================================

def render_result_card(result: AnalysisResult):
    """Render a single parameter result"""
    icon = "✅" if result.status == "ok" else ("⚠️" if result.status == "warning" else "🚨")
    status_class = f"status-{result.status}"
    rgb = result.detected_color
    
    st.markdown(f"""
    <div class="result-card">
        <span class="color-swatch" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"></span>
        <strong>{result.display_name}</strong>: {result.value:.2f} {result.unit}
        <span class="{status_class}">{icon} {result.status.upper()}</span>
        <small style="color:#666; margin-left:10px;">({result.confidence:.0f}% conf)</small>
    </div>
    """, unsafe_allow_html=True)
    
    if result.recommendations:
        with st.expander("📋 Recommendations"):
            for rec in result.recommendations:
                st.markdown(f"• {rec}")


def main():
    st.title("🐠 Aquarium Water Analyzer")
    st.markdown("*Automatic analysis for SJ Wave test strips*")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'manual_colors' not in st.session_state:
        st.session_state.manual_colors = {}
    
    # ===== CONFIGURATION =====
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        water_type = st.selectbox("🌊 Water Type", ["Saltwater", "Freshwater"])
    
    with col2:
        specific_gravity = None
        if water_type == "Saltwater":
            sg_input = st.number_input(
                "📏 Specific Gravity",
                min_value=1.015,
                max_value=1.035,
                value=1.025,
                step=0.001,
                format="%.3f"
            )
            specific_gravity = sg_input
    
    # ===== INSTRUCTIONS =====
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Best Results:**
        1. Place test strip on the color chart (gray area on left side)
        2. Take photo with good lighting (natural daylight best)
        3. Keep camera parallel to chart
        4. Wait full reaction time (15s for most, 60s for nitrate/nitrite)
        
        **Options:**
        - **Auto Mode**: Upload photo, app detects colors automatically
        - **Manual Mode**: Use color picker if auto-detection is inaccurate
        """)
    
    # ===== TABS =====
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 10-in-1 Strip", "🧪 Ammonia Strip", "🎨 Manual Entry"])
    
    all_results = []
    
    # ----- 10-in-1 Auto Tab -----
    with tab1:
        st.markdown("### Upload 10-in-1 Strip Photo")
        
        uploaded = st.file_uploader(
            "Photo of strip on color chart",
            type=['jpg', 'jpeg', 'png'],
            key="upload_10in1"
        )
        
        if uploaded:
            img_array = safe_load_image(uploaded)
            
            if img_array is not None:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img_array, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    with st.spinner("🔍 Analyzing..."):
                        results, error = analyze_10in1_strip_auto(img_array)
                    
                    if error:
                        st.error(error)
                        st.info("Try the Manual Entry tab for better accuracy.")
                    elif results:
                        st.markdown("### Detected Values")
                        
                        for r in results:
                            config = PARAM_CONFIG.get(r["name"], {})
                            display_name = config.get("display", r["name"])
                            unit = config.get("unit", "")
                            status = get_status(r["name"], r["value"], water_type)
                            recs = get_recommendations(r["name"], r["value"], status, water_type)
                            
                            result = AnalysisResult(
                                name=r["name"],
                                display_name=display_name,
                                value=r["value"],
                                unit=unit,
                                status=status,
                                detected_color=r["color"],
                                confidence=r["confidence"],
                                recommendations=recs
                            )
                            all_results.append(result)
                            render_result_card(result)
            else:
                st.error("Could not load image. Please try a different file.")
    
    # ----- Ammonia Tab -----
    with tab2:
        st.markdown("### Upload Ammonia Strip Photo")
        st.info("💡 The ammonia pad is the colored square at the end of the strip")
        
        uploaded_nh3 = st.file_uploader(
            "Photo of ammonia strip",
            type=['jpg', 'jpeg', 'png'],
            key="upload_ammonia"
        )
        
        if uploaded_nh3:
            img_array = safe_load_image(uploaded_nh3)
            
            if img_array is not None:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img_array, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    with st.spinner("🔍 Analyzing..."):
                        r = analyze_ammonia_strip_auto(img_array)
                    
                    config = PARAM_CONFIG.get("Ammonia", {})
                    status = get_status("Ammonia", r["value"], water_type)
                    recs = get_recommendations("Ammonia", r["value"], status, water_type)
                    
                    result = AnalysisResult(
                        name="Ammonia",
                        display_name=config.get("display", "Ammonia"),
                        value=r["value"],
                        unit=config.get("unit", "ppm"),
                        status=status,
                        detected_color=r["color"],
                        confidence=r["confidence"],
                        recommendations=recs
                    )
                    all_results.append(result)
                    
                    st.markdown("### Detected Value")
                    render_result_card(result)
            else:
                st.error("Could not load image.")
    
    # ----- Manual Entry Tab -----
    with tab3:
        st.markdown("### Manual Color Entry")
        st.markdown("Use this if automatic detection is inaccurate.")
        
        manual_params = ["Ammonia"] + STRIP_PARAMS_ORDER
        selected_param = st.selectbox(
            "Select parameter:",
            manual_params,
            format_func=lambda x: PARAM_CONFIG.get(x, {}).get("display", x)
        )
        
        color_picker = st.color_picker(
            f"Pick color for {PARAM_CONFIG.get(selected_param, {}).get('display', selected_param)}",
            value="#FFFFFF"
        )
        
        if st.button(f"Analyze {selected_param}", type="primary"):
            rgb = hex_to_rgb(color_picker)
            
            # Get reference colors
            if selected_param == "Ammonia":
                color_chart = AMMONIA_COLORS
            else:
                color_chart = REFERENCE_COLORS.get(selected_param, {})
            
            value, confidence = match_color_to_value(rgb, color_chart)
            config = PARAM_CONFIG.get(selected_param, {})
            status = get_status(selected_param, value, water_type)
            recs = get_recommendations(selected_param, value, status, water_type)
            
            result = AnalysisResult(
                name=selected_param,
                display_name=config.get("display", selected_param),
                value=value,
                unit=config.get("unit", ""),
                status=status,
                detected_color=rgb,
                confidence=confidence,
                recommendations=recs
            )
            
            st.session_state.manual_colors[selected_param] = result
            st.success(f"✅ Added {selected_param}: {value:.2f}")
        
        # Show manual entries
        if st.session_state.manual_colors:
            st.markdown("### Manual Entries")
            for param, result in st.session_state.manual_colors.items():
                render_result_card(result)
                if st.button(f"Remove {param}", key=f"rm_{param}"):
                    del st.session_state.manual_colors[param]
                    st.rerun()
    
    # ===== COMBINED ANALYSIS =====
    st.markdown("---")
    st.markdown("## 📋 Analysis & Recommendations")
    
    # Combine all results
    combined_results = all_results + list(st.session_state.manual_colors.values())
    
    if not combined_results:
        st.info("👆 Upload test strip photos or use manual entry to analyze water parameters")
    else:
        # Overall status
        statuses = [r.status for r in combined_results]
        
        if "danger" in statuses:
            st.error("## 🚨 ACTION REQUIRED")
            st.markdown("Critical issues detected. Take immediate action!")
        elif "warning" in statuses:
            st.warning("## ⚠️ ATTENTION NEEDED")
            st.markdown("Some parameters need attention.")
        else:
            st.success("## ✅ ALL GOOD!")
            st.markdown("Your water parameters look healthy!")
            st.balloons()
        
        # Priority actions
        priority_recs = []
        for r in combined_results:
            if r.status in ["danger", "warning"]:
                priority_recs.extend(r.recommendations)
        
        if priority_recs:
            st.markdown("### 🎯 Priority Actions")
            for rec in list(dict.fromkeys(priority_recs))[:6]:  # Dedupe, limit to 6
                st.markdown(f"""<div class="priority-action">• {rec}</div>""", unsafe_allow_html=True)
        
        # Nitrogen cycle
        nh3 = next((r.value for r in combined_results if r.name == "Ammonia"), None)
        no2 = next((r.value for r in combined_results if r.name == "Nitrite"), None)
        no3 = next((r.value for r in combined_results if r.name == "Nitrate"), None)
        
        if nh3 is not None or no2 is not None:
            st.markdown("### 🔄 Nitrogen Cycle Status")
            
            nh3 = nh3 or 0
            no2 = no2 or 0
            no3 = no3 or 0
            
            if nh3 == 0 and no2 == 0 and no3 > 0:
                st.success("✅ Fully cycled - beneficial bacteria working properly")
            elif nh3 > 0 and no2 == 0 and no3 == 0:
                st.warning("⚠️ Early cycling or ammonia source issue")
            elif nh3 > 0 and no2 > 0:
                st.error("🚨 Active cycling - frequent water changes needed")
            elif nh3 == 0 and no2 > 0:
                st.warning("⚠️ Mid-cycle - nitrite-converting bacteria establishing")
            else:
                st.info("Upload both 10-in-1 and ammonia strips for full cycle assessment")
        
        # Specific gravity recommendations
        if specific_gravity and water_type == "Saltwater":
            st.markdown("### 📏 Specific Gravity")
            if 1.024 <= specific_gravity <= 1.026:
                st.success(f"✅ SG {specific_gravity:.3f} - Ideal range")
            elif 1.022 <= specific_gravity <= 1.027:
                st.info(f"ℹ️ SG {specific_gravity:.3f} - Acceptable")
            elif specific_gravity < 1.022:
                st.warning(f"⚠️ SG {specific_gravity:.3f} - Low. Gradually add salt mix.")
            else:
                st.warning(f"⚠️ SG {specific_gravity:.3f} - High. Top off with freshwater.")
        
        # Download report
        st.markdown("---")
        report = generate_report(combined_results, water_type, specific_gravity)
        st.download_button(
            "📥 Download Report",
            data=report,
            file_name="water_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Clear button
    if combined_results or st.session_state.manual_colors:
        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.manual_colors = {}
            st.session_state.analysis_results = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    🐠 Aquarium Water Analyzer v3.0<br>
    For SJ Wave Test Strips<br>
    <em>Always verify critical readings with professional testing</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.markdown("Please refresh the page and try again.")
        if st.checkbox("Show error details"):
            st.code(traceback.format_exc())
