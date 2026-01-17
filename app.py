"""
Aquarium Water Test Strip Analyzer - Enhanced Version
A Streamlit web app for analyzing SJ Wave test strips (10-in-1 and Ammonia)
With automatic chart detection and click-to-sample functionality

Features:
- Automatic chart/strip detection and perspective correction
- Click on image to sample colors
- Comprehensive water quality report
- Saltwater/Freshwater support
- Specific gravity integration
- Mobile-optimized interface
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import json

# Page configuration
st.set_page_config(
    page_title="🐠 Aquarium Analyzer",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-friendly CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    h1, h2, h3 {
        color: #4fc3f7 !important;
    }
    
    .result-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .status-ok {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
    }
    
    .param-row {
        display: flex;
        align-items: center;
        padding: 15px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin: 8px 0;
    }
    
    .color-swatch {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        margin-right: 15px;
        border: 2px solid white;
    }
    
    .instruction-box {
        background: rgba(79, 195, 247, 0.1);
        border-left: 4px solid #4fc3f7;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    .stSelectbox, .stTextInput {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA CLASSES ====================

@dataclass
class ColorReference:
    """Reference color for a specific value"""
    value: float
    rgb: Tuple[int, int, int]
    label: str = ""


@dataclass
class ParameterConfig:
    """Configuration for a water parameter"""
    name: str
    display_name: str
    unit: str
    colors: List[ColorReference]
    ideal_range: Tuple[float, float]
    ok_range: Tuple[float, float]
    warning_range: Tuple[float, float]
    

@dataclass 
class AnalysisResult:
    """Result of analyzing a single parameter"""
    parameter: str
    display_name: str
    value: float
    unit: str
    status: str
    detected_color: Tuple[int, int, int]
    recommendations: List[str]
    confidence: float = 0.0


# ==================== REFERENCE DATA ====================

# Ammonia color chart (SJ Wave) - Yellow to Teal-Green
AMMONIA_COLORS = [
    ColorReference(0, (230, 225, 140), "Ideal"),
    ColorReference(0.25, (200, 220, 160), "Safe"),
    ColorReference(0.5, (170, 210, 170), "Safe"),
    ColorReference(1, (140, 195, 165), "Stress"),
    ColorReference(3, (110, 175, 155), "Harmful"),
    ColorReference(6, (85, 155, 145), "Danger"),
]

# 10-in-1 color charts
PARAM_CONFIGS = {
    'iron': {
        'display_name': 'Iron (Fe)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (250, 240, 230), "OK"),
            ColorReference(5, (255, 200, 170), ""),
            ColorReference(10, (255, 170, 130), ""),
            ColorReference(25, (255, 140, 100), ""),
            ColorReference(50, (255, 110, 70), ""),
            ColorReference(100, (230, 80, 50), ""),
        ],
        'saltwater': {'ideal': (0, 0), 'ok': (0, 5), 'warning': (5, 25), 'danger': (25, 999)},
        'freshwater': {'ideal': (0, 0), 'ok': (0, 5), 'warning': (5, 25), 'danger': (25, 999)},
    },
    'copper': {
        'display_name': 'Copper (Cu)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (250, 245, 240), "OK"),
            ColorReference(10, (240, 225, 215), ""),
            ColorReference(30, (230, 200, 195), ""),
            ColorReference(100, (210, 160, 175), ""),
            ColorReference(200, (185, 130, 155), ""),
            ColorReference(300, (155, 100, 135), ""),
        ],
        'saltwater': {'ideal': (0, 0), 'ok': (0, 0), 'warning': (0, 10), 'danger': (10, 999)},
        'freshwater': {'ideal': (0, 0), 'ok': (0, 10), 'warning': (10, 30), 'danger': (30, 999)},
    },
    'nitrate': {
        'display_name': 'Nitrate (NO₃)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (255, 255, 245), ""),
            ColorReference(10, (255, 240, 200), "Safe"),
            ColorReference(25, (255, 220, 150), "Safe"),
            ColorReference(50, (240, 190, 100), ""),
            ColorReference(100, (220, 150, 70), "Water Change"),
            ColorReference(250, (190, 110, 50), "Water Change"),
        ],
        'saltwater': {'ideal': (0, 10), 'ok': (0, 25), 'warning': (25, 50), 'danger': (50, 999)},
        'freshwater': {'ideal': (0, 20), 'ok': (0, 40), 'warning': (40, 80), 'danger': (80, 999)},
    },
    'nitrite': {
        'display_name': 'Nitrite (NO₂)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (255, 250, 252), "OK"),
            ColorReference(1, (255, 210, 230), ""),
            ColorReference(5, (255, 160, 195), "Water Change"),
            ColorReference(10, (240, 110, 165), "Water Change"),
        ],
        'saltwater': {'ideal': (0, 0), 'ok': (0, 0.5), 'warning': (0.5, 1), 'danger': (1, 999)},
        'freshwater': {'ideal': (0, 0), 'ok': (0, 0.5), 'warning': (0.5, 1), 'danger': (1, 999)},
    },
    'chlorine': {
        'display_name': 'Chlorine (Cl₂)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (255, 252, 255), "OK"),
            ColorReference(0.8, (245, 225, 245), ""),
            ColorReference(1.5, (230, 190, 230), "Water Change"),
            ColorReference(3, (210, 150, 210), "Water Change"),
        ],
        'saltwater': {'ideal': (0, 0), 'ok': (0, 0), 'warning': (0, 0.5), 'danger': (0.5, 999)},
        'freshwater': {'ideal': (0, 0), 'ok': (0, 0), 'warning': (0, 0.5), 'danger': (0.5, 999)},
    },
    'hardness': {
        'display_name': 'Total Hardness (GH)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (245, 235, 255), ""),
            ColorReference(25, (210, 190, 235), ""),
            ColorReference(75, (180, 160, 215), "OK"),
            ColorReference(150, (150, 130, 195), "OK"),
            ColorReference(300, (120, 100, 175), ""),
        ],
        'saltwater': {'ideal': (150, 300), 'ok': (75, 300), 'warning': (25, 75), 'danger': (0, 25)},
        'freshwater': {'ideal': (75, 150), 'ok': (50, 200), 'warning': (0, 50), 'danger': (200, 999)},
    },
    'alkalinity': {
        'display_name': 'Total Alkalinity (TAL)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (200, 245, 235), ""),
            ColorReference(40, (160, 230, 210), ""),
            ColorReference(80, (120, 210, 180), "OK"),
            ColorReference(120, (85, 190, 150), "OK"),
            ColorReference(180, (55, 170, 120), ""),
            ColorReference(300, (35, 150, 90), ""),
        ],
        'saltwater': {'ideal': (120, 180), 'ok': (80, 200), 'warning': (40, 80), 'danger': (0, 40)},
        'freshwater': {'ideal': (80, 120), 'ok': (40, 180), 'warning': (0, 40), 'danger': (180, 999)},
    },
    'carbonate': {
        'display_name': 'Carbonate Hardness (KH)',
        'unit': 'ppm',
        'colors': [
            ColorReference(0, (205, 245, 225), ""),
            ColorReference(40, (165, 230, 190), ""),
            ColorReference(80, (125, 210, 150), "OK"),
            ColorReference(120, (90, 190, 115), "OK"),
            ColorReference(180, (60, 170, 85), ""),
            ColorReference(300, (40, 150, 60), ""),
        ],
        'saltwater': {'ideal': (120, 180), 'ok': (80, 200), 'warning': (40, 80), 'danger': (0, 40)},
        'freshwater': {'ideal': (80, 120), 'ok': (40, 180), 'warning': (0, 40), 'danger': (180, 999)},
    },
    'ph': {
        'display_name': 'pH',
        'unit': '',
        'colors_freshwater': [
            ColorReference(6.4, (255, 200, 100), ""),
            ColorReference(6.8, (255, 220, 80), ""),
            ColorReference(7.2, (235, 235, 65), "OK"),
            ColorReference(7.6, (185, 225, 85), "OK"),
            ColorReference(8.0, (135, 205, 105), ""),
            ColorReference(8.4, (90, 185, 125), ""),
        ],
        'colors_saltwater': [
            ColorReference(6.8, (255, 185, 105), ""),
            ColorReference(7.2, (255, 145, 85), ""),
            ColorReference(7.6, (255, 105, 105), ""),
            ColorReference(8.0, (255, 85, 125), "OK"),
            ColorReference(8.4, (235, 65, 145), "OK"),
            ColorReference(9.0, (205, 55, 165), ""),
        ],
        'saltwater': {'ideal': (8.0, 8.4), 'ok': (7.8, 8.5), 'warning': (7.6, 7.8), 'danger': (0, 7.6)},
        'freshwater': {'ideal': (6.8, 7.6), 'ok': (6.4, 8.0), 'warning': (6.0, 6.4), 'danger': (0, 6.0)},
    },
}

# Specific gravity ranges for saltwater
SG_RANGES = {
    'ideal': (1.024, 1.026),
    'ok': (1.022, 1.027),
    'warning': (1.019, 1.022),
    'danger': (0, 1.019),
}


# ==================== COLOR ANALYSIS ====================

def weighted_color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptually-weighted color distance.
    Uses a modified formula that accounts for human color perception.
    """
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    
    rmean = (r1 + r2) / 2
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    
    # Weighted Euclidean distance
    return np.sqrt(
        (2 + rmean/256) * dr**2 +
        4 * dg**2 +
        (2 + (255-rmean)/256) * db**2
    )


def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSV"""
    r, g, b = [x/255.0 for x in rgb]
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    # Hue
    if diff == 0:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation
    s = 0 if cmax == 0 else (diff / cmax) * 100
    
    # Value
    v = cmax * 100
    
    return (h, s, v)


def find_best_match(detected_color: Tuple[int, int, int], 
                    color_refs: List[ColorReference]) -> Tuple[float, float]:
    """
    Find the best matching value from color references.
    Returns (interpolated_value, confidence)
    """
    if not color_refs:
        return 0, 0
    
    # Calculate distances to all references
    distances = []
    for ref in color_refs:
        dist = weighted_color_distance(detected_color, ref.rgb)
        distances.append((ref.value, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    # Get best match
    best_value, best_dist = distances[0]
    
    # Interpolate between two closest if possible
    if len(distances) >= 2:
        v1, d1 = distances[0]
        v2, d2 = distances[1]
        
        if d1 + d2 > 0:
            # Weighted interpolation
            weight = d1 / (d1 + d2)
            interpolated = v1 + (v2 - v1) * weight * 0.5  # Conservative interpolation
        else:
            interpolated = v1
    else:
        interpolated = best_value
    
    # Calculate confidence (inverse of normalized distance)
    max_possible_dist = 765  # sqrt(255^2 * 3) approx
    confidence = max(0, min(100, (1 - best_dist / max_possible_dist) * 100))
    
    return interpolated, confidence


def get_status_for_value(value: float, ranges: Dict, reverse: bool = False) -> str:
    """Determine status based on value and ranges"""
    ideal = ranges.get('ideal', (0, 0))
    ok = ranges.get('ok', (0, 0))
    warning = ranges.get('warning', (0, 0))
    
    if ideal[0] <= value <= ideal[1]:
        return 'ok'
    elif ok[0] <= value <= ok[1]:
        return 'ok'
    elif warning[0] <= value <= warning[1]:
        return 'warning'
    else:
        return 'danger'


# ==================== IMAGE PROCESSING ====================

def preprocess_for_color_detection(image: np.ndarray) -> np.ndarray:
    """Preprocess image for accurate color detection"""
    # Convert to LAB for perceptual uniformity
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Adaptive histogram equalization on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Slight Gaussian blur to reduce noise
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    
    return processed


def sample_color_at_point(image: np.ndarray, x: int, y: int, 
                          radius: int = 10) -> Tuple[int, int, int]:
    """Sample the median color in a circular region around a point"""
    h, w = image.shape[:2]
    
    # Clamp coordinates
    x = max(radius, min(w - radius, x))
    y = max(radius, min(h - radius, y))
    
    # Extract region
    region = image[y-radius:y+radius, x-radius:x+radius]
    
    if region.size == 0:
        return (128, 128, 128)
    
    # Create circular mask
    mask = np.zeros((radius*2, radius*2), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    
    # Apply mask and get median
    pixels = region[mask > 0]
    if len(pixels) > 0:
        median_color = np.median(pixels, axis=0).astype(int)
        return tuple(median_color)
    
    return (128, 128, 128)


def detect_strip_orientation(image: np.ndarray) -> str:
    """Detect if strip is horizontal or vertical"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return 'vertical'
    
    horizontal_count = 0
    vertical_count = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:
            horizontal_count += 1
        elif 60 < angle < 120:
            vertical_count += 1
    
    return 'horizontal' if horizontal_count > vertical_count else 'vertical'


# ==================== RECOMMENDATIONS ====================

def get_recommendations(param: str, value: float, status: str, water_type: str) -> List[str]:
    """Generate recommendations based on parameter and value"""
    recs = []
    
    if param == 'ammonia':
        if status == 'warning':
            recs = [
                "Perform 25% water change",
                "Reduce feeding for 2-3 days",
                "Check for dead organisms or decaying matter",
                "Test for nitrite to check nitrogen cycle"
            ]
        elif status == 'danger':
            recs = [
                "🚨 EMERGENCY: Perform 50% water change immediately",
                "Add ammonia neutralizer (Seachem Prime or similar)",
                "Stop feeding completely for 48 hours",
                "Check and clean filter media (in tank water)",
                "Consider adding beneficial bacteria supplement"
            ]
    
    elif param == 'nitrite':
        if status == 'warning':
            recs = [
                "Perform 25-30% water change",
                "Nitrogen cycle may be incomplete",
                "Add beneficial bacteria supplement",
                "Reduce feeding"
            ]
        elif status == 'danger':
            recs = [
                "🚨 URGENT: 50% water change needed",
                "Add Prime to detoxify nitrite",
                "Heavily reduce or stop feeding",
                "Check filter function"
            ]
    
    elif param == 'nitrate':
        if status == 'warning':
            recs = [
                "Schedule 25-30% water change",
                f"{'Consider adding macroalgae or refugium' if water_type == 'saltwater' else 'Add live plants to absorb nitrates'}",
                "Review feeding schedule"
            ]
        elif status == 'danger':
            recs = [
                "Large water change needed (40-50%)",
                "Check for hidden detritus",
                "Review and reduce feeding amount",
                "Consider more frequent water changes"
            ]
    
    elif param == 'ph':
        if water_type == 'saltwater':
            if value < 8.0:
                recs = [
                    "pH is low for saltwater",
                    "Check and boost alkalinity",
                    "Increase surface agitation for CO2 off-gassing",
                    "Consider pH buffer"
                ]
            elif value > 8.5:
                recs = [
                    "pH is elevated",
                    "Check dosing equipment",
                    "Ensure proper lighting schedule"
                ]
        else:
            if value < 6.5:
                recs = [
                    "pH is low for most freshwater fish",
                    "Add crushed coral or limestone",
                    "Check KH levels",
                    "Consider pH buffer"
                ]
            elif value > 8.0:
                recs = [
                    "pH is elevated",
                    "Add driftwood or peat",
                    "Use RO water for water changes"
                ]
    
    elif param == 'alkalinity' or param == 'carbonate':
        if water_type == 'saltwater':
            if value < 80:
                recs = [
                    "Alkalinity too low for reef keeping",
                    "Add alkalinity supplement (baking soda or commercial)",
                    "Check calcium reactor or kalkwasser dosing"
                ]
            elif value > 200:
                recs = [
                    "Alkalinity elevated",
                    "Reduce dosing",
                    "Check for precipitation"
                ]
    
    elif param == 'copper':
        if value > 0:
            recs = [
                "🚨 Copper detected - toxic to invertebrates!",
                "Run activated carbon immediately",
                "Check for copper contamination source",
                "Do NOT add invertebrates until copper is 0"
            ]
    
    elif param == 'chlorine':
        if value > 0:
            recs = [
                "🚨 Chlorine detected!",
                "Add dechlorinator immediately",
                "Always treat tap water before adding to tank"
            ]
    
    return recs


def get_sg_recommendations(sg: float) -> List[str]:
    """Get recommendations for specific gravity"""
    recs = []
    
    if sg < 1.020:
        recs = [
            "Salinity critically low",
            "Add salt mix gradually over several hours",
            "Do not raise more than 0.002 per day"
        ]
    elif sg < 1.022:
        recs = [
            "Salinity slightly low",
            "Gradually add salt mix to raise to 1.024-1.026"
        ]
    elif sg > 1.028:
        recs = [
            "Salinity too high",
            "Top off with freshwater (not saltwater)",
            "Reduce slowly over time"
        ]
    elif sg > 1.026:
        recs = [
            "Salinity slightly elevated",
            "Use freshwater for top-offs"
        ]
    
    return recs


# ==================== REPORT GENERATION ====================

def generate_text_report(results: Dict[str, AnalysisResult], 
                         water_type: str,
                         specific_gravity: Optional[float] = None) -> str:
    """Generate a downloadable text report"""
    report = []
    report.append("=" * 50)
    report.append("AQUARIUM WATER QUALITY REPORT")
    report.append("=" * 50)
    report.append("")
    report.append(f"Water Type: {water_type.upper()}")
    if specific_gravity:
        report.append(f"Specific Gravity: {specific_gravity:.3f}")
    report.append("")
    report.append("-" * 50)
    report.append("PARAMETER RESULTS")
    report.append("-" * 50)
    
    for param, result in results.items():
        report.append("")
        status_icon = "✅" if result.status == "ok" else ("⚠️" if result.status == "warning" else "🚨")
        report.append(f"{status_icon} {result.display_name}: {result.value:.2f} {result.unit}")
        report.append(f"   Status: {result.status.upper()}")
        report.append(f"   Confidence: {result.confidence:.0f}%")
        
        if result.recommendations:
            report.append("   Recommendations:")
            for rec in result.recommendations:
                report.append(f"   • {rec}")
    
    report.append("")
    report.append("-" * 50)
    report.append("END OF REPORT")
    report.append("-" * 50)
    
    return "\n".join(report)


# ==================== STREAMLIT UI ====================

def create_color_preview(color: Tuple[int, int, int], size: int = 50) -> np.ndarray:
    """Create a color preview image"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def render_parameter_card(result: AnalysisResult):
    """Render a parameter result card"""
    status_colors = {
        'ok': ('#00b09b', '#96c93d'),
        'warning': ('#f093fb', '#f5576c'),
        'danger': ('#eb3349', '#f45c43')
    }
    
    colors = status_colors.get(result.status, status_colors['ok'])
    icon = "✅" if result.status == "ok" else ("⚠️" if result.status == "warning" else "🚨")
    
    with st.expander(f"{icon} {result.display_name}: {result.value:.2f} {result.unit}", 
                     expanded=(result.status != 'ok')):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            swatch = create_color_preview(result.detected_color, 60)
            st.image(swatch, caption="Detected", use_container_width=False)
        
        with col2:
            st.markdown(f"**Value:** {result.value:.2f} {result.unit}")
            st.markdown(f"**Status:** {result.status.upper()}")
            st.progress(result.confidence / 100, text=f"Confidence: {result.confidence:.0f}%")
        
        if result.recommendations:
            st.markdown("**Recommendations:**")
            for rec in result.recommendations:
                st.markdown(f"• {rec}")


def main():
    st.title("🐠 Aquarium Water Analyzer")
    st.markdown("*Analyze your SJ Wave test strips with AI-powered color detection*")
    
    # Initialize session state
    if 'sampled_colors' not in st.session_state:
        st.session_state.sampled_colors = {}
    if 'ammonia_color' not in st.session_state:
        st.session_state.ammonia_color = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # ===== CONFIGURATION SECTION =====
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        water_type = st.selectbox(
            "🌊 Water Type",
            ["Saltwater", "Freshwater"],
            key="water_type"
        )
        water_type_key = water_type.lower()
    
    with col2:
        if water_type_key == 'saltwater':
            sg_input = st.text_input(
                "📏 Specific Gravity",
                placeholder="e.g., 1.025",
                help="Enter your refractometer reading"
            )
            specific_gravity = None
            if sg_input:
                try:
                    specific_gravity = float(sg_input)
                    if specific_gravity < 1.0 or specific_gravity > 1.1:
                        st.warning("Value seems unusual. Normal range is 1.020-1.030")
                except:
                    st.error("Enter a valid number (e.g., 1.025)")
        else:
            specific_gravity = None
    
    # ===== IMAGE UPLOAD SECTION =====
    st.markdown("---")
    st.markdown("### 📸 Upload Test Strip Images")
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Instructions:</strong>
    <ol>
    <li>Take photos of your test strips next to their color charts</li>
    <li>Ensure good natural lighting (avoid shadows)</li>
    <li>Upload images and sample colors from each pad</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 10-in-1 Strip", "🧪 Ammonia Strip"])
    
    # ----- 10-in-1 Tab -----
    with tab1:
        ten_in_one_file = st.file_uploader(
            "Upload 10-in-1 test strip image",
            type=['jpg', 'jpeg', 'png'],
            key="upload_10in1",
            help="Photo should show the strip alongside the color chart"
        )
        
        if ten_in_one_file:
            image = Image.open(ten_in_one_file)
            img_array = np.array(image)
            
            # Preprocess for better color detection
            processed = preprocess_for_color_detection(img_array)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.markdown("#### 🎨 Sample Colors")
            st.markdown("Select a parameter, then enter coordinates or use the color picker.")
            
            # Parameter selector
            param_names = list(PARAM_CONFIGS.keys())
            selected_param = st.selectbox(
                "Select parameter to sample:",
                param_names,
                format_func=lambda x: PARAM_CONFIGS[x]['display_name'],
                key="param_select"
            )
            
            # Sampling methods
            method = st.radio(
                "Sampling method:",
                ["Color Picker", "Coordinates"],
                horizontal=True,
                key="sample_method"
            )
            
            if method == "Color Picker":
                picked = st.color_picker(
                    f"Pick color for {PARAM_CONFIGS[selected_param]['display_name']}",
                    value="#FFFFFF",
                    key=f"picker_{selected_param}"
                )
                
                if st.button(f"Set {selected_param} color", key=f"set_{selected_param}"):
                    hex_color = picked.lstrip('#')
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    st.session_state.sampled_colors[selected_param] = rgb
                    st.success(f"✅ Set {PARAM_CONFIGS[selected_param]['display_name']} to RGB{rgb}")
            
            else:  # Coordinates
                st.markdown(f"Image size: {img_array.shape[1]} x {img_array.shape[0]}")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    x_val = st.number_input("X", 0, img_array.shape[1]-1, 100, key=f"x_{selected_param}")
                with col_y:
                    y_val = st.number_input("Y", 0, img_array.shape[0]-1, 100, key=f"y_{selected_param}")
                
                if st.button(f"Sample at ({x_val}, {y_val})", key=f"sample_coord_{selected_param}"):
                    color = sample_color_at_point(processed, int(x_val), int(y_val), radius=12)
                    st.session_state.sampled_colors[selected_param] = color
                    
                    swatch = create_color_preview(color, 50)
                    st.image(swatch, caption=f"Sampled: RGB{color}", width=50)
                    st.success(f"✅ Sampled {PARAM_CONFIGS[selected_param]['display_name']}")
            
            # Show all sampled colors
            if st.session_state.sampled_colors:
                st.markdown("#### 📋 Sampled Parameters")
                
                cols = st.columns(3)
                for i, (param, color) in enumerate(st.session_state.sampled_colors.items()):
                    with cols[i % 3]:
                        swatch = create_color_preview(color, 40)
                        display_name = PARAM_CONFIGS.get(param, {}).get('display_name', param)
                        st.image(swatch, caption=f"{display_name}", width=40)
                        if st.button("❌", key=f"remove_{param}"):
                            del st.session_state.sampled_colors[param]
                            st.rerun()
    
    # ----- Ammonia Tab -----
    with tab2:
        ammonia_file = st.file_uploader(
            "Upload ammonia test strip image",
            type=['jpg', 'jpeg', 'png'],
            key="upload_ammonia",
            help="The ammonia pad is at the END of the strip"
        )
        
        if ammonia_file:
            image = Image.open(ammonia_file)
            img_array = np.array(image)
            processed = preprocess_for_color_detection(img_array)
            
            st.image(image, caption="Uploaded Ammonia Strip", use_container_width=True)
            
            st.markdown("""
            **Tip:** The ammonia test pad is the small colored square at the very end 
            of the strip. It changes from yellow (0 ppm) to green/teal (high ammonia).
            """)
            
            method = st.radio(
                "Sampling method:",
                ["Color Picker", "Coordinates"],
                horizontal=True,
                key="ammonia_method"
            )
            
            if method == "Color Picker":
                ammonia_picked = st.color_picker(
                    "Pick ammonia pad color",
                    value="#E6E696",
                    key="ammonia_picker"
                )
                
                if st.button("Set ammonia color", key="set_ammonia"):
                    hex_color = ammonia_picked.lstrip('#')
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    st.session_state.ammonia_color = rgb
                    st.success(f"✅ Set ammonia to RGB{rgb}")
            
            else:
                col_x, col_y = st.columns(2)
                with col_x:
                    ax = st.number_input("X", 0, img_array.shape[1]-1, 50, key="ax")
                with col_y:
                    ay = st.number_input("Y", 0, img_array.shape[0]-1, 50, key="ay")
                
                if st.button("Sample ammonia", key="sample_ammonia_coord"):
                    color = sample_color_at_point(processed, int(ax), int(ay), radius=10)
                    st.session_state.ammonia_color = color
                    
                    swatch = create_color_preview(color, 50)
                    st.image(swatch, caption=f"Sampled: RGB{color}", width=50)
                    st.success("✅ Sampled ammonia color")
            
            if st.session_state.ammonia_color:
                swatch = create_color_preview(st.session_state.ammonia_color, 50)
                st.image(swatch, caption=f"Current: RGB{st.session_state.ammonia_color}", width=50)
    
    # ===== ANALYSIS SECTION =====
    st.markdown("---")
    st.markdown("### 🔬 Analysis")
    
    has_data = bool(st.session_state.sampled_colors) or st.session_state.ammonia_color is not None
    
    if not has_data:
        st.info("👆 Upload images and sample colors to analyze your water parameters")
    else:
        if st.button("🔬 Analyze Water Parameters", type="primary", use_container_width=True):
            with st.spinner("Analyzing colors..."):
                results = {}
                
                # Analyze 10-in-1 parameters
                for param, color in st.session_state.sampled_colors.items():
                    config = PARAM_CONFIGS.get(param)
                    if not config:
                        continue
                    
                    # Get appropriate color references
                    if param == 'ph':
                        colors = config.get(f'colors_{water_type_key}', config.get('colors_freshwater'))
                    else:
                        colors = config.get('colors', [])
                    
                    color_refs = [ColorReference(c.value, c.rgb, c.label) for c in colors]
                    
                    # Find best match
                    value, confidence = find_best_match(color, color_refs)
                    
                    # Get status
                    ranges = config.get(water_type_key, config.get('freshwater', {}))
                    status = get_status_for_value(value, ranges)
                    
                    # Get recommendations
                    recs = get_recommendations(param, value, status, water_type_key)
                    
                    results[param] = AnalysisResult(
                        parameter=param,
                        display_name=config['display_name'],
                        value=value,
                        unit=config['unit'],
                        status=status,
                        detected_color=color,
                        recommendations=recs,
                        confidence=confidence
                    )
                
                # Analyze ammonia
                if st.session_state.ammonia_color:
                    color = st.session_state.ammonia_color
                    color_refs = [ColorReference(c.value, c.rgb, c.label) for c in AMMONIA_COLORS]
                    value, confidence = find_best_match(color, color_refs)
                    
                    ranges = {'ideal': (0, 0), 'ok': (0, 0.25), 'warning': (0.25, 1), 'danger': (1, 999)}
                    status = get_status_for_value(value, ranges)
                    recs = get_recommendations('ammonia', value, status, water_type_key)
                    
                    results['ammonia'] = AnalysisResult(
                        parameter='ammonia',
                        display_name='Ammonia (NH₃/NH₄⁺)',
                        value=value,
                        unit='ppm',
                        status=status,
                        detected_color=color,
                        recommendations=recs,
                        confidence=confidence
                    )
                
                # Analyze specific gravity
                if specific_gravity and water_type_key == 'saltwater':
                    status = get_status_for_value(specific_gravity, SG_RANGES)
                    recs = get_sg_recommendations(specific_gravity)
                    
                    results['specific_gravity'] = AnalysisResult(
                        parameter='specific_gravity',
                        display_name='Specific Gravity',
                        value=specific_gravity,
                        unit='',
                        status=status,
                        detected_color=(0, 0, 0),
                        recommendations=recs,
                        confidence=100
                    )
                
                st.session_state.analysis_results = results
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Overall status
            statuses = [r.status for r in results.values()]
            if 'danger' in statuses:
                overall = 'danger'
                st.error("## 🚨 ACTION REQUIRED")
                st.markdown("Critical issues detected. Take immediate action!")
            elif 'warning' in statuses:
                overall = 'warning'
                st.warning("## ⚠️ ATTENTION NEEDED")
                st.markdown("Some parameters need attention.")
            else:
                overall = 'ok'
                st.success("## ✅ ALL GOOD!")
                st.markdown("Your water parameters look healthy!")
                st.balloons()
            
            # Priority actions
            all_recs = []
            for r in results.values():
                if r.status in ['danger', 'warning']:
                    all_recs.extend(r.recommendations)
            
            if all_recs:
                st.markdown("### 🎯 Priority Actions")
                for rec in list(set(all_recs))[:5]:
                    st.markdown(f"• {rec}")
            
            # Parameter cards
            st.markdown("### 📊 Detailed Results")
            
            for param, result in results.items():
                render_parameter_card(result)
            
            # Nitrogen cycle summary (for saltwater)
            if water_type_key == 'saltwater' and 'ammonia' in results and 'nitrite' in results:
                st.markdown("### 🔄 Nitrogen Cycle Status")
                
                ammonia_val = results['ammonia'].value
                nitrite_val = results['nitrite'].value
                nitrate_val = results.get('nitrate', AnalysisResult("", "", 0, "", "ok", (0,0,0), [], 0)).value
                
                if ammonia_val == 0 and nitrite_val == 0:
                    st.success("✅ Nitrogen cycle established and healthy")
                elif ammonia_val > 0 and nitrite_val == 0:
                    st.warning("⚠️ Ammonia present - cycle may be starting or disrupted")
                elif ammonia_val > 0 and nitrite_val > 0:
                    st.warning("⚠️ Both ammonia and nitrite present - cycle incomplete")
                elif ammonia_val == 0 and nitrite_val > 0:
                    st.info("🔄 Nitrite present - cycle in progress, needs more time")
            
            # Download report
            st.markdown("---")
            st.markdown("### 💾 Export")
            
            report = generate_text_report(results, water_type_key, specific_gravity)
            
            st.download_button(
                "📥 Download Report",
                data=report,
                file_name="aquarium_water_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Clear button
    if has_data:
        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.sampled_colors = {}
            st.session_state.ammonia_color = None
            st.session_state.analysis_results = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    🐠 Aquarium Water Analyzer v2.0<br>
    For SJ Wave Test Strips<br>
    <em>Always verify with professional testing for critical decisions</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
