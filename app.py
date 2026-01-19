"""
Aquarium Water Test Strip Analyzer - v4.1 FINAL
For SJ Wave 10-in-1 and Ammonia test strips

CALIBRATED: Strip pads are on the VERY LEFT edge of the card (x=card_left to card_left+30)
Gray/white pads = values near 0 (safe levels)
"""

import streamlit as st
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import traceback

st.set_page_config(page_title="🐠 Aquarium Analyzer", page_icon="🐠", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding: 1rem; max-width: 100%; }
    .result-card { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; padding: 16px; margin: 10px 0; border-left: 4px solid #1976D2; }
    .status-ok { color: #2E7D32; font-weight: bold; background: #E8F5E9; padding: 4px 12px; border-radius: 20px; }
    .status-warning { color: #F57C00; font-weight: bold; background: #FFF3E0; padding: 4px 12px; border-radius: 20px; }
    .status-danger { color: #C62828; font-weight: bold; background: #FFEBEE; padding: 4px 12px; border-radius: 20px; }
    .color-swatch { display: inline-block; width: 28px; height: 28px; border-radius: 6px; border: 2px solid #333; vertical-align: middle; margin-right: 12px; }
    .priority-action { background: #FFEBEE; border-left: 4px solid #C62828; padding: 12px; border-radius: 0 8px 8px 0; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

@dataclass
class AnalysisResult:
    name: str
    display_name: str
    value: float
    unit: str
    status: str
    detected_color: Tuple[int, int, int]
    confidence: float
    recommendations: List[str]

# Reference colors for SJ Wave strips
REFERENCE_COLORS = {
    "Iron": {0: (220, 220, 215), 5: (255, 210, 190), 10: (255, 185, 160), 25: (255, 150, 120), 50: (255, 120, 90), 100: (220, 90, 70)},
    "Copper": {0: (220, 220, 215), 10: (200, 190, 210), 30: (160, 140, 180), 100: (100, 80, 140), 200: (70, 50, 110), 300: (50, 40, 85)},
    "Nitrate": {0: (220, 220, 215), 10: (245, 240, 180), 25: (235, 215, 120), 50: (215, 185, 80), 100: (185, 145, 60), 250: (155, 95, 50)},
    "Nitrite": {0: (220, 220, 215), 1: (250, 215, 230), 5: (245, 175, 205), 10: (225, 125, 175)},
    "Chlorine": {0: (220, 220, 215), 0.8: (240, 225, 245), 1.5: (225, 195, 235), 3: (195, 155, 215)},
    "Hardness": {0: (220, 220, 215), 25: (190, 215, 210), 75: (175, 195, 200), 150: (165, 175, 200), 300: (155, 150, 205)},
    "Alkalinity": {0: (220, 220, 215), 40: (175, 215, 195), 80: (145, 205, 175), 120: (115, 195, 155), 180: (85, 180, 130), 300: (55, 160, 100)},
    "Carbonate": {0: (220, 220, 215), 40: (175, 215, 195), 80: (145, 205, 175), 120: (115, 195, 155), 180: (85, 180, 130), 300: (55, 160, 100)},
    "pH": {6.4: (250, 240, 100), 6.8: (235, 225, 80), 7.2: (195, 205, 60), 7.6: (155, 195, 80), 8.0: (105, 190, 100), 8.4: (65, 175, 145), 9.0: (175, 95, 175)},
}

AMMONIA_COLORS = {0: (225, 225, 145), 0.25: (205, 220, 155), 0.5: (175, 210, 160), 1: (145, 195, 165), 3: (105, 175, 160), 6: (80, 155, 150)}

PARAM_CONFIG = {
    "Iron": {"unit": "ppm", "display": "Iron (Fe)", "saltwater": {"ideal": (0, 0), "ok": (0, 5), "warn": 10, "danger": 25}, "freshwater": {"ideal": (0, 0), "ok": (0, 5), "warn": 10, "danger": 25}},
    "Copper": {"unit": "ppm", "display": "Copper (Cu)", "saltwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 1, "danger": 10}, "freshwater": {"ideal": (0, 0), "ok": (0, 10), "warn": 30, "danger": 100}},
    "Nitrate": {"unit": "ppm", "display": "Nitrate (NO₃)", "saltwater": {"ideal": (0, 10), "ok": (0, 25), "warn": 50, "danger": 80}, "freshwater": {"ideal": (0, 20), "ok": (0, 40), "warn": 60, "danger": 100}},
    "Nitrite": {"unit": "ppm", "display": "Nitrite (NO₂)", "saltwater": {"ideal": (0, 0), "ok": (0, 0.25), "warn": 0.5, "danger": 1}, "freshwater": {"ideal": (0, 0), "ok": (0, 0.5), "warn": 1, "danger": 5}},
    "Chlorine": {"unit": "ppm", "display": "Chlorine (Cl₂)", "saltwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 0.5, "danger": 1}, "freshwater": {"ideal": (0, 0), "ok": (0, 0), "warn": 0.5, "danger": 1}},
    "Hardness": {"unit": "ppm", "display": "Hardness (GH)", "saltwater": {"ideal": (150, 300), "ok": (75, 300), "warn_low": 50}, "freshwater": {"ideal": (75, 150), "ok": (50, 200), "warn_low": 25, "warn_high": 250}},
    "Alkalinity": {"unit": "ppm", "display": "Alkalinity (KH)", "saltwater": {"ideal": (120, 180), "ok": (100, 200), "warn_low": 80}, "freshwater": {"ideal": (80, 120), "ok": (40, 180), "warn_low": 40}},
    "Carbonate": {"unit": "ppm", "display": "Carbonate (KH)", "saltwater": {"ideal": (120, 180), "ok": (100, 200), "warn_low": 80}, "freshwater": {"ideal": (80, 120), "ok": (40, 180), "warn_low": 40}},
    "pH": {"unit": "", "display": "pH", "saltwater": {"ideal": (8.0, 8.4), "ok": (7.8, 8.5), "warn_low": 7.6, "warn_high": 8.6}, "freshwater": {"ideal": (6.8, 7.6), "ok": (6.4, 8.0), "warn_low": 6.0, "warn_high": 8.2}},
    "Ammonia": {"unit": "ppm", "display": "Ammonia (NH₃/NH₄⁺)", "saltwater": {"ideal": (0, 0), "ok": (0, 0.1), "warn": 0.25, "danger": 0.5}, "freshwater": {"ideal": (0, 0), "ok": (0, 0.25), "warn": 0.5, "danger": 1}},
}

STRIP_PARAMS = ["Iron", "Copper", "Nitrate", "Nitrite", "Chlorine", "Hardness", "Alkalinity", "Carbonate", "pH"]

def color_distance(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    rmean = (r1 + r2) / 2
    return np.sqrt((2 + rmean/256) * (r1-r2)**2 + 4 * (g1-g2)**2 + (2 + (255-rmean)/256) * (b1-b2)**2)

def match_color_to_value(detected_rgb, color_chart):
    if not color_chart:
        return 0, 0
    
    r, g, b = detected_rgb
    sat = (max(r,g,b) - min(r,g,b)) / max(r,g,b) if max(r,g,b) > 0 else 0
    brightness = (r + g + b) / 3
    
    # Gray/white pad (low saturation, high brightness) = value near 0
    if sat < 0.10 and brightness > 180:
        return 0, 95  # High confidence for "0" value
    
    values = sorted(color_chart.keys())
    distances = [(v, color_distance(detected_rgb, color_chart[v])) for v in values]
    distances.sort(key=lambda x: x[1])
    v1, d1 = distances[0]
    confidence = max(30, min(95, 100 - d1 * 0.10))
    
    if len(distances) >= 2 and d1 < 200:
        v2, d2 = distances[1]
        if d1 + d2 > 0:
            weight = d2 / (d1 + d2)
            return round(v1 * weight + v2 * (1 - weight), 2), confidence
    return v1, confidence

def get_status(param_name, value, water_type):
    config = PARAM_CONFIG.get(param_name, {})
    ranges = config.get(water_type.lower(), config.get("freshwater", {}))
    if not ranges:
        return "ok"
    if ranges.get("danger") is not None and value >= ranges["danger"]:
        return "danger"
    if ranges.get("warn") is not None and value >= ranges["warn"]:
        return "warning"
    if ranges.get("warn_low") is not None and value < ranges["warn_low"]:
        return "warning"
    if ranges.get("warn_high") is not None and value > ranges["warn_high"]:
        return "warning"
    ok = ranges.get("ok", (None, None))
    if ok and ok[0] is not None and ok[0] <= value <= ok[1]:
        return "ok"
    ideal = ranges.get("ideal", (None, None))
    if ideal and ideal[0] is not None:
        if ideal[0] <= value <= ideal[1]:
            return "ok"
        return "warning"
    return "ok"

def get_recommendations(param_name, value, status, water_type):
    recs = []
    wt = water_type.lower()
    if param_name == "Ammonia":
        if status == "danger":
            recs = ["🚨 EMERGENCY: 50% water change immediately!", "Add ammonia detoxifier (Prime, AmGuard)", "Stop feeding 24-48 hours"]
        elif status == "warning":
            recs = ["25-30% water change", "Reduce feeding"]
    elif param_name == "Nitrite":
        if status == "danger":
            recs = ["🚨 50% water change needed", "Add Prime to detoxify"]
        elif status == "warning":
            recs = ["25-30% water change", "Add bacteria supplement"]
    elif param_name == "Nitrate" and status in ["danger", "warning"]:
        recs = ["Water change (30-50%)", "Add macroalgae" if wt == "saltwater" else "Add live plants"]
    elif param_name == "Copper" and value > 0:
        recs = ["🚨 Copper toxic to inverts!", "Run activated carbon immediately"]
    elif param_name == "Chlorine" and value > 0:
        recs = ["🚨 Add dechlorinator immediately"]
    elif param_name == "pH" and wt == "saltwater" and value < 8.0:
        recs = ["pH low - boost alkalinity"]
    elif param_name in ["Alkalinity", "Carbonate"] and wt == "saltwater" and value < 100:
        recs = ["Add alkalinity supplement"]
    return recs

def safe_load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size[0] < 50 or image.size[1] < 50:
            return None
        max_dim = 2000
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.Resampling.LANCZOS)
        return np.array(image)
    except:
        return None

def sample_region(img, x, y, radius=12):
    h, w = img.shape[:2]
    x1, x2 = max(0, x - radius), min(w, x + radius)
    y1, y2 = max(0, y - radius), min(h, y + radius)
    if x2 <= x1 or y2 <= y1:
        return (128, 128, 128)
    region = img[y1:y2, x1:x2]
    return (int(np.median(region[:,:,0])), int(np.median(region[:,:,1])), int(np.median(region[:,:,2])))

def find_card_bounds(img):
    """Find white card using brightness threshold 200"""
    gray = np.mean(img, axis=2)
    bright = gray > 200
    rows = np.where(np.any(bright, axis=1))[0]
    cols = np.where(np.any(bright, axis=0))[0]
    if len(rows) > 50 and len(cols) > 50:
        top = int(np.percentile(rows, 5))
        bottom = int(np.percentile(rows, 95))
        left = int(np.percentile(cols, 5))
        right = int(np.percentile(cols, 95))
        if right - left > 200 and bottom - top > 200:
            return left, right, top, bottom
    h, w = img.shape[:2]
    return int(w*0.1), int(w*0.9), int(h*0.1), int(h*0.9)

def analyze_10in1_strip(image):
    """
    Analyze 10-in-1 strip - FIXED VERSION
    
    Key insight: Strip pads are on the VERY LEFT edge of the card (x=card_left to card_left+25)
    Gray/white pads = values at 0 (safe levels)
    """
    try:
        height, width = image.shape[:2]
        card_left, card_right, card_top, card_bottom = find_card_bounds(image)
        card_width = card_right - card_left
        card_height = card_bottom - card_top
        
        if card_width < 200 or card_height < 200:
            return None, "Could not detect color chart"
        
        # CRITICAL: Strip placement area is the VERY LEFT edge of card
        # Only 25-30 pixels wide, starting from card_left
        strip_x_start = card_left
        strip_x_end = card_left + 25
        
        # Row positions - calibrated from actual image analysis
        # Rows: Iron, Copper, Nitrate, Nitrite, Chlorine, Hardness, Alkalinity, Carbonate, pH(FW), pH(SW)
        # But the strip only has 9 pads (pH is one pad matching against FW or SW scale)
        
        # Row centers based on analysis: approximately 110 pixels apart
        # First row (Iron) starts around card_top + 110
        row_height = card_height / 12  # ~112 pixels for 1353 height
        
        results = []
        
        for i, param in enumerate(STRIP_PARAMS):
            # Row centers: row 0 at ~card_top+110, then +110 for each row
            y_center = card_top + int(row_height * (i + 1.1))
            
            # Sample from the strip placement area (very left edge)
            best_color = None
            best_sat = -1
            
            for x in range(strip_x_start + 5, strip_x_end, 3):
                color = sample_region(image, x, y_center, radius=10)
                r, g, b = color
                sat = (max(r,g,b) - min(r,g,b)) / max(r,g,b) if max(r,g,b) > 0 else 0
                
                # Keep the most saturated color (most likely to be the actual pad)
                if sat > best_sat:
                    best_sat = sat
                    best_color = color
            
            if best_color is None:
                best_color = (220, 220, 215)
            
            value, confidence = match_color_to_value(best_color, REFERENCE_COLORS.get(param, {}))
            results.append({"name": param, "color": best_color, "value": value, "confidence": confidence})
        
        return results, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def analyze_ammonia_strip(image):
    """Analyze ammonia strip - find yellow-green-teal pad"""
    try:
        height, width = image.shape[:2]
        
        best_score = -1000
        best_x, best_y = width // 4, height // 4
        best_color = (225, 225, 145)
        
        for y in range(50, height - 50, 10):
            for x in range(50, width - 50, 10):
                color = image[y, x, :]
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                brightness = (r + g + b) / 3
                
                if brightness < 80 or brightness > 240:
                    continue
                
                # Score for ammonia colors (yellow to teal)
                score = g - 0.3 * abs(r - g) - 0.2 * b
                sat = (max(r,g,b) - min(r,g,b)) / max(r,g,b) if max(r,g,b) > 0 else 0
                if sat > 0.1:
                    score += sat * 50
                
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y
                    best_color = (r, g, b)
        
        final_color = sample_region(image, best_x, best_y, radius=18)
        r, g, b = final_color
        sat = (max(r,g,b) - min(r,g,b)) / max(r,g,b) if max(r,g,b) > 0 else 0
        if sat < 0.08:
            final_color = best_color
        
        value, confidence = match_color_to_value(final_color, AMMONIA_COLORS)
        return {"name": "Ammonia", "color": final_color, "value": value, "confidence": confidence}
    except:
        return {"name": "Ammonia", "color": (225, 225, 145), "value": 0, "confidence": 30}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_report(results, water_type, sg=None):
    lines = ["=" * 50, "AQUARIUM WATER QUALITY REPORT", "=" * 50, "", f"Water Type: {water_type}"]
    if sg:
        lines.append(f"Specific Gravity: {sg:.3f}")
    lines.extend(["", "-" * 50, "RESULTS", "-" * 50, ""])
    for r in results:
        icon = "✅" if r.status == "ok" else ("⚠️" if r.status == "warning" else "🚨")
        lines.append(f"{icon} {r.display_name}: {r.value:.2f} {r.unit}")
        for rec in r.recommendations:
            lines.append(f"   • {rec}")
    lines.extend(["-" * 50, "END OF REPORT", "-" * 50])
    return "\n".join(lines)

def render_result(result):
    icon = "✅" if result.status == "ok" else ("⚠️" if result.status == "warning" else "🚨")
    rgb = result.detected_color
    st.markdown(f"""
    <div class="result-card">
        <span class="color-swatch" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"></span>
        <strong>{result.display_name}</strong>: {result.value:.2f} {result.unit}
        <span class="status-{result.status}">{icon} {result.status.upper()}</span>
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
    
    if 'manual_colors' not in st.session_state:
        st.session_state.manual_colors = {}
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        water_type = st.selectbox("🌊 Water Type", ["Saltwater", "Freshwater"])
    with col2:
        sg = None
        if water_type == "Saltwater":
            sg = st.number_input("📏 Specific Gravity", 1.015, 1.035, 1.025, 0.001, "%.3f")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Place strip ON the color chart card (gray area on left edge)
        2. Take photo with good lighting
        3. Upload for automatic analysis
        
        **Note:** Gray/white pads indicate safe (0) values - this is normal!
        """)
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 10-in-1", "🧪 Ammonia", "🎨 Manual"])
    
    all_results = []
    
    with tab1:
        st.markdown("### Upload 10-in-1 Strip Photo")
        uploaded = st.file_uploader("Photo of strip on chart", type=['jpg', 'jpeg', 'png'], key="u1")
        
        if uploaded:
            img = safe_load_image(uploaded)
            if img is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(img, caption="Uploaded", use_container_width=True)
                with c2:
                    with st.spinner("🔍 Analyzing..."):
                        results, error = analyze_10in1_strip(img)
                    if error:
                        st.error(error)
                    elif results:
                        st.markdown("### Results")
                        for r in results:
                            cfg = PARAM_CONFIG.get(r["name"], {})
                            status = get_status(r["name"], r["value"], water_type)
                            recs = get_recommendations(r["name"], r["value"], status, water_type)
                            result = AnalysisResult(r["name"], cfg.get("display", r["name"]), r["value"], cfg.get("unit", ""), status, r["color"], r["confidence"], recs)
                            all_results.append(result)
                            render_result(result)
    
    with tab2:
        st.markdown("### Upload Ammonia Strip Photo")
        uploaded2 = st.file_uploader("Photo of ammonia strip", type=['jpg', 'jpeg', 'png'], key="u2")
        
        if uploaded2:
            img = safe_load_image(uploaded2)
            if img is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(img, caption="Uploaded", use_container_width=True)
                with c2:
                    with st.spinner("🔍 Analyzing..."):
                        r = analyze_ammonia_strip(img)
                    cfg = PARAM_CONFIG.get("Ammonia", {})
                    status = get_status("Ammonia", r["value"], water_type)
                    recs = get_recommendations("Ammonia", r["value"], status, water_type)
                    result = AnalysisResult("Ammonia", cfg.get("display", "Ammonia"), r["value"], cfg.get("unit", "ppm"), status, r["color"], r["confidence"], recs)
                    all_results.append(result)
                    st.markdown("### Result")
                    render_result(result)
    
    with tab3:
        st.markdown("### Manual Entry")
        params = ["Ammonia"] + STRIP_PARAMS
        sel = st.selectbox("Parameter:", params, format_func=lambda x: PARAM_CONFIG.get(x, {}).get("display", x))
        color = st.color_picker(f"Pick {sel} color", "#FFFFFF")
        if st.button(f"Analyze {sel}", type="primary"):
            rgb = hex_to_rgb(color)
            chart = AMMONIA_COLORS if sel == "Ammonia" else REFERENCE_COLORS.get(sel, {})
            val, conf = match_color_to_value(rgb, chart)
            cfg = PARAM_CONFIG.get(sel, {})
            status = get_status(sel, val, water_type)
            recs = get_recommendations(sel, val, status, water_type)
            result = AnalysisResult(sel, cfg.get("display", sel), val, cfg.get("unit", ""), status, rgb, conf, recs)
            st.session_state.manual_colors[sel] = result
            st.success(f"✅ {sel}: {val:.2f}")
        
        if st.session_state.manual_colors:
            st.markdown("### Manual Entries")
            for p, r in st.session_state.manual_colors.items():
                render_result(r)
                if st.button(f"Remove {p}", key=f"rm_{p}"):
                    del st.session_state.manual_colors[p]
                    st.rerun()
    
    st.markdown("---")
    st.markdown("## 📋 Summary")
    
    combined = all_results + list(st.session_state.manual_colors.values())
    
    if not combined:
        st.info("👆 Upload photos or use manual entry")
    else:
        statuses = [r.status for r in combined]
        if "danger" in statuses:
            st.error("## 🚨 ACTION REQUIRED")
        elif "warning" in statuses:
            st.warning("## ⚠️ ATTENTION NEEDED")
        else:
            st.success("## ✅ ALL GOOD!")
            st.balloons()
        
        priority_recs = []
        for r in combined:
            if r.status in ["danger", "warning"]:
                priority_recs.extend(r.recommendations)
        if priority_recs:
            st.markdown("### 🎯 Priority Actions")
            for rec in list(dict.fromkeys(priority_recs))[:5]:
                st.markdown(f"""<div class="priority-action">• {rec}</div>""", unsafe_allow_html=True)
        
        nh3 = next((r.value for r in combined if r.name == "Ammonia"), None)
        no2 = next((r.value for r in combined if r.name == "Nitrite"), None)
        no3 = next((r.value for r in combined if r.name == "Nitrate"), None)
        
        if nh3 is not None or no2 is not None:
            st.markdown("### 🔄 Nitrogen Cycle")
            nh3, no2, no3 = nh3 or 0, no2 or 0, no3 or 0
            if nh3 == 0 and no2 == 0 and no3 > 0:
                st.success("✅ Fully cycled")
            elif nh3 == 0 and no2 == 0 and no3 == 0:
                st.success("✅ All nitrogen parameters at safe levels")
            elif nh3 > 0 and no2 > 0:
                st.error("🚨 Active cycling")
            elif nh3 > 0:
                st.warning("⚠️ Ammonia present")
            elif no2 > 0:
                st.warning("⚠️ Nitrite present")
        
        if sg and water_type == "Saltwater":
            st.markdown("### 📏 Specific Gravity")
            if 1.024 <= sg <= 1.026:
                st.success(f"✅ SG {sg:.3f} - Ideal")
            elif 1.022 <= sg <= 1.027:
                st.info(f"ℹ️ SG {sg:.3f} - OK")
            else:
                st.warning(f"⚠️ SG {sg:.3f} - Adjust")
        
        st.markdown("---")
        report = generate_report(combined, water_type, sg)
        st.download_button("📥 Download Report", report, "water_report.txt", "text/plain", use_container_width=True)
    
    if combined or st.session_state.manual_colors:
        st.markdown("---")
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.manual_colors = {}
            st.rerun()
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#888;font-size:12px;'>🐠 Aquarium Analyzer v4.1 | SJ Wave Strips</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        if st.checkbox("Show details"):
            st.code(traceback.format_exc())
