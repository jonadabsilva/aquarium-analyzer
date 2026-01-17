import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_cie76

st.set_page_config(page_title="Pocket Aquarist v2", page_icon="🐠", layout="centered")

# ==========================================
#    ROBUST COMPUTER VISION ENGINE
# ==========================================
class WaterTestAnalyzer:
    def __init__(self):
        # 10-in-1 Configuration
        self.CONFIG_MULTI = {
            'strip_x_pct': 0.15, 
            'rows': [
                {'name': 'Iron', 'y_pct': 0.258, 'values': [0, 5, 10, 25, 50, 100], 'unit': 'mg/L'},
                {'name': 'Copper', 'y_pct': 0.322, 'values': [0, 10, 30, 100, 200, 300], 'unit': 'mg/L'},
                {'name': 'Nitrate', 'y_pct': 0.386, 'values': [0, 10, 25, 50, 100, 250], 'unit': 'mg/L'},
                {'name': 'Nitrite', 'y_pct': 0.450, 'values': [0, 1, 5, 10], 'unit': 'mg/L'}, 
                {'name': 'Chlorine', 'y_pct': 0.514, 'values': [0, 0.8, 1.5, 3], 'unit': 'mg/L'},
                {'name': 'Hardness', 'y_pct': 0.578, 'values': [0, 25, 75, 150, 300], 'unit': 'mg/L'},
                {'name': 'Alkalinity', 'y_pct': 0.642, 'values': [0, 40, 80, 120, 180, 300], 'unit': 'mg/L'},
                {'name': 'Carbonate', 'y_pct': 0.706, 'values': [0, 40, 80, 120, 180, 300], 'unit': 'mg/L'},
            ],
            'ph_fresh': {'name': 'pH', 'y_pct': 0.770, 'values': [6.4, 6.8, 7.2, 7.6, 8.0, 8.4], 'unit': ''},
            'ph_salt':  {'name': 'pH', 'y_pct': 0.838, 'values': [6.8, 7.2, 7.6, 8.0, 8.4, 9.0], 'unit': ''},
            'ref_x_start': 0.35, 'ref_x_end': 0.95
        }

        # Ammonia Configuration
        self.CONFIG_AMMONIA = {
            'values': [0, 0.25, 0.5, 1, 3, 6], 'unit': 'ppm',
            'ref_y_pct': 0.28, 
            'ref_x_start': 0.35, 'ref_x_end': 0.75
        }

    # --- NEW: Automatic White Balance ---
    def correct_white_balance(self, img, reference_mask=None):
        """
        Adjusts the image colors so the 'reference_mask' area (the chart) becomes true white/gray.
        This removes yellow tint from indoor lighting.
        """
        result = img.copy()
        if reference_mask is None:
            # If no mask, assume the center of the image is the reference (fallback)
            h, w = img.shape[:2]
            reference_mask = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(reference_mask, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), 255, -1)

        # Calculate average color of the reference area (the white chart)
        mean_bgr = cv2.mean(img, mask=reference_mask)[:3]
        
        # Calculate scaling factors to make it neutral gray (128, 128, 128) or maintain brightness
        # We target the average brightness of the detected white area to avoid darkening
        avg_brightness = (mean_bgr[0] + mean_bgr[1] + mean_bgr[2]) / 3
        
        b_scale = avg_brightness / (mean_bgr[0] + 1e-6)
        g_scale = avg_brightness / (mean_bgr[1] + 1e-6)
        r_scale = avg_brightness / (mean_bgr[2] + 1e-6)
        
        # Apply scaling
        result = result.astype(np.float32)
        result[:, :, 0] *= b_scale
        result[:, :, 1] *= g_scale
        result[:, :, 2] *= r_scale
        
        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # --- Helper Functions ---
    def get_avg_color(self, img, x, y, radius=6):
        h, w = img.shape[:2]
        # Safety bounds
        if x < 0 or x >= w or y < 0 or y >= h: return [0,0,0]
        
        mask = np.zeros((h, w), dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
        return cv2.mean(img, mask=mask)[:3]

    def match_color(self, target_bgr, ref_bgrs, values):
        if not ref_bgrs: return 0, 100.0
        t_lab = rgb2lab(np.uint8([[target_bgr[::-1]]]))
        r_labs = rgb2lab(np.uint8([[r[::-1] for r in ref_bgrs]]))
        diffs = [deltaE_cie76(t_lab[0][0], r) for r in r_labs[0]]
        idx = np.argmin(diffs)
        return values[idx], diffs[idx]

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def detect_chart_contour(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 10000:
                return approx.reshape(4, 2)
        return None

    # --- 10-in-1 Pipeline ---
    def process_multi(self, img_file, mode):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        chart_pts = self.detect_chart_contour(img)
        if chart_pts is None: raise Exception("Chart not found. Keep camera parallel.")
        
        # 1. Warp
        rect = self.order_points(chart_pts)
        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
        height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
        dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect, dst), (width, height))
        
        # 2. White Balance (Using the empty space at bottom left of chart as reference)
        # Create a mask for a known white area on this specific chart (bottom left corner approx)
        wb_mask = np.zeros(warped.shape[:2], dtype="uint8")
        cv2.rectangle(wb_mask, (int(width*0.05), int(height*0.85)), (int(width*0.25), int(height*0.95)), 255, -1)
        warped = self.correct_white_balance(warped, wb_mask)
        
        debug_img = warped.copy()
        cfg = self.CONFIG_MULTI
        rows = cfg['rows'].copy()
        rows.append(cfg['ph_fresh'] if mode == 'Freshwater' else cfg['ph_salt'])
        
        results = []
        strip_x = int(width * cfg['strip_x_pct'])
        
        for row in rows:
            y = int(height * row['y_pct'])
            strip_color = self.get_avg_color(warped, strip_x, y)
            cv2.circle(debug_img, (strip_x, y), 8, (0,0,255), 2)
            
            ref_colors = []
            vals = row['values']
            r_start, r_end = int(width*cfg['ref_x_start']), int(width*cfg['ref_x_end'])
            step = (r_end - r_start) / (len(vals)-1) if len(vals) > 1 else 0
            
            for i in range(len(vals)):
                rx = int(r_start + i*step)
                ref_colors.append(self.get_avg_color(warped, rx, y))
                cv2.circle(debug_img, (rx, y), 5, (255,100,0), 1)
            
            val, _ = self.match_color(strip_color, ref_colors, vals)
            results.append({'Parameter': row['name'], 'Value': val, 'Unit': row['unit']})
            
        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), results

    # --- IMPROVED Ammonia Pipeline ---
    def process_ammonia(self, img_file):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 1. Find Chart
        chart_pts = self.detect_chart_contour(img)
        if chart_pts is None: raise Exception("Could not find the Chart.")
        
        # 2. White Balance
        # Create mask of the chart to use as white reference
        h, w = img.shape[:2]
        chart_mask = np.zeros((h, w), dtype="uint8")
        cv2.fillPoly(chart_mask, [chart_pts.astype(int)], 255)
        # Correct the WHOLE image colors based on the chart white
        img_corrected = self.correct_white_balance(img, chart_mask)
        debug_img = img_corrected.copy()
        
        # 3. Get References from Chart (Warp from corrected image)
        rect = self.order_points(chart_pts)
        dst = np.array([[0,0], [600,0], [600,800], [0,800]], dtype="float32") # Normalize size
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_chart = cv2.warpPerspective(img_corrected, M, (601, 801))
        
        cfg = self.CONFIG_AMMONIA
        ref_colors = []
        # Sample standardized locations on the warped chart
        ref_y = int(801 * cfg['ref_y_pct'])
        r_start, r_end = int(601*cfg['ref_x_start']), int(601*cfg['ref_x_end'])
        step = (r_end - r_start) / (len(cfg['values'])-1)
        for i in range(len(cfg['values'])):
            rx = int(r_start + i*step)
            ref_colors.append(self.get_avg_color(warped_chart, rx, ref_y))

        # 4. Find Strip (Advanced Filtering)
        # Mask OUT the chart so we don't find it again
        search_area = cv2.bitwise_not(chart_mask)
        # Focus on bright objects (the strip is white)
        gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
        # High threshold to find white plastic vs black mat
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=search_area)
        
        # Morphological Open to remove "speckles" from yoga mat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        cnts, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        strip_rect = None
        best_area = 0
        
        for c in cnts:
            # Filter by area (too small = noise)
            area = cv2.contourArea(c)
            if area < 500: continue
            
            # Filter by shape (Strip is long and thin)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(max(w,h)) / min(w,h)
            
            # Draw rejected contours in RED for debug
            cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 1) 
            
            if aspect_ratio > 3.0 and area > best_area:
                best_area = area
                strip_rect = (x, y, w, h)
                cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2) # Found strip in GREEN

        if strip_rect is None:
            return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), [], "Strip Not Found"
            
        # 5. Sample Pad
        sx, sy, sw, sh = strip_rect
        # Heuristic: Pad is usually at the "top" relative to text or just at one end.
        # We'll sample 15% from the top of the bounding rect.
        # User Instruction: "Place strip vertical, pad at top"
        pad_x = int(sx + sw/2)
        pad_y = int(sy + sh * 0.15) 
        
        strip_color = self.get_avg_color(img_corrected, pad_x, pad_y, radius=10)
        cv2.circle(debug_img, (pad_x, pad_y), 10, (255, 0, 255), 3) # Magenta circle on pad

        # 6. Match
        val, dist = self.match_color(strip_color, ref_colors, cfg['values'])
        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), [{'Parameter': 'Ammonia', 'Value': val, 'Unit': 'ppm'}], None

# ==========================================
#    UI & REPORTING
# ==========================================
st.title("🌊 Pocket Aquarist v2.0")
st.markdown("### Robust Color & Strip Detection")

with st.expander("Settings", expanded=True):
    water_mode = st.radio("Water Type", ["Saltwater", "Freshwater"], horizontal=True)
    if water_mode == "Saltwater":
        sg = st.number_input("Specific Gravity", 1.000, 1.050, 1.025, 0.001, format="%.3f")
    else:
        sg = None

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 1. Multi-Test")
    img_multi = st.file_uploader("Upload 10-in-1", type=['jpg','png'], key="m")
with col2:
    st.markdown("#### 2. Ammonia")
    st.caption("Place strip **vertically** next to chart. **Pad at TOP**.")
    img_amm = st.file_uploader("Upload Ammonia", type=['jpg','png'], key="a")

if st.button("Analyze", type="primary"):
    analyzer = WaterTestAnalyzer()
    data = []
    
    # Analyze Multi
    if img_multi:
        try:
            dimg, res = analyzer.process_multi(img_multi, water_mode)
            data.extend(res)
            st.image(dimg, caption="Multi-Test Analysis", use_column_width=True)
        except Exception as e:
            st.error(f"Multi-Test Error: {e}")

    # Analyze Ammonia
    if img_amm:
        try:
            dimg, res, err = analyzer.process_ammonia(img_amm)
            st.image(dimg, caption="Ammonia Analysis (Green=Strip, Red=Ignored Noise)", use_column_width=True)
            if err:
                st.error(f"Could not localize strip: {err}. Check the 'Red' contours in image above.")
            else:
                data.extend(res)
        except Exception as e:
            st.error(f"Ammonia Error: {e}")

    # Report
    if data:
        st.divider()
        st.subheader("Results")
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True)
        
        # Ammonia Logic
        amm_row = next((x for x in data if x['Parameter'] == 'Ammonia'), None)
        if amm_row:
            val = amm_row['Value']
            if val == 0:
                st.success("✅ Ammonia is Safe (0 ppm)")
            elif val < 0.5:
                st.warning(f"⚠️ Ammonia Detected ({val} ppm). Monitor closely.")
            else:
                st.error(f"🚨 Ammonia Critical ({val} ppm). Water change immediately.")
