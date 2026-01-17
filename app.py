import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_cie76

# --- Page Config for Mobile ---
st.set_page_config(page_title="Pocket Aquarist", page_icon="🐠", layout="centered")

# ==========================================
#    COMPUTER VISION & ANALYSIS ENGINE
# ==========================================
class WaterTestAnalyzer:
    def __init__(self):
        # --- Configuration for 10-in-1 (Strip ON Chart) ---
        self.CONFIG_MULTI = {
            # Search for strip in the left 5%-25% of the warped chart width
            'strip_x_pct': 0.15, 
            'rows': [
                # Y-positions based on your chart image
                {'name': 'Iron',        'y_pct': 0.258, 'values': [0, 5, 10, 25, 50, 100], 'unit': 'mg/L'},
                {'name': 'Copper',      'y_pct': 0.322, 'values': [0, 10, 30, 100, 200, 300], 'unit': 'mg/L'},
                {'name': 'Nitrate',     'y_pct': 0.386, 'values': [0, 10, 25, 50, 100, 250], 'unit': 'mg/L'},
                {'name': 'Nitrite',     'y_pct': 0.450, 'values': [0, 1, 5, 10], 'unit': 'mg/L'}, 
                {'name': 'Chlorine',    'y_pct': 0.514, 'values': [0, 0.8, 1.5, 3], 'unit': 'mg/L'},
                {'name': 'Hardness',    'y_pct': 0.578, 'values': [0, 25, 75, 150, 300], 'unit': 'mg/L'},
                {'name': 'Alkalinity',  'y_pct': 0.642, 'values': [0, 40, 80, 120, 180, 300], 'unit': 'mg/L'},
                {'name': 'Carbonate',   'y_pct': 0.706, 'values': [0, 40, 80, 120, 180, 300], 'unit': 'mg/L'},
            ],
            # Select appropriate pH row based on water mode
            'ph_fresh': {'name': 'pH', 'y_pct': 0.770, 'values': [6.4, 6.8, 7.2, 7.6, 8.0, 8.4], 'unit': ''},
            'ph_salt':  {'name': 'pH', 'y_pct': 0.838, 'values': [6.8, 7.2, 7.6, 8.0, 8.4, 9.0], 'unit': ''},
            # Reference block locations across the chart width
            'ref_x_start': 0.35, 'ref_x_end': 0.95
        }

        # --- Configuration for Ammonia (Strip NEXT TO Chart) ---
        self.CONFIG_AMMONIA = {
            'values': [0, 0.25, 0.5, 1, 3, 6], 'unit': 'ppm',
            # Location of reference blocks on the *warped chart*
            'ref_y_pct': 0.28, 
            'ref_x_start': 0.35, 'ref_x_end': 0.75
        }

    # --- Core CV Helper Functions ---
    def get_avg_color(self, img, x, y, radius=6):
        """Safe color sampling with bounds checking"""
        h, w = img.shape[:2]
        if x < radius or x > w-radius or y < radius or y > h-radius: return [0,0,0]
        mask = np.zeros((h, w), dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
        return cv2.mean(img, mask=mask)[:3]

    def match_color(self, target_bgr, ref_bgrs, values):
        """Matches colors using human perception standard (CIELAB DeltaE)"""
        if not ref_bgrs: return 0, 100.0
        # Convert BGR to LAB
        t_lab = rgb2lab(np.uint8([[target_bgr[::-1]]]))
        r_labs = rgb2lab(np.uint8([[r[::-1] for r in ref_bgrs]]))
        # Calculate perceptual difference
        diffs = [deltaE_cie76(t_lab[0][0], r) for r in r_labs[0]]
        idx = np.argmin(diffs)
        return values[idx], diffs[idx]

    def order_points(self, pts):
        """Orders coordinates: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def detect_and_warp_chart(self, img):
        """Finds the largest rectangle (the chart) and warps it flat."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        # Use adaptive thresholding to find edges even in varying light
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 10000: # Ensure it's big enough
                screenCnt = approx
                break
        
        if screenCnt is None: return None # Could not find chart

        # Warp perspective
        rect = self.order_points(screenCnt.reshape(4, 2))
        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
        height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
        dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (width, height))

    # --- Pipeline 1: 10-in-1 Processing (Strip ON Chart) ---
    def process_multi(self, img_file, mode):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        warped = self.detect_and_warp_chart(img)
        if warped is None: raise Exception("Could not detect the 10-in-1 chart boundary. Ensure the whole chart is visible.")
        
        h, w = warped.shape[:2]
        debug_img = warped.copy()
        cfg = self.CONFIG_MULTI
        
        rows = cfg['rows'].copy()
        rows.append(cfg['ph_fresh'] if mode == 'Freshwater' else cfg['ph_salt'])
        
        results = []
        # Strip is assumed to be placed on the left side designated area
        strip_x = int(w * cfg['strip_x_pct'])
        
        for row in rows:
            y = int(h * row['y_pct'])
            # 1. Sample Strip
            strip_color = self.get_avg_color(warped, strip_x, y)
            cv2.circle(debug_img, (strip_x, y), 10, (0,0,255), 3) # Red circle
            
            # 2. Sample References
            ref_colors = []
            vals = row['values']
            r_start, r_end = int(w*cfg['ref_x_start']), int(w*cfg['ref_x_end'])
            step = (r_end - r_start) / (len(vals)-1) if len(vals) > 1 else 0
            
            for i in range(len(vals)):
                rx = int(r_start + i*step)
                ref_colors.append(self.get_avg_color(warped, rx, y))
                cv2.circle(debug_img, (rx, y), 6, (255,255,0), 2) # Cyan circle
            
            # 3. Match
            val, _ = self.match_color(strip_color, ref_colors, vals)
            results.append({'Parameter': row['name'], 'Value': val, 'Unit': row['unit']})
            
        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), results

    # --- Pipeline 2: Ammonia Processing (Strip NEXT TO Chart) ---
    def process_ammonia(self, img_file):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        debug_img = img.copy()

        # A. Find Chart & Get Reference Colors
        warped_chart = self.detect_and_warp_chart(img)
        if warped_chart is None: raise Exception("Could not detect Ammonia chart boundary.")
        
        wh, ww = warped_chart.shape[:2]
        cfg = self.CONFIG_AMMONIA
        ref_y = int(wh * cfg['ref_y_pct'])
        ref_colors = []
        vals = cfg['values']
        r_start, r_end = int(ww*cfg['ref_x_start']), int(ww*cfg['ref_x_end'])
        step = (r_end - r_start) / (len(vals)-1)
        
        # Sample references from the clean, warped chart image
        for i in range(len(vals)):
            rx = int(r_start + i*step)
            ref_colors.append(self.get_avg_color(warped_chart, rx, ref_y))
            # Note: We can't easily draw these on the debug image as it's unwarped

        # B. Find Strip & Sample Pad Color in Original Image
        # 1. Convert to grayscale and threshold to find white objects
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use Otsu's thresholding for separating foreground (strip/chart) from background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        strip_cnt = None
        
        # 2. Filter contours to find the strip based on shape (long and thin)
        for c in cnts:
            if cv2.contourArea(c) < 500: continue # Ignore noise
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(max(w, h)) / min(w, h)
            
            # A test strip is very long and thin (high aspect ratio, e.g., > 5)
            # The chart is closer to a square (lower aspect ratio)
            if aspect_ratio > 4.0: 
                strip_cnt = c
                break
        
        if strip_cnt is None: raise Exception("Could not locate the test strip sitting next to the chart.")
        
        # 3. Locate the pad on the strip
        x, y, w, h = cv2.boundingRect(strip_cnt)
        cv2.rectangle(debug_img, (x,y), (x+w, y+h), (0,255,0), 2) # Draw box around found strip
        
        # Assumption based on user image: Strip is placed vertically, pad is at the top.
        # Sample the top 15% of the strip's bounding box area.
        pad_x = int(x + w / 2)
        pad_y = int(y + h * 0.15) 
        
        strip_color = self.get_avg_color(img, pad_x, pad_y, radius=10)
        cv2.circle(debug_img, (pad_x, pad_y), 12, (0,0,255), 3) # Red circle on pad

        # C. Match
        val, _ = self.match_color(strip_color, ref_colors, vals)
        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), [{'Parameter': 'Ammonia', 'Value': val, 'Unit': cfg['unit']}]

# ==========================================
#    INTERPRETATION & REPORT ENGINE
# ==========================================
def generate_report(data, water_mode, sg=None):
    report = []
    
    def get_val(name):
        row = next((item for item in data if item["Parameter"] == name), None)
        return float(row['Value']) if row else None

    # --- Critical Parameters ---
    nh3 = get_val('Ammonia')
    if nh3 is not None:
        if nh3 >= 0.5:
            report.append(("CRITICAL", f"Ammonia is DANGEROUS ({nh3} ppm).", 
                           "Perform an immediate 30-50% water change. Stop feeding. Add an ammonia detoxifier (e.g., Seachem Prime)."))
        elif nh3 > 0:
             report.append(("WARNING", f"Ammonia detected ({nh3} ppm).", 
                           "Ensure tank is fully cycled. Monitor daily. Reduce feeding."))
        else:
            report.append(("GOOD", "Ammonia is ideal (0 ppm).", ""))

    no2 = get_val('Nitrite')
    if no2 is not None:
        if no2 >= 1.0:
            report.append(("CRITICAL", f"Nitrite is DANGEROUS ({no2} ppm).", 
                           "Toxic to fish. Immediate water change required. Add detoxifier."))
        elif no2 > 0:
             report.append(("WARNING", f"Nitrite detected ({no2} ppm).", 
                           "Tank may be cycling. Monitor closely. Add aquarium salt if freshwater."))
        else:
            report.append(("GOOD", "Nitrite is ideal (0 ppm).", ""))

    # --- Important Parameters ---
    no3 = get_val('Nitrate')
    if no3 is not None:
        limit = 40 if water_mode == 'Freshwater' else 25
        if no3 > limit:
            report.append(("WARNING", f"Nitrate is high ({no3} ppm).", 
                           "Perform a partial water change (20%). Check filters for trapped debris. Reduce feeding frequency."))
        else:
            report.append(("GOOD", f"Nitrate is safe ({no3} ppm).", ""))

    ph = get_val('pH')
    if ph is not None:
        if water_mode == 'Saltwater':
            if ph < 7.8: report.append(("WARNING", f"pH is low ({ph}).", "Check Alkalinity (KH). Ensure good surface agitation for gas exchange."))
            elif ph > 8.5: report.append(("WARNING", f"pH is high ({ph}).", "Check for recent dosing errors. Ensure adequate aeration."))
            else: report.append(("GOOD", f"pH is ideal ({ph}).", ""))
        else: # Freshwater
            report.append(("INFO", f"pH is {ph}.", "Ideal range depends specifically on your fish species."))

    # --- Salinity (Manual Input) ---
    if water_mode == 'Saltwater' and sg is not None and sg > 1.000:
        if sg < 1.020:
            report.append(("WARNING", f"Salinity is low (SG {sg:.3f}).", "Top up evaporation with saltwater slowly over a day."))
        elif sg > 1.027:
            report.append(("WARNING", f"Salinity is high (SG {sg:.3f}).", "Remove some tank water and replace with fresh RO/DI water slowly."))
        else:
            report.append(("GOOD", f"Salinity is ideal (SG {sg:.3f}).", "Range 1.020 - 1.026 typically recommended."))

    return report

# ==========================================
#    STREAMLIT MOBILE WEB INTERFACE
# ==========================================
st.title("🌊 Pocket Aquarist")
st.write("Instant analysis and actionable advice for your fish tank.")

# 1. Settings & Manual Input
with st.container():
    st.subheader("1. Tank Settings")
    water_mode = st.radio("Water Type:", ["Saltwater", "Freshwater"], horizontal=True, label_visibility="collapsed")
    
    sg_input = None
    if water_mode == "Saltwater":
        sg_input = st.number_input("Specific Gravity (SG) Reading:", min_value=1.000, max_value=1.050, value=1.025, step=0.001, format="%.3f", help="Enter the reading from your refractometer or hydrometer.")

st.divider()

# 2. Photo Uploads
st.subheader("2. Upload Test Photos")
st.info("On mobile, tap 'Browse files' then select 'Take Photo'.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**10-in-1 Test**")
    st.image("https://i.imgur.com/8Z9Qx8A.png", width=100, caption="Strip ON Chart") # Placeholder visual cue
    img_multi = st.file_uploader("Upload 10-in-1", type=['jpg','png','jpeg'], key="multi_u")

with col2:
    st.markdown("**Ammonia Test**")
    # Visual cue showing strip next to chart
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="width: 20px; height: 80px; background-color: #eee; border: 1px solid #ccc; margin-right: 5px; display: flex; flex-direction: column; justify-content: flex-start; align-items: center;"><div style="width: 16px; height: 16px; background-color: yellow; margin-top: 5px;"></div></div>
        <div style="width: 60px; height: 80px; background-color: #ddd; border: 1px solid #999; display: flex; align-items: center; justify-content: center; font-size: 10px;">Chart</div>
    </div>
    <div style="text-align: center; font-size: 12px; color: #666;">Strip NEXT TO Chart<br>(Pad at top)</div>
    """, unsafe_allow_html=True)
    img_ammonia = st.file_uploader("Upload Ammonia", type=['jpg','png','jpeg'], key="amm_u")

st.divider()

# 3. Analysis Action
if st.button("Analyze Results Now", type="primary", use_container_width=True):
    analyzer = WaterTestAnalyzer()
    combined_data = []
    has_errors = False

    # Process 10-in-1
    if img_multi:
        with st.spinner("Analyzing 10-in-1 Strip..."):
            try:
                debug_m, res_m = analyzer.process_multi(img_multi, water_mode)
                combined_data.extend(res_m)
                with st.expander("View 10-in-1 Scan (Debug)"):
                    st.image(debug_m, caption="Red=Strip Sample, Cyan=References", use_column_width=True)
            except Exception as e:
                st.error(f"10-in-1 Error: {e}")
                has_errors = True

    # Process Ammonia
    if img_ammonia:
        with st.spinner("Analyzing Ammonia Strip..."):
            try:
                debug_a, res_a = analyzer.process_ammonia(img_ammonia)
                combined_data.extend(res_a)
                with st.expander("View Ammonia Scan (Debug)"):
                    st.image(debug_a, caption="Green Box=Strip Found, Red Circle=Pad Sample", use_column_width=True)
            except Exception as e:
                st.error(f"Ammonia Error: {e}. Ensure strip is next to chart and background is not white.")
                has_errors = True

    # Generate Comprehensive Report
    if combined_data or (water_mode == 'Saltwater' and sg_input and sg_input != 1.025):
        st.header("📋 Water Quality Report")
        
        advice = generate_report(combined_data, water_mode, sg_input)
        
        if not advice and not has_errors:
             st.success("Everything looks good based on the inputs provided!", icon="✅")

        # Display Alerts with actionable steps
        for status, msg, action in advice:
            if status == "CRITICAL":
                st.error(f"**{msg}**\n\n🛠️ *Action Required:* {action}", icon="🚨")
            elif status == "WARNING":
                st.warning(f"**{msg}**\n\n⚠️ *Attention Needed:* {action}", icon="⚠️")
            elif status == "GOOD":
                st.success(f"**{msg}** {action}", icon="✅")
            else:
                 st.info(f"**{msg}** {action}", icon="ℹ️")

        # Measurements Table
        if combined_data:
            st.subheader("Measurements")
            df = pd.DataFrame(combined_data)
            # Format value column nicely
            df['Reading'] = df.apply(lambda x: f"{x['Value']} {x['Unit']}", axis=1)
            st.dataframe(df[['Parameter', 'Reading']], hide_index=True, use_container_width=True)
            
    elif not has_errors:
        st.warning("Please upload at least one test image or enter a specific gravity reading to get a report.")
