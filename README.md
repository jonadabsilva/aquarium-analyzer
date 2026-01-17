# 🐠 Aquarium Water Analyzer

A mobile-friendly web application for analyzing SJ Wave aquarium test strips using AI-powered color detection.

## Features

- **10-in-1 Test Strip Analysis**: Iron, Copper, Nitrate, Nitrite, Chlorine, Hardness, Alkalinity, Carbonate, pH
- **Ammonia Strip Analysis**: Separate ammonia test strip support
- **Specific Gravity Input**: For saltwater tanks
- **Freshwater & Saltwater Support**: Different reference ranges for each
- **Comprehensive Reports**: Detailed analysis with recommendations
- **Mobile-Optimized**: Works great on phones
- **Downloadable Reports**: Export your results

## Quick Start

### Option 1: Run Locally

```bash
# Clone or download this folder
cd aquarium_analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_enhanced.py

# Open in browser (usually http://localhost:8501)
```

### Option 2: Deploy to Streamlit Cloud (FREE)

1. **Create a GitHub Repository**
   - Go to https://github.com/new
   - Name it `aquarium-analyzer`
   - Upload all files from this folder

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app_enhanced.py`
   - Click "Deploy"

3. **Access from Phone**
   - Open the provided URL on your phone
   - Add to home screen for app-like experience

### Option 3: Deploy to Render (FREE)

1. Push code to GitHub
2. Go to https://render.com
3. Create new "Web Service"
4. Connect your GitHub repo
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `streamlit run app_enhanced.py --server.port $PORT --server.address 0.0.0.0`

## How to Use

### Step 1: Setup
1. Open the app on your phone
2. Select **Saltwater** or **Freshwater**
3. For saltwater: enter your refractometer reading (specific gravity)

### Step 2: Take Photos
1. Place your test strip next to the color chart
2. Ensure good natural lighting (avoid shadows)
3. Take a clear photo

### Step 3: Upload & Sample
1. Upload your 10-in-1 strip image
2. For each parameter:
   - Select the parameter from dropdown
   - Use the color picker to eyedrop the color from your strip pad
   - OR enter coordinates to sample from the image
   - Click "Set color"

3. Upload your ammonia strip image
4. Sample the ammonia pad color (it's at the end of the strip)

### Step 4: Analyze
1. Click "🔬 Analyze Water Parameters"
2. Review your results
3. Follow any recommendations
4. Download the report if needed

## Understanding Results

### Status Indicators
- ✅ **OK** - Parameter is within healthy range
- ⚠️ **Warning** - Needs attention soon
- 🚨 **Danger** - Immediate action required

### For Saltwater Tanks
| Parameter | Ideal Range |
|-----------|-------------|
| Ammonia | 0 ppm |
| Nitrite | 0 ppm |
| Nitrate | 0-25 ppm |
| pH | 8.0-8.4 |
| Alkalinity | 120-180 ppm |
| Specific Gravity | 1.024-1.026 |

### For Freshwater Tanks
| Parameter | Ideal Range |
|-----------|-------------|
| Ammonia | 0 ppm |
| Nitrite | 0 ppm |
| Nitrate | 0-40 ppm |
| pH | 6.8-7.6 |
| Alkalinity | 80-120 ppm |

## Troubleshooting

### Colors don't match
- Ensure consistent lighting when taking photos
- Take photo in natural daylight
- Wait the correct time before reading (15 seconds for most, 60 seconds for nitrate/nitrite)
- Don't shake water off the strip

### App not loading
- Check internet connection
- Try refreshing the page
- Clear browser cache

### Inaccurate readings
- The strip may have expired
- Keep strips in a cool, dry place
- Don't touch the test pads with fingers

## Files

```
aquarium_analyzer/
├── app_enhanced.py      # Main application (enhanced version)
├── app.py               # Basic version
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Details

- **Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Color Matching**: Weighted Euclidean distance in RGB space
- **Interpolation**: Linear interpolation between reference colors

## Tips for Best Results

1. **Lighting**: Use natural daylight, avoid artificial lights that can alter colors
2. **Timing**: Read the strip at the correct time (15s or 60s per the instructions)
3. **Consistency**: Take photos at the same angle and distance each time
4. **Storage**: Keep test strips sealed and dry
5. **Verification**: For critical decisions, verify with liquid test kits

## Contributing

Feel free to fork and improve! Issues and PRs welcome.

## Disclaimer

This app provides estimates based on color analysis. Always verify with professional testing equipment for critical livestock decisions. The app creators are not responsible for any losses due to inaccurate readings.

---

Made with ❤️ for aquarium hobbyists
