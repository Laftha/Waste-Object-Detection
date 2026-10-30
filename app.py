import streamlit as st
from PIL import Image
import numpy as np
import os

# Set page configuration for wide layout and custom styling
st.set_page_config(
    page_title="EcoSort - Smart Waste Detector",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .group-badge {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 15px;
        border: 1px solid #C8E6C9;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

os.environ["YOLO_VERBOSE"] = "False"

# Sidebar Watermark & Info
with st.sidebar:
    st.markdown("<div class='group-badge'>👥 Developed by Group 4</div>", unsafe_allow_html=True)
    st.title("About EcoSort")
    st.info("EcoSort leverages AI computer vision to instantly classify waste items and promote proper recycling habits.")
    st.markdown("---")
    st.markdown("### Supported Categories")
    st.write("• Fabrics\n• Glass Bottles\n• Metals\n• Nylon\n• Organic Waste\n• Plastic Bottles")
    st.markdown("---")
    st.caption("🎓 Capstone / Group Project")

# Header Section with Group Watermark
st.markdown("<div style='text-align: center;'><span class='group-badge'>📌 Group 4 Project Presentation</span></div>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>♻️ EcoSort AI: Smart Waste Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload an image or take a live picture of waste to identify categories and view recycling tips.</p>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("best.pt")

try:
    model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Error loading AI model weights: {e}")

# Knowledge Base with Detailed Eco-Metrics
waste_info = {
    'Fabrics': {
        'icon': '👕',
        'decomposition': 'Up to 200 years (synthetic)',
        'info': "Textiles account for over 92 million tons of global landfill waste annually.",
        'tip': "Donate wearable clothes to local charities or take unwearable textiles to specialized fabric recycling bins.",
        'impact': "Recycling 1kg of cotton saves up to 20,000 liters of water."
    },
    'Glass Bottle': {
        'icon': '🍾',
        'decomposition': 'Over 1,000,000 years',
        'info': "Glass is 100% endlessly recyclable without any loss in quality or purity.",
        'tip': "Rinse out liquids thoroughly and remove lids/caps before placing in glass recycling bins.",
        'impact': "Recycling one glass bottle saves enough energy to power a 100-watt bulb for 4 hours."
    },
    'Metal': {
        'icon': '🥫',
        'decomposition': '50 - 500 years',
        'info': "Aluminum cans are infinitely recyclable and can return to store shelves in just 60 days.",
        'tip': "Rinse food residue out. Metal cans are fully accepted in standard recycling bins.",
        'impact': "Recycling aluminum uses 95% less energy than producing new metal from raw bauxite."
    },
    'Nylon': {
        'icon': '🛍️',
        'decomposition': '30 - 40 years',
        'info': "Nylon is a synthetic plastic derivative that breaks down into persistent microplastics.",
        'tip': "Reuse bags as many times as possible or deposit them at retail plastic film drop-off points.",
        'impact': "Reusing a bag just 4 times significantly offsets its carbon footprint."
    },
    'Organic Waste': {
        'icon': '🍎',
        'decomposition': '2 - 6 weeks',
        'info': "When organic waste decomposes without oxygen in landfills, it generates methane—a potent greenhouse gas.",
        'tip': "Compost food scraps at home or place them in municipal green waste bins.",
        'impact': "Composting enriches soil naturally and reduces greenhouse gas emissions."
    },
    'Plastic Bottles': {
        'icon': '🧴',
        'decomposition': '450 years',
        'info': "Over 1 million plastic bottles are purchased every minute worldwide, with low recycling rates.",
        'tip': "Empty, rinse, and flatten bottles to save bin space before recycling.",
        'impact': "Recycling 1 ton of plastic saves 5,774 kWh of electricity and 16.3 barrels of oil."
    }
}

# Main Layout: Two Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Provide Waste Image")
    input_type = st.radio("Choose Input Method:", ["📁 Upload File", "📸 Use Camera"], horizontal=True)
    
    uploaded_file = None
    camera_file = None
    
    if input_type == "📁 Upload File":
        uploaded_file = st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png", "JPG", "JPEG"])
    else:
        st.markdown("""
            <style>
            div[data-testid="stCameraInput"] video {
                transform: scaleX(1) !important;
                object-fit: cover !important;
            }
            </style>
        """, unsafe_allow_html=True)
        camera_file = st.camera_input("Take a snapshot of the waste item")

with col2:
    st.subheader("2. AI Analysis & Insights")
    
    # Check directly if a file has been uploaded or captured
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file).convert("RGB")
        st.image(image_input, caption="Selected Waste Image", use_container_width=True)
        
        if st.button("🔍 Run AI Classification"):
            with st.spinner("Analyzing image features..."):
                results = model.predict(image_input, save=False, conf=0.40)
                r = results[0]
                im_array = r.plot()
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detection Result", use_container_width=True)
                
                # Show detected classes
                classes = [model.names[int(box.cls[0])] for box in r.boxes]
                if classes:
                    st.success(f"Detected: {', '.join(set(classes))}")
                else:
                    st.warning("No waste detected. Try another angle.")
                    
    elif camera_file is not None:
        image_input = Image.open(camera_file).convert("RGB")
        st.image(image_input, caption="Captured Camera Image", use_container_width=True)
        
        if st.button("🔍 Run AI Classification"):
            with st.spinner("Analyzing image features..."):
                results = model.predict(image_input, save=False, conf=0.25)
                r = results[0]
                im_array = r.plot()
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detection Result", use_container_width=True)
                
                classes = [model.names[int(box.cls[0])] for box in r.boxes]
                if classes:
                    st.success(f"Detected: {', '.join(set(classes))}")
                else:
                    st.warning("No waste detected. Try another angle.")
    else:
        st.info("👈 Upload an image or take a photo in Step 1 to unlock AI classification.")

# Footer Disclaimer
st.markdown("""
    <style>
    /* Subtle centered watermark background */
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95)), 
                          url("YOUR_LOGO_URL_HERE");
        background-size: 300px;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)
st.caption("⚠️ **Disclaimer:** EcoSort AI provides automated waste classification for educational and demonstration purposes. Model predictions should be verified independently before making critical waste disposal decisions.")
