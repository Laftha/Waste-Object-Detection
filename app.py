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
    
    if image_input is not None:
        st.image(image_input, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Run AI Classification"):
            with st.spinner("Analyzing image..."):
                # 1. Run inference
                results = model.predict(image_input, conf=0.25)
                r = results[0]
                
                # 2. Render bounding boxes image
                im_array = r.plot()
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detection Result", use_container_width=True)
                
                # 3. PASTE THE INSIGHTS CODE HERE:
                classes = [model.names[int(box.cls[0])] for box in r.boxes]
                
                if classes:
                    detected_item = classes[0].lower()
                    st.success(f"**Detected Items:** {', '.join(set(classes))}")
                    
                    st.markdown("### 💡 Recycling & Disposal Insights")
                    
                    if "plastic" in detected_item or "bottle" in detected_item:
                        st.info("**Category:** Plastic Waste\n\n**Action:** Rinse thoroughly and place in the **Recycling Bin**. Make sure caps are screwed back on.")
                    elif "paper" in detected_item or "cardboard" in detected_item:
                        st.info("**Category:** Paper / Cardboard\n\n**Action:** Flatten boxes and ensure paper is dry before placing in **Paper Recycling**.")
                    elif "organic" in detected_item or "food" in detected_item or "fruit" in detected_item:
                        st.info("**Category:** Organic Waste\n\n**Action:** Ideal for composting. Place in the **Organic / Compost Bin**.")
                    elif "metal" in detected_item or "can" in detected_item:
                        st.info("**Category:** Metal\n\n**Action:** Empty leftover liquid, rinse lightly, and drop into **Metal Recycling**.")
                    else:
                        st.info(f"**Category:** {detected_item.capitalize()}\n\n**Action:** Check local municipal guidelines for proper disposal.")
                else:
                    st.warning("No waste items detected clearly. Try taking a photo closer to the object.")
    else:
        st.info("👈 Upload an image or take a photo in Step 1 to unlock AI classification.")
        
st.markdown("---")
col_left, col_center, col_right = st.columns([2, 1, 2])

with col_center:
    try:
        st.image("Screenshot 2026-10-30 011304.jpg", width=160)
    except Exception:
        # Fallback if filename differs
        pass
# Footer Disclaimer
st.caption("⚠️ **Disclaimer:** EcoSort AI provides automated waste classification for educational and demonstration purposes. Model predictions should be verified independently before making critical waste disposal decisions.")
