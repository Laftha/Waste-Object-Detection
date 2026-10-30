import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Page Configuration
st.set_page_config(
    page_title="EcoSort - Smart Waste Detector",
    page_icon="♻️",
    layout="wide"
)
# Sidebar Information
with st.sidebar:
    st.image("Screenshot 2026-10-30 011304.jpg", width=120)
    st.title("♻️ About EcoSort")
    st.markdown("""
    **Group Name:** GROUP 4
    
    **Project Description:**  
    A smart waste classification capstone project designed to help identify recyclable materials and provide disposal insights.
    """)
    
    st.markdown("---")
    st.subheader("🎯 Covered Classes")
    st.markdown("""
    * 🧴 **Plastic Bottles**
    * 🛍️ **Nylon / Soft Plastics**
    * 🍾 **Glass Bottles**
    * 👕 **Fabric / Textiles**
    * 🥫 **Metal Waste**
    """)
    
    st.markdown("---")
    st.caption("Capstone Project Demo")
# Header Layout with Title and Logo side-by-side
col_title, col_logo = st.columns([5, 1])

with col_logo:
    st.image("Screenshot 2026-10-30 011304.jpg", width=150)
# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Header Information
st.title("♻️ EcoSort AI: Smart Waste Detector")
st.markdown("Upload an image or take a live picture of waste to identify categories and view recycling tips.")

# Recycling Knowledge Base
RECYCLING_INFO = {
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
    
    # Correctly select between file uploader and camera input using 'image_file'
    image_file = uploaded_file if input_type == "📁 Upload File" else camera_file

    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Run AI Classification"):
            with st.spinner("Analyzing image..."):
                # 1. Run inference using image_file
                results = model.predict(image_file, conf=0.25)
                r = results[0]
                
                # 2. Render bounding boxes
                im_array = r.plot()
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detection Result", use_container_width=True)
                
                # 3. Render Insights and Classes
                classes = [model.names[int(box.cls[0])] for box in r.boxes]
                if classes:
                    detected_item = classes[0].lower()
                    st.success(f"**Detected Items:** {', '.join(set(classes))}")
                    st.markdown("### 💡 Recycling & Disposal Insights")
                    
                    if "plastic" in detected_item or "bottle" in detected_item:
                        st.info("**Category:** Plastic Waste\n\n**Action:** Rinse thoroughly and place in the **Recycling Bin**. Make sure caps are screwed back on.")
                    elif "nylon" in detected_item:
                        st.info("**Category:** Nylon / Soft Plastic\n\n**Action:** Keep clean and dry. Check local store drop-off points for recycling.")
                    elif "glass" in detected_item:
                        st.info("**Category:** Glass Waste\n\n**Action:** Rinse and place in the **Glass Recycling Bin**.")
                    else:
                        st.info(f"**Category:** {detected_item.title()}\n\n**Action:** Follow local municipal recycling guidelines.")
                else:
                    st.warning("No waste items detected clearly. Try taking a photo closer to the object.")
    else:
        st.info("👈 Upload an image or take a photo in Step 1 to unlock AI classification.")

st.markdown("---")

# Footer Disclaimer
st.caption("⚠️ **Disclaimer:** EcoSort AI provides automated waste classification for educational and demonstration purposes. Model predictions should be verified independently before making critical waste disposal decisions.")
