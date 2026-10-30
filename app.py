import os
import streamlit as st
import torch
# 1. Disable PyTorch safe weights checking for Ultralytics models
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# 2. Register all internal YOLO modules into PyTorch safe globals
try:
    from ultralytics.nn import tasks
    torch.serialization.add_safe_globals([tasks.DetectionModel])
except Exception:
    pass
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Prevent Ultralytics from executing cv2 GUI functions
import os
os.environ["YOLO_VERBOSE"] = "False"

st.title("Smart Waste Detection App")
st.write("Upload an image to classify waste items.")

# 2. Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 3. File Uploader & Detection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Waste"):
        results = model(image)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detected Waste", use_container_width=True)

# Waste category facts and information dictionary
waste_info = {
    'Fabrics': {
        'info': "Textiles and fabrics account for a significant portion of landfill waste. Many natural fibers take months to decompose, while synthetic fabrics can take hundreds of years.",
        'tip': "Donate wearable clothes or drop them off at local textile recycling bins instead of throwing them away."
    },
    'Glass Bottle': {
        'info': "Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity.",
        'tip': "Rinse thoroughly and remove caps/lids before placing them in glass recycling containers."
    },
    'Metal': {
        'info': "Recycling metals like aluminum and steel saves up to 95% of the energy required to make them from raw materials.",
        'tip': "Crush aluminum cans to save space in your recycling bin and ensure they are clean."
    },
    'Nylon': {
        'info': "Nylon is a petroleum-based synthetic polymer that is extremely durable and non-biodegradable, making it a major pollutant if discarded improperly.",
        'tip': "Try to repurpose nylon items or check for specialized textile recycling facilities that handle synthetic blends."
    },
    'Organic Waste': {
        'info': "Organic waste (food scraps, yard waste) produces potent methane gas when trapped in oxygen-deprived landfills.",
        'tip': "Compost organic waste at home or use local community composting programs to turn it into nutrient-rich soil."
    },
    'Plastic Bottle': {
        'info': "Most plastic bottles are made of PET (polyethylene terephthalate) and can take over 450 years to break down in the environment.",
        'tip': "Empty liquids, flatten bottles to save space, and check local guidelines on whether caps should stay on."
    }
}

st.title("🌱 Smart Waste Detection & Classification App")
st.write("Upload an image of waste items to detect, classify, and learn proper disposal tips using your custom YOLOv8 model!")

# Sidebar for Developer Disclaimer and Feedback
with st.sidebar:
    st.header("ℹ️ Developer & Project Info")
    st.markdown("""
    **Developer Disclaimer:**
    This application is developed as an AI-powered educational and experimental tool for waste classification (Group 4 Project). 
    
    While trained on custom datasets, predictions and bounding box accuracies may vary. Always verify local municipal waste management guidelines before disposing of hazardous or specialized materials.
    """)
    
    st.markdown("---")
    st.header("💬 Feedback")
    st.write("Found a misclassification or have suggestions?")
    
    with st.form("feedback_form"):
        user_name = st.text_input("Your Name / Email (Optional)")
        feedback_text = st.text_area("Your Feedback")
        submit_button = st.form_submit_button("Submit Feedback")
        
        if submit_button:
            if feedback_text.strip():
                st.success("Thank you! Your feedback has been recorded.")
            else:
                st.warning("Please enter some feedback before submitting.")

# Main App Interface
uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Run Prediction'):
        with st.spinner('Analyzing image with YOLOv8...'):
            results = model(np.array(image))
            res_plotted = results[0].plot()
            st.image(res_plotted, caption='Detected Waste Results', use_column_width=True)
            
            detected_classes = set()
            boxes = results[0].boxes
            if boxes is not None:
                for cls_id in boxes.cls:
                    class_name = model.names[int(cls_id)]
                    detected_classes.add(class_name)
            
            if detected_classes:
                st.markdown("### ♻️ Insights & Disposal Tips for Detected Items")
                for c_name in detected_classes:
                    if c_name in waste_info:
                        with st.expander(f"📌 About: {c_name}", expanded=True):
                            st.info(waste_info[c_name]['info'])
                            st.success(f"**Eco-Tip:** {waste_info[c_name]['tip']}")
            else:
                st.info("No distinct waste classes were detected with high confidence in this image.")
