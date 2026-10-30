import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

# Prevent Ultralytics from running GUI functions
os.environ["YOLO_VERBOSE"] = "False"

st.title("Smart Waste Detection App")
st.write("Upload an image to classify waste items.")

# Load trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Waste information dictionary
waste_info = {
    'Fabric': {
        'info': "Textiles and fabrics account for a significant portion of landfill waste.",
        'tip': "Donate wearable clothes or drop them off at local textile recycling bins."
    },
    'Glass Bottle': {
        'info': "Glass is 100% recyclable and can be recycled endlessly.",
        'tip': "Rinse thoroughly and remove caps before recycling."
    },
    'Metal': {
        'info': "Recycling metals saves up to 95% of the energy required to make them from raw materials.",
        'tip': "Crush aluminum cans to save space in your recycling bin."
    },
    'Nylon': {
        'info': "Nylon takes hundreds of years to decompose in landfills.",
        'tip': "Reuse bags whenever possible or take them to specific plastic film drop-offs."
    },
    'Organic Waste': {
        'info': "Organic waste produces methane when decomposing in landfills.",
        'tip': "Compost food scraps to enrich soil and reduce greenhouse gas emissions."
    },
    'Plastic Bottles': {
        'info': "Plastic bottles take up to 450 years to decompose.",
        'tip': "Empty and flatten bottles before placing them in recycling bins."
    }
}

# Image File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Waste"):
        results = model(image)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detected Waste", use_container_width=True)
        
        # Display waste tips for detected classes
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                if class_name in waste_info:
                    st.subheader(f"Information: {class_name}")
                    st.write(f"**Fact:** {waste_info[class_name]['info']}")
                    st.write(f"**Tip:** {waste_info[class_name]['tip']}")
