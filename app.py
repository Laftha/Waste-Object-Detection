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
    **Project Description:** A smart waste classification capstone project designed to help identify recyclable materials and provide disposal insights.
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
    
    st.markdown("---")
    st.subheader("💬 Send Feedback")
    with st.sidebar.form("feedback_form"):
        user_contact = st.text_input("Name or Email (Optional)", placeholder="e.g., John or john@email.com")
        user_feedback = st.text_area("Help us improve EcoSort:", placeholder="Type your feedback here...")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            if user_feedback.strip():
                st.success("Thank you! Your feedback has been received.")
                with open("feedback_log.txt", "a") as f:
                    f.write(f"Contact: {user_contact or 'Anonymous'}\nFeedback: {user_feedback}\n---\n")
            else:
                st.warning("Please write some feedback before submitting.")

# --- MAIN PAGE HEADER ---
col_title, col_logo = st.columns([5, 1])

with col_title:
    st.title("♻️ EcoSort AI: Smart Waste Detector")
    st.markdown("Upload an image or take a live picture of waste to identify categories and view recycling tips.")

with col_logo:
    st.image("Screenshot 2026-10-30 011304.jpg", width=140)

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Main Layout: Two Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Provide Waste Image")
    input_type = st.radio("Choose Input Method:", ["📁 Upload File", "📸 Use Camera"], horizontal=True)
    
    uploaded_file = None
    camera_file = None
    
if input_type == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload a waste photo from folder", type=["jpg", "jpeg", "png", "JPG", "JPEG"])
else:
    st.subheader("📸 Back Camera Capture")
    # Streamlit's native camera input handles the device camera cleanly
    camera_file = st.camera_input("Take a picture of the waste", facingMode="environment")
# Use whichever image the user provided
image_file = uploaded_file if input_type == "📁 Upload File" else camera_file

with col2:
    st.subheader("2. AI Analysis & Insights")
    
    # Select input source
    image_file = uploaded_file if input_type == "📁 Upload File" else camera_file
    
    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_container_width=True)
        
        # Custom styling for green button
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #28a745;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        div.stButton > button:first-child:hover {
            background-color: #218838;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Run AI Classification"):
            with st.spinner("Analyzing image..."):
                temp_path = "temp_image.jpg"
                img_pil = Image.open(image_file).convert("RGB")
                img_pil.save(temp_path)
                
                # Run inference
                results = model.predict(temp_path, conf=0.25)
                r = results[0]
                
                # Render bounding boxes
                im_array = r.plot()
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detection Result", use_container_width=True)
                
                # Render Insights and Classes with Detailed Cards
                classes = [model.names[int(box.cls[0])] for box in r.boxes]
                if classes:
                    unique_classes = set(classes)
                    st.success(f"**Detected Items:** {', '.join(unique_classes)}")
                    st.markdown("### 💡 Environmental Insights & Guidelines")
                    
                    for cls_name in unique_classes:
                        item_lower = cls_name.lower()
                        
                        if "plastic" in item_lower or "bottle" in item_lower:
                            with st.expander("♻️ Plastic Bottles Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** 450 years")
                                st.markdown("ℹ️ **Did You Know?** Over 1 million plastic bottles are purchased every minute worldwide, with low recycling rates.")
                                st.markdown("🌱 **Actionable Tip:** Empty, rinse, and flatten bottles to save bin space before recycling.")
                                st.info("🌍 **Eco Impact:** Recycling 1 ton of plastic saves 5,774 kWh of electricity and 16.3 barrels of oil.")
                                
                        elif "nylon" in item_lower or "soft" in item_lower:
                            with st.expander("🛍️ Nylon / Soft Plastics Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** 20 - 500 years")
                                st.markdown("ℹ️ **Did You Know?** Soft plastics and plastic bags often clog municipal recycling machinery if mixed with rigid plastics.")
                                st.markdown("🌱 **Actionable Tip:** Keep clean and dry. Return them to local grocery store collection bins.")
                                st.info("🌍 **Eco Impact:** Diverting soft plastics from landfills reduces litter and prevents wildlife entanglement.")
                                
                        elif "glass" in item_lower:
                            with st.expander("🍾 Glass Bottles Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** 1+ million years (virtually infinite)")
                                st.markdown("ℹ️ **Did You Know?** Glass can be recycled infinitely without loss in quality or purity.")
                                st.markdown("🌱 **Actionable Tip:** Rinse thoroughly and separate by color if required by your local facility.")
                                st.info("🌍 **Eco Impact:** Recycling glass significantly reduces greenhouse gas emissions from melting raw materials.")
                                
                        elif "fabric" in item_lower or "textile" in item_lower:
                            with st.expander("👕 Fabric / Textiles Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** 20 - 200+ years (depending on synthetic content)")
                                st.markdown("ℹ️ **Did You Know?** Millions of tons of textiles end up in landfills annually, though a vast majority are reusable or recyclable.")
                                st.markdown("🌱 **Actionable Tip:** Donate wearable clothes and drop off unwearable textiles at community textile recycling bins.")
                                st.info("🌍 **Eco Impact:** Textile recycling saves water, reduces pesticide use, and curtails carbon footprints.")
                                
                        elif "metal" in item_lower:
                            with st.expander("🥫 Metal Waste Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** 50 - 500 years")
                                st.markdown("ℹ️ **Did You Know?** Aluminum can be recycled and back on store shelves in as little as 60 days.")
                                st.markdown("🌱 **Actionable Tip:** Rinse cans and crush them to maximize recycling bin capacity.")
                                st.info("🌍 **Eco Impact:** Recycling aluminum uses 95% less energy than producing new metal from raw bauxite ore.")
                                
                        else:
                            with st.expander(f"📦 {cls_name} Details", expanded=True):
                                st.markdown("⏱️ **Decomposition Time:** Varies by material")
                                st.markdown("🌱 **Actionable Tip:** Check local municipal recycling guidelines for proper sorting.")
                                st.info("🌍 **Eco Impact:** Proper sorting reduces landfill overflow and promotes a circular economy.")
                else:
                    st.warning("No waste items detected clearly. Try taking a photo closer to the object.")
    else:
        st.info("👈 Upload an image or take a photo in Step 1 to unlock AI classification.")

# Footer Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** EcoSort AI provides automated waste classification for educational and demonstration purposes. Model predictions should be verified independently before making critical waste disposal decisions.")
