import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO

# Load YOLO model
model_path = "best (1).pt"  # Update your model path
model = YOLO(model_path)

# Streamlit page config
st.set_page_config(page_title="PlastiScan 🌊", page_icon="🌍", layout="wide")

# Custom CSS with gradient (orange to red)
st.markdown("""
    <style>
    /* Target the main app container */
    [data-testid="stAppViewContainer"] {
        background: #e319e6;
background: linear-gradient(90deg,rgba(227, 25, 230, 1) 0%, rgba(87, 199, 188, 1) 50%, rgba(237, 221, 83, 1) 100%);
        background-attachment: fixed;
    }

    .main {
        padding: 2rem;
        border-radius: 20px;
    }

    .title {
        background: green;
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
    }

   .upload-section, .output-section {
    border: 2px solid #ff6a00;
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ffecd2, #fcb69f);
    background-blend-mode: overlay;
    margin: 1rem 0;
    
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: all 0.3s ease-in-out;
}


    .stButton>button {
        background: #19ace6;
background: linear-gradient(90deg,rgba(25, 172, 230, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);

        color: Blue;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        font-weight: bolder;
        font-size: 1rem;
        
        transition: 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background: #19ace6;
background: linear-gradient(90deg,rgba(25, 172, 230, 1) 0%, rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);

        transform: scale(1.05);
    }

    .footer {
        text-align: center;
        font-size: 20px;
        color: #34495e;
        margin-top: 2rem;
        
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🌊 PlastiScan - Ocean Plastic Waste Detector</div>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#34495e;'>Detect ocean plastic waste with YOLOv8. Just upload an image and click detect!</h4>", unsafe_allow_html=True)

# Upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)

st.markdown("""
    <h3 style='text-align: center; color: #8e44ad; font-size: 1.8rem; font-weight: bold;'>
        📤 Upload an Image
    </h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("</div>", unsafe_allow_html=True)


# If image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Styled caption
    st.markdown("""
        <h3 style='text-align:center; color:#1f618d; font-weight: bold;'>
            🖼️ Uploaded Image
        </h3>
    """, unsafe_allow_html=True)

    detect_btn = st.button("🔍 Detect Plastic")

    if detect_btn:
        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # YOLO detection
        results = model(image_np, conf=0.1)
        class_names = model.names
        plastic_found = False

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Plastic if not 'misc'
                is_plastic = label.lower() != "misc"
                if is_plastic:
                    plastic_found = True
                    box_color = (255, 0, 0)  # Red for plastic
                else:
                    box_color = (0, 255, 0)  # Green for misc

                # Draw bounding box and label
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), box_color, 2)
                text = f"{label} ({conf:.2f})"
                cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Convert back to RGB
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(image_cv)

        # Display output image
        st.markdown("<div class='output-section'>", unsafe_allow_html=True)
        st.image(output_img, use_container_width=True)

        # Custom styled caption
        st.markdown("""
            <h3 style='text-align:center; color:#1f618d; font-weight: bold;'>
                ✅ Detected Plastic Waste
            </h3>
        """, unsafe_allow_html=True)

        # Display detection result
        if plastic_found:
            st.markdown("""
                <h3 style='text-align:center; color:#8B008B; font-weight: bold;'>
                    🟢 Plastic Detected in the Image!
                </h3>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <h3 style='text-align:center; color:blue; font-weight: bold;'>
                    🔵 No Plastic Found.
                </h3>
            """, unsafe_allow_html=True)

        # Download option
        img_buffer = BytesIO()
        output_img.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        st.download_button(
            "⬇️ Download Detected Image",
            data=img_bytes,
            file_name="plastic_detected.jpeg",
            mime="image/jpeg"
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='footer' style='
    font-size: 20px; 
    font-weight: bold; 
    text-align: center; 
    background: linear-gradient(90deg, #ff9a9e, #fad0c4, #fad0c4, #fbc2eb, #a6c1ee);
    -webkit-background-clip: text;
    color: transparent;
    padding: 10px;
'>
    💡 This app uses a <span style="color:#e74c3c;">YOLOv8</span> model to identify 
    <span style="color:yellow;">plastic waste</span> in the ocean. 🌊 
    Save our <span style="color:#3498db;">blue planet</span>! 💙🐠
</div>
""", unsafe_allow_html=True)

