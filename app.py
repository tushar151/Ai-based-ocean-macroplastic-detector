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

# Title
st.markdown(
    "<h1 style='text-align:center; font-size: 3rem; font-weight: bold;'>🌊 PlastiScan - Ocean Plastic Waste Detector</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align:center; color:#34495e;'>Detect ocean plastic waste with YOLOv8. Just upload an image and click detect!</h4>",
    unsafe_allow_html=True)

# Upload section
st.markdown(
    "<div style='padding: 25px; border-radius: 20px; background: linear-gradient(135deg, #ffecd2, #fcb69f); margin: 1rem 0; box-shadow: 0 8px 20px rgba(0,0,0,0.2); transition: all 0.3s ease-in-out;'>",
    unsafe_allow_html=True)

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

    # Slider with inline CSS
    st.markdown("""
        <style>
            /* Slider label and number color */
            label[data-testid="stSliderLabel"] {
                font-size: 1.3rem !important;
                font-weight: bold !important;
                color: #003366 !important;  /* Dark blue */
            }

            /* Slider track and background */
            .stSlider > div > div > div > div {
                background-color: black !important;  /* Black background for the slider track */
            }

            /* Slider handle */
            .stSlider > div > div > div > div:nth-child(3) {
                background-color: black !important;  /* Black handle */
                border-radius: 50%;
            }

            /* Value numbers on the slider */
            .stSlider .css-1lv4x1j, .stSlider .css-14xtw13 {
                color: black !important;  /* Black color for the number values on the slider */
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider(
        "🛠️ Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Adjust this to reduce false positives like misclassifying turtles as plastic."
    )

    detect_btn = st.button("🔍 Detect Plastic")

    if detect_btn:
        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # YOLO detection
        results = model(image_np, conf=conf_threshold)
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
        st.markdown(
            "<div style='padding: 25px; border-radius: 20px; background: linear-gradient(135deg, #ffecd2, #fcb69f); margin: 1rem 0; box-shadow: 0 8px 20px rgba(0,0,0,0.2); transition: all 0.3s ease-in-out;'>",
            unsafe_allow_html=True)
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

st.markdown(
    """<div class='footer' style='font-size: 20px; font-weight: bold; text-align: center; background: linear-gradient(90deg, #ff9a9e, #fad0c4, #fad0c4, #fbc2eb, #a6c1ee); -webkit-background-clip: text; color: transparent; padding: 10px;'>💡 This app uses a <span style="color:#e74c3c;">YOLOv8</span> model to identify <span style="color:yellow;">plastic waste</span> in the ocean. 🌊 Save our <span style="color:#3498db;">blue planet</span>! 💙🐠</div>""",
    unsafe_allow_html=True)
