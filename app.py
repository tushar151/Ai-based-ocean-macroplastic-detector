import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO

# Load YOLO model
model_path = "C:/Users/TUSHAR SETHI/Downloads/best (1).pt"  # Update with your model path
model = YOLO(model_path)

# Define colors
plastic_color = (255, 0, 0)      # Blue for Plastic
non_plastic_color = (0, 0, 255)  # Red for Not Plastic

# Streamlit UI
st.set_page_config(page_title="PlastiScan 🌊", page_icon="🌍", layout="wide")
st.title("🔍 PlastiScan")
st.markdown(
    "<h4 style='color: #3498db;'>Upload an image to detect plastic waste in the ocean! 🌊</h4>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    # Display uploaded image
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")  # Ensure 3-channel RGB
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    image_np = np.array(image)  # Now it's guaranteed to be RGB
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    # Run YOLO detection

    results = model(image_np, conf=0.05)
    # Adjust confidence as needed


    # Get class names
    class_names = model.names

    # Process detection results
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Class index
            conf = float(box.conf[0])  # Confidence score
            label = class_names[cls]  # Get original class name

            # Assign category
            if label == "misc":
                final_label = "Not Plastic"
                color = non_plastic_color  # Red
            else:
                final_label = "Plastic"
                color = plastic_color  # Blue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

            # Put label on image
            text = f"{final_label} ({conf:.2f})"
            cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert back to RGB for display
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    with col2:
        st.markdown("### 🔍 Detected Image")
        st.image(image_cv, caption="Processed Image", use_column_width=True)

    # Convert image for downloading
    output_img = Image.fromarray(image_cv)
    img_buffer = BytesIO()
    output_img.save(img_buffer, format="JPEG")  # Use PNG if needed
    img_bytes = img_buffer.getvalue()

    # Download button
    st.download_button(
        "⬇️ Download Processed Image",
        data=img_bytes,
        file_name="processed_image.jpeg",
        mime="image/jpeg"
    )

st.markdown("---")
st.markdown("💡 **This app detects ocean Plastic Waste using a YOLOv8 model. Help save the oceans! 🌎💙**")
