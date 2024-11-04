
import streamlit as st
import numpy as np
import cv2
import stone
import math
import os

# Define example RGB values for each skin tone category
skin_tone_examples = {
    "light": [(247, 218, 200), (234, 238, 183), (237, 221, 205), (232, 199, 179), (229, 204, 199), (235, 218, 207),
              (232, 192, 183)],
    "pale": [(241, 206, 199), (246, 201, 177), (239, 193, 166), (229, 195, 161), (235, 184, 150)],
    "medium": [(238, 185, 138), (229, 171, 117), (226, 158, 105), (220, 182, 141), (213, 170, 134), (214, 144, 86)],
    "tanned": [(199, 160, 124), (171, 138, 119), (150, 121, 101), (167, 133, 94), (157, 110, 63), (142, 122, 86)],
    "dark/deep": [(136, 114, 96), (134, 90, 51), (118, 75, 41), (147, 98, 67), (134, 85, 52), (122, 71, 40)],
    "black": [(99, 53, 23), (81, 53, 33), (81, 63, 55), (70, 51, 37), (80, 36, 10), (68, 20, 6)]
}


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def euclidean_distance(rgb1, rgb2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def get_skin_tone_category(rgb_value):
    closest_category = "unknown"
    smallest_distance = float('inf')
    for category, examples in skin_tone_examples.items():
        for example in examples:
            distance = euclidean_distance(rgb_value, example)
            if distance < smallest_distance:
                smallest_distance = distance
                closest_category = category
    return closest_category


# Streamlit app starts here
st.title("Skin Tone Classification")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image directly
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    image_path = 'uploaded_image.png'
    cv2.imwrite(image_path, image)

    result = stone.process(image_path, image_type="color", return_report_image=True)

    for face in result["faces"]:
        rgb_value = hex_to_rgb(face["skin_tone"])
        category = get_skin_tone_category(rgb_value)
        st.markdown(f"""
            <div style="font-size:24px; font-weight:bold;">
                Face has skin tone category: <span style="font-weight:bold;">{category}</span>
            </div>
            """, unsafe_allow_html=True)

    report_images = result.pop("report_images")
    if report_images:
        for face_id, img in report_images.items():
            # Ensure the image is in the correct format (uint8)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # Convert BGR image to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(rgb_img, caption=f"Report Image for Face ID {face_id}", use_column_width=True)
    else:
        st.write("No report images available.")
else:
    st.write("Please upload an image.")
