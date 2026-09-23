
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/tomato_disease_efficientnet.keras"
    )

    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_model()


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(img_array, model, pred_index=None):

    augmentation = model.get_layer("sequential")
    base_model = model.get_layer("efficientnetb0")

    pooling = model.get_layer(
        "global_average_pooling2d"
    )

    dropout = model.get_layer("dropout")
    classifier = model.get_layer("dense")

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("top_conv").output,
            base_model.output
        ]
    )

    x = augmentation(
        img_array,
        training=False
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = grad_model(
            x,
            training=False
        )

        x = pooling(features)

        x = dropout(
            x,
            training=False
        )

        predictions = classifier(x)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def create_overlay(image, heatmap):

    original = np.array(image)

    heatmap_resized = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return heatmap_resized, overlay


# =========================================================
# TITLE
# =========================================================

st.title("🍅 Tomato Leaf Disease Detection")

st.write(
    "Upload a tomato leaf image to identify the disease "
    "using an EfficientNetB0 deep learning model."
)

st.write(
    "Grad-CAM is used to visualize the regions "
    "that influenced the prediction."
)

st.divider()


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # ---------------------------------------------
    # ORIGINAL IMAGE
    # ---------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=400
    )

    # ---------------------------------------------
    # PREPROCESS
    # ---------------------------------------------

    resized_image = image.resize(
        (224, 224)
    )

    img_array = np.array(
        resized_image
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype("float32")

    # ---------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    # ---------------------------------------------
    # RESULT
    # ---------------------------------------------

    st.subheader("Prediction")

    st.success(
        f"🌿 Disease: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # ---------------------------------------------
    # TOP 3
    # ---------------------------------------------

    st.subheader("Top 3 Predictions")

    top3_indices = np.argsort(
        predictions
    )[-3:][::-1]

    for rank, index in enumerate(
        top3_indices,
        start=1
    ):

        st.write(
            f"**{rank}. {class_names[index]}** "
            f"— {predictions[index] * 100:.2f}%"
        )

    # ---------------------------------------------
    # GRAD-CAM
    # ---------------------------------------------

    st.subheader("🔍 Grad-CAM Explanation")

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        predicted_index
    )

    heatmap_resized, overlay = create_overlay(
        image,
        heatmap
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            heatmap_resized,
            caption="Grad-CAM",
            use_container_width=True,
            clamp=True
        )

    with col3:

        st.image(
            overlay,
            caption="Prediction Explanation",
            use_container_width=True
        )
