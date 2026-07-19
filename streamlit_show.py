from tensorflow.keras.preprocessing import image
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

st.set_page_config(page_title="Upload Photo + Predict", page_icon="📷")

st.title("Do you wear a Mask? SO, How do you feel today? Prediction")
st.write("Take a photo or upload one, then run it through the CNN model and optionally save it")

# ---------- Settings ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_MASK = os.path.join(BASE_DIR, 'last_mask_model_cnn.keras')
MODEL_PATH_EMOTIONS = os.path.join(BASE_DIR, 'face_net_emotions_6_classes.keras')
IMG_SIZE_MASK = (64, 64 )  # must match training target_size
IMG_SIZE_EMOTIONS = (160, 160)

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

st.session_state.setdefault('without_mask',None)
st.session_state.setdefault('mode',None)
st.session_state.setdefault('step',0)
st.session_state.setdefault('img_path',None)
st.session_state.setdefault('img',None)

def scaling_fix(x, scale=0.17):
    return x * scale

@st.cache_resource
def get_model_mask(p):
    return load_model(p)


@st.cache_resource
def get_model_emotions(p):
    return load_model(
        p,
        custom_objects={'scaling': scaling_fix},
        compile=False,
        safe_mode=False
    )

mask_model = get_model_mask(MODEL_PATH_MASK)
emotions_model = get_model_emotions(MODEL_PATH_EMOTIONS)

def drop_image(img_path):
    try:
        os.remove(img_path)
    except Exception as e:
        raise (str(e))

def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path

def preprocess_for_model(uploaded_file,mode,img_path):
    # Streamlit UploadedFile -> bytes -> PIL Image
    if mode == None:
        return None

    if mode == "mask":
        img = Image.open(uploaded_file)

        # Ensure 3 channels (RGB). This also handles PNG with alpha (RGBA)
        img = img.convert("RGB")

        # Resize to model expected input
        img = img.resize(IMG_SIZE_MASK)

        # PIL -> np array
        arr = np.array(img, dtype=np.float32)

        # Normalize
        arr /= 255.0

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)
        return arr

    else:
        test_image = image.load_img(
            img_path,
            target_size=(160, 160),
            color_mode='rgb'
        )

        # Convert image to array
        arr = image.img_to_array(test_image)

        # Normalize pixel values
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

        # Expand dimensions
        arr = np.expand_dims(arr, axis=0)
        return arr

def prediction_for_cnn(x1,model1,mode):
    if mode is None:
        return None, None

    feel_classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    mask_classes = ['incorrect mask','with mask','without mask']

    emotions_conf = 0.20
    mask_conf = 0.33

    classes = mask_classes if mode == 'mask' else feel_classes

    result = model1.predict(x1)

    argmax = np.argmax(result[0])
    acc = result[0][argmax]
    cl = classes[argmax]

    if mode == 'mask':
        if acc <= mask_conf + 0.05:
            cl = 'UnKnown'
    else:
        if acc <= emotions_conf + 0.05:
            cl = 'UnKnown'


    return cl,acc

# ---------- UI ----------
st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

st.session_state["img"] = camera_photo if camera_photo is not None else gallery_photo

if st.session_state["img"] is not None:
    st.image(st.session_state["img"], caption="Preview", use_container_width=True)


    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Predict Mask Or No (or false mask)"):
            st.session_state['mode'] = 'mask'
            st.session_state['img_path'] = save_uploaded_image(st.session_state["img"])

            try:
                mode = st.session_state['mode']

                x_mask = preprocess_for_model(st.session_state["img"],mode,st.session_state['img_path'])
                mask_prediction,mask_confidence = prediction_for_cnn(x_mask,mask_model,mode)

                if mask_prediction == 'without mask':
                    st.session_state['without_mask'] = True

                st.info(f"From Class {mask_prediction}\nAccuracy = {mask_confidence*100:.2f}% ")

                # Debug: show the shape to prove it's correct
                st.caption(f"Sent to model with shape: {x_mask.shape} (should be (1, 64, 64, 3))")

                drop_image(st.session_state["img_path"])
                st.session_state["img_path"] = None

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        if st.button("💾 Save image"):
            try:
                saved_path = save_uploaded_image(st.session_state["img"], prefix="phone")
                st.success(f"Saved ✅  {saved_path.resolve()}")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    if st.session_state['without_mask'] is None:
        st.info(f"For emotion diagnose first we have to see you don't wear a mask\nPush The predict Mask button")
        st.stop()

    if st.session_state['without_mask'] is True:
        if st.button("Tell me How I Feel"):
                st.session_state['mode'] = 'emotions'
                st.session_state['img_path'] = save_uploaded_image(st.session_state["img"])

                try:
                    mode = st.session_state['mode']
                    x_feel = preprocess_for_model(st.session_state["img"],mode,st.session_state['img_path'])
                    emotion_prediction, emotion_confidence = prediction_for_cnn(x_feel, emotions_model, mode)

                    if emotion_prediction is not None and emotion_confidence is not None:
                        st.session_state['mode'] = None
                        st.session_state['without_mask'] = None

                    st.info(f"From Class {emotion_prediction}\nConfidence = {emotion_confidence * 100:.2f}%")

                    drop_image(st.session_state["img_path"])
                    st.session_state["img_path"] = None

                except Exception as e:
                    st.error(f"Failed to Predict: {e}")


st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")
st.caption(f"Model Mask Catcher file: {Path(MODEL_PATH_MASK).resolve()}")
st.caption(f"Model Feeling Recognization file: {Path(MODEL_PATH_EMOTIONS).resolve()}")