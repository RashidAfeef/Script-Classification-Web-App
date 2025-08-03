import os
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st


model_path = 'Model_2Loss_Aug.zip'
model_name = 'Model_2Loss_Aug.keras'
target_size = (28,28)

@st.cache_resource
def load_model(zipped_path, unzipped_filename):
    if not os.path.exists(unzipped_filename):
        st.info(f"Unzipping model from {zipped_path}...")
        try:
            with zipfile.ZipFile(zipped_path, 'r') as zip_ref:
                zip_ref.extractall('.') # Extract all contents to the current directory
            st.success("Model unzipped successfully!")
        except FileNotFoundError:
            st.error(f"Error: Zipped model file not found at {zipped_path}.")
            st.stop()
        except Exception as e:
            st.error(f"Error unzipping model: {e}")
            st.stop()

    try:
        model = tf.keras.models.load_model(unzipped_filename)
        return model
    except FileNotFoundError:
        st.error(f"Error: Unzipped model file '{unzipped_filename}' not found after extraction. Please check the zip contents.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def preprocess_image(image, img_size): 
    img = image.convert('RGB')
    img = img.resize(img_size)
    img_np = np.array(img).astype('float32')/255.0
    return img_np


model = load_model(model_path)
with open('y1_map.pkl', 'rb') as f:
    y1_map = pickle.load(f)
with open('y2_map.pkl', 'rb') as f:
    y2_map = pickle.load(f)
st.title("CNN Image Classifier 📸")
st.write("Upload an image or use your camera to get a prediction from the CNN model.")
st.subheader("Choose Image Input:")

camera_image = st.camera_input("Take a picture")    
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

image_to_process = None

if camera_image is not None:
    image_to_process = Image.open(camera_image)
    st.image(image_to_process, caption='Image from Camera', use_column_width=True)
elif uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
    st.image(image_to_process, caption='Uploaded Image', use_column_width=True)


if image_to_process is not None:
    if model:
        st.subheader("Processing and Prediction:")
        with st.spinner("Processing image and making prediction..."):
            try:
                image_to_process = preprocess_image(image_to_process, target_size)

                c1_start = 3
                c1_end = c1_start + 28
                c2_start = c1_end
                c2_end = c2_start + 20
                c3_start = c2_end
                c3_end = c3_start + 32


                pred = model(tf.expand_dims(image_to_process,axis=0))[0]
                #y1 logits
                bali_pred = pred[0]
                jawa_pred = pred[1]
                sunda_pred = pred[2]

                #y2 maximum logits's index for each script
                bali_label = tf.argmax(pred[c1_start:c1_end]).numpy()
                jawa_label = tf.argmax(pred[c2_start:c2_end]).numpy()
                sunda_label = tf.argmax(pred[c3_start:c3_end]).numpy()

                #y2 maximum logits for each script
                bali_prob = tf.reduce_max(pred[c1_start:c1_end])
                jawa_prob = tf.reduce_max(pred[c2_start:c2_end])
                sunda_prob = tf.reduce_max(pred[c3_start:c3_end])

                #Highest average of y1 & y2 logits
                overall_pred = tf.constant([tf.reduce_mean([bali_pred,bali_prob]).numpy(),tf.reduce_mean([jawa_pred,jawa_prob]).numpy(),tf.reduce_mean([sunda_pred,sunda_prob]).numpy()])
                overall_pred = tf.argmax(overall_pred)

                #Get the result
                y1_overall = None
                y2_overall = None

                if overall_pred == 0:
                    y1_overall = "Bali"
                    y2_overall = y2_map[bali_label]
                elif overall_pred == 1:
                    y1_overall = "Jawa"
                    y2_overall = y2_map[jawa_label + 28]
                elif overall_pred == 2:
                    y1_overall = "Sunda"
                    y2_overall = y2_map[sunda_label + 28 + 20]
                
                st.success("Prediction Complete!")
                st.write(f"**Aksara:** {y1_overall}")
                st.write(f"**Karakter:** {y2_overall}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure your image is valid and matches the expected input for the model.")
    else:
        st.warning("Model could not be loaded. Please check the model file and path.")
else:
    st.info("Please take a picture or upload an image to get a prediction.")