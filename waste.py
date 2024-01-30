import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = 'waste_model.h5'
DATA_PATH = 'waste1.jpg'


@st.cache
def load_data(image_path):
    '''Function to load image'''
    img = Image.open(image_path)
    return img


@st.cache
def load_models(model_path):
    '''Function to load model'''
    model = load_model(model_path, compile=False)
    return model


def image_processing(img):
    '''Function for image processing'''
    img_resized = img.resize((64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array.reshape((1, 64, 64, 3))
    return img_array


# Main Streamlit app
st.title('Waste prediction')
st.markdown('**This application will let you verify if your waste may be recyclable or not.**')
st.markdown('To see prdiction you may use a sample data or upload your own image.')

st.sidebar.title('Select options:')
page = st.sidebar.selectbox("Choose a page:", ["Sample Data", "Upload an Image"])

if page == "Sample Data":
    st.header("Sample prediction for waste")
    if st.checkbox('Show Sample Data'):
        st.info("Sample image:")
        image = load_data(DATA_PATH)
        st.image(image, caption='Sample Data', use_column_width=True)
        image_array = image_processing(image)
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of simple image'):
            model = load_models(MODEL_PATH)
            result = model.predict(image_array)
            st.write(result)
            prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'
            st.write(prediction)
            st.success("It is prediction for sample image!")

if page == "Upload an Image":
    st.header("Your prediction of waste")
    uploaded_file = st.file_uploader("Choose your image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.info("Show your image:")
        st.image(img, caption="Upload image", use_column_width=True)
        img_array = image_processing(image)        
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of your image'):
            model = load_models(MODEL_PATH)
            result = model.predict(img_array)
            st.write(result)
            prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'
            st.write(prediction)
            st.success("That is your prediction!")
