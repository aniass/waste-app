import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

MODELSPATH = 'waste_model.h5'
DATAPATH = 'waste1.jpg'


@st.cache
def load_data():
    img = Image.open(DATAPATH)
    return img


def load_models():
    model = load_model(MODELSPATH, compile=False)
    return model


st.title('Waste prediction')
st.markdown('**This application will let you verify if your waste may be recyclable or not.**')
st.markdown('To see prdiction you may use a sample data or upload your own image.')

st.sidebar.title('Select options:')
page = st.sidebar.selectbox("Choose a page:", ["Sample Data", "Upload an Image"])

if page == "Sample Data":
    st.header("Sample prediction for waste")
    if st.checkbox('Show Sample Data'):
        st.info("Sample image:")
        image = load_data()
        st.image(image, caption='Sample Data', use_column_width=True)
        image = image.resize((64, 64))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of simple image'):
            model = load_models()
            result = model.predict(image)
            st.write(result)
            if result[0][0] == 1:
                prediction = 'Recyclable Waste'
            else:
                prediction = 'Organic Waste'
            st.write(prediction)
            st.success("It is prediction for sample image!")

if page == "Upload an Image":
    st.header("Your prediction of waste")
    uploaded_file = st.file_uploader("Choose your image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.info("Show your image:")
        st.image(img, caption="Upload image", use_column_width=True)
        img = img.resize((64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.reshape((1, 64, 64, 3))
        st.subheader("Check waste prediction")
        if st.checkbox('Show Prediction of your image'):
            model = load_models()
            result = model.predict(img)
            st.write(result)
            if result[0][0] == 1:
                prediction = 'Recyclable Waste'
            else:
                prediction = 'Organic Waste'
            st.write(prediction)
            st.success("That is your prediction!")
