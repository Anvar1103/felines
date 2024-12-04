import streamlit as st

from fastai.vision.all import *
import pathlib
import plotly.express as pxn25
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Rasimlarni klassifikatsiya qilish")

files=st.file_uploader("Rasim yuklash", type=["jpg","svg","png"])

if files:
    st.image(files, caption="Yuklangan rasim", use_column_width=True)
    
    # Save the uploaded file temporarily
    img = PILImage.create(files)
    
    # Load your pre-trained model (make sure the 'model.pkl' file is accessible)
    model = load_learner('model.pkl')
    
    # Make a prediction
    pred, pred_idx, probs = model.predict(img)
    
    # Display the prediction and the probability
    st.write(f"Bashorat: {pred}")
    st.write(f"Ishonch darajasi: {probs[pred_idx]:.4f}")