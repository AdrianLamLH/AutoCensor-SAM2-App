# Streamlit Libraries
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# SAM2 Libraries
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 3rd Party Libraries
import funcs

st.write("Hello World!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image =  np.array(Image.open(uploaded_file).convert("RGB"))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    st.pyplot(fig)

    