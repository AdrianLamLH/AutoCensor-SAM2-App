# Streamlit Libraries
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# SAM2 Libraries
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


#! /opt/anaconda3/envs/autocensor-env/bin/python
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch

# 3rd Party Libraries
from funcs import *
st.write("Hello World!")

def main():
    
    # if 'initialized' not in st.session_state or not st.session_state.initialized:

    # device = initialize_cuda()
    
    # device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # st.session_state.initialized = True  # Mark initialization as done to avoid re-running
    # else:
    #     st.write("Initialization already done.")
    # Upload your image to censor
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        image =  np.array(Image.open(uploaded_file).convert("RGB"))
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(fig)

        # Initialize SAM2 Automatic Mask Generator class
        sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

        # mask_generator = SAM2AutomaticMaskGenerator(sam2)
        # masks = mask_generator.generate(image)
        # print(len(masks))
        # print(masks[0].keys())
        # fig2 = plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # show_anns(masks)
        # plt.axis('off')
        # st.pyplot(fig2)

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        fig2 = plt.figure(figsize=(10, 10))
        plt.imshow(image)

        input_point = np.array([[200, 600]])
        input_label = np.array([1])

        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        st.pyplot(fig2)  
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        fig3 = show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
        st.pyplot(fig3)

# ----------------------------------------------------------------------------------------------

        init_image = image

        generator = torch.Generator(device="cpu").manual_seed(1)
        mask_image = masks[0]



        def make_inpaint_condition(image, image_mask):
            # image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            # image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image


        control_image = make_inpaint_condition(init_image, mask_image)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32
        ).to(device)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
        ).to(device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()

        # generate image
        image = pipe(
            "blue trenchcoat",
            num_inference_steps=20, 
            generator=generator,
            eta=1.0,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
        ).images[0]
        fig4 = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        st.pyplot(fig4)



if __name__ == "__main__":
    main()