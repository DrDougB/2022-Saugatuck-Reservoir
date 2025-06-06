#---------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024/2025: Douglas J. Buettner, PhD. GPL-3.0 license
# specific terms of this GPL-3.0 license can be found here:
# https://github.com/DrDougB/XXX
#
# Python code to read in each of the frames extracted from a video and "brighten" them.
# 
# The code was generated in part with GPT-4, OpenAI's large-scale language-generation
# model. Upon generating draft language, the author reviewed, edited, and revised the content 
# to his own liking (for example fixing errors) and takes ultimate responsibility for the content.
#
# OpenAI's Terms of Use statement dated March 14, 2023 (last visited Aug 8th, 2023) is 
# available here: https://openai.com/policies/terms-of-use)
#
# NO WARRANTY NOTE: Use at your own risk! While some attempt has been made to test the functionality
#                   the author does not claim that all permuations through the code (especially turning 
#                   on and off various analytical methods) works correctly.
#
# Change History: 
# Initial Version 1.0 - DJB (12/28/24):
#---------------------------------------------------------------------------------------------------------------------

import os
import cv2
import numpy as np

def increase_brightness(img_bgr, val=0.3, adj=20):
    """
    Closely mimics Mathematica's frameLighten algorithm with:
      1. Binarize[img, val]
      2. ColorNegate
      3. ImageAdjust[img, {0, adj}] => multiply by adj, clip to [0..1]
      4. Compose brightened image onto original where mask == 1

    Parameters
    ----------
    img_bgr : np.ndarray
        8-bit BGR image as read by cv2.imread.
    val : float
        Threshold in [0, 1]. Pixels < val become 'lightened' region.
    adj : float
        Factor to multiply for ImageAdjust.

    Returns
    -------
    result_bgr_8u : np.ndarray
        8-bit BGR result with darker regions lightened.
    """

    # 1) Convert original BGR to float32 [0..1]
    img_float = img_bgr.astype(np.float32) / 255.0

    # 2) Compute grayscale in [0..1] for thresholding
    #    (OpenCV's BGR2GRAY formula is 0.299*R + 0.587*G + 0.114*B).
    gray_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # 3) Binarize -> In Mathematica, Binarize[img, val] sets
    #      pixel >= val -> 1
    #      pixel <  val -> 0
    #    OpenCV threshold can invert this logic, so let's do it step by step:
    mask_bin = (gray_float >= val).astype(np.float32)  # 1 where >= val, 0 where < val

    # 4) ColorNegate -> we invert the mask so that
    #      mask_final = 1 where gray < val
    #      mask_final = 0 where gray >= val
    mask_final = 1.0 - mask_bin  # now 1 if pixel < val, else 0

    # 5) Create a "brightened" version by multiplying by 'adj' and clipping to 1
    bright_float = img_float * adj
    bright_float = np.clip(bright_float, 0.0, 1.0)

    # 6) Compose => wherever mask_final == 1, use bright_float; else original
    #    Equivalent to: result = original*(1 - mask_final) + bright*(mask_final)
    #    Because mask_final is 1 for dark pixels, 0 for others.
    #    We need to broadcast the single-channel mask to 3 channels
    mask_3ch = cv2.merge([mask_final, mask_final, mask_final])
    result_float = img_float * (1.0 - mask_3ch) + bright_float * mask_3ch

    # 7) Convert back to 8-bit BGR
    result_bgr_8u = (result_float * 255.0).astype(np.uint8)
    return result_bgr_8u
    
    
def process_frames(frames_dir, lightened_dir, val = 0.5, adj=30):
    """
    Reads each PNG (or JPG) from frames_dir, applies lighten_image, 
    and saves results to lightened_dir.

    Parameters:
    -----------
    frames_dir : str
        Path to directory containing original frames, e.g. 'frames'.
    lightened_dir : str
        Path to directory where the new lightened images will be saved.
    val : float
        Threshold in [0,1] for the lighten_image function.
    adj : float
        Amount to lighten the dark regions by.
    """
    # Create output dir if it doesn't exist
    if not os.path.exists(lightened_dir):
        os.makedirs(lightened_dir)

    # List all images in frames_dir (adjust extension if needed)
    # For example, you might have "out-0001.png", "out-0002.png", etc.
    files = sorted(f for f in os.listdir(frames_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    i = 1
    for filename in files:
        input_path = os.path.join(frames_dir, filename)
        output_path = os.path.join(lightened_dir, filename)

        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: could not read image {input_path}, skipping.")
            continue

        # Lighten the image
        result = increase_brightness(img, val=val, adj=adj)

        # Save the resulting image
        cv2.imwrite(output_path, result)

        print(f"{i}: Saved lightened version of {filename}")
        i = i + 1


###########################################################################
#
# EXECUTE THE FUNCTION WITH APPROPRIATE INPUTS
# CHANGE THE PATHS IF YOU WANT TO DO THE ANALYSIS ON DIFFERENT IMAGES
#
###########################################################################
image_folder     = 'D:\\UAP-Projects\\Fairfield\\video\\frames'    # Path to the frames to BRIGHTEN/LIGHTEN
lightened_folder = 'D:\\UAP-Projects\\Fairfield\\video\\lightened' # Path to the RESULTING BRIGHTEN/LIGHTEN frames

process_frames(image_folder, lightened_folder, val=0.3, adj=6)
