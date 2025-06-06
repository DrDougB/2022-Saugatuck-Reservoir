#---------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024/2025: Douglas J. Buettner, PhD. GPL-3.0 license
# specific terms of this GPL-3.0 license can be found here:
# https://github.com/DrDougB/XXX
#
# Python code to read in each of the frames, extract a small region, and identify that same region in subsequent 
# frames. The script reads the start and stop frames and the image subsection locations to use from a csv data file.
# 
#
# The code was generated in part with GPT-4, OpenAI's large-scale language-generation
# model. Upon generating draft language, the author reviewed, edited, and revised the content 
# to his own liking (for example fixing errors) and takes ultimate responsibility for the content.
#
# OpenAI's Terms of Use statement dated December 11, 2024 (last visited December 27, 2024) is 
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
import csv
import cv2
import glob
import numpy as np


def readTemplateCSVFile(csv_file):
    """
    Reads a CSV file containing rows of:
      frame, centroid_x, centroid_y, half_width, half_height
    Returns a list of dicts, one per row.
    """
    specs = []
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip an optional header row
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue  # Skip incomplete rows
            frame       = int(row[0])
            centroid_x  = int(row[1])
            centroid_y  = int(row[2])
            half_width  = int(row[3])
            half_height = int(row[4])

            specs.append({
                'frame':       frame,
                'centroid_x':  centroid_x,
                'centroid_y':  centroid_y,
                'half_width':  half_width,
                'half_height': half_height
            })
    return specs

def extractTemplateImage(img, centroid_x, centroid_y, half_width, half_height):
    """
    Given an image and a bounding box defined by centroid and half-dimensions,
    extract and return the sub-image (template). 
    Assumes the bounding box is within the image boundaries.
    """
    x1 = centroid_x - half_width
    y1 = centroid_y - half_height
    x2 = centroid_x + half_width
    y2 = centroid_y + half_height
    template = img[y1:y2, x1:x2]
    return template

def matchTemplate(gray_img, gray_template, half_width, half_height, threshold=0.7):
    """
    Uses the multi-template-matching library's matchTemplates function to find the 
    best match of 'gray_template' in 'gray_img' with a given threshold. We only fetch
    the top single result (N_object=1).
    Useful documents:
    https://stackoverflow.com/questions/58158129/understanding-and-evaluating-template-matching-methods
    https://docs.opencv.org/4.0.0/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d

    Returns (best_x, best_y, best_score):
      - best_x, best_y is the top-left corner of the best match
      - best_score is the match score from MTM
      or (None, None, None) if no match above threshold is found.
    """
    # Perform match operations. 
    res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_SQDIFF_NORMED) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    center_x = min_loc[0] + half_width   
    center_y = min_loc[1] + half_height
    best_score = 1 - min_val
    
    if best_score < threshold:
        center_x   = None   
        center_y   = None
        best_score = None

    return (center_x, center_y, best_score)

def processFrames(
    frames_dir,
    csv_dir,
    specs_csv,
    threshold=0.7
):
    """
    1) Reads 'template_specs.csv' where each row has:
         frame, centroid_x, centroid_y, half_width, half_height
    2) For each row:
         - Load the specified template_frame from frames_dir ("out-####.png"),
           extract the sub-image as the template.
         - Enumerate ALL frames in frames_dir matching "out-*.png".
         - For each frame, run matchTemplate (mtm).
    """
    template_file_pattern = "out-{:04d}.png"

    # Read all specs from CSV
    specs = readTemplateCSVFile(specs_csv)

    # Gather all frames in frames_dir
    all_files = sorted(glob.glob(os.path.join(frames_dir, "out-*.png")))

    # Confirm the csv_dir exists, otherwise create it
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for spec_index, spec in enumerate(specs, start=1):
        frame_no = spec['frame']
        cx = spec['centroid_x']
        cy = spec['centroid_y']
        hw = spec['half_width']
        hh = spec['half_height']

        print(f"Extracting template from frame {frame_no} ...")

        # Output CSV for this line
        output_csv = os.path.join(csv_dir, f"{spec_index}_image_centroids.csv")
        
        with open(output_csv, 'w', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(["frame_name", "matched_centroid_x", "matched_centroid_y", "match_score"])

            # 1) Load the template frame
            template_path = os.path.join(frames_dir, template_file_pattern.format(frame_no))
            template_img  = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template_img is None:
                print(f"WARNING: Could not read template frame {template_path}, skipping spec #{spec_index}")
                continue

            # 2) Extract the template
            template = extractTemplateImage(template_img, cx, cy, hw, hh)
            if template.size == 0:
                print(f"WARNING: extracted template is empty for spec #{spec_index}, skipping.")
                continue
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            t_h, t_w = gray_template.shape
            print(f"Template shape: {gray_template.shape}")

            # 3) Loop over ALL frames, use multi-template-matching
            for frame_path in all_files:
                filename_only = os.path.basename(frame_path)
                full_img = cv2.imread(frame_path)
                if full_img is None:
                    print(f"WARNING: Could not read {frame_path}, skipping.")
                    continue
                gray_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

                matched_x, matched_y, score = matchTemplate(gray_img, gray_template, hw, hh)

                # If matched_x is None => no valid match
                if matched_x is not None:
                    print(f"Match specindex_file = {spec_index}_{filename_only}: center=({matched_x},{matched_y}), score={score:.3f}")
                    writer.writerow([filename_only, matched_x, matched_y, f"{score:.3f}"])

def main():
    frames_directory = 'PATH GOES HERE' # Path to the frames to search
    specs_csv_file   = 'PATH GOES HERE' # Path to the CSV template specs file
    csv_directory    = 'PATH GOES HERE' # Path to the CSV output file


    processFrames(
        frames_dir=frames_directory,
        csv_dir=csv_directory,
        specs_csv=specs_csv_file
    )

if __name__ == "__main__":
    main()
