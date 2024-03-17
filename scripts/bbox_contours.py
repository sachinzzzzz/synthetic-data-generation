import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import json
import math
from math import radians
import cv2, imutils
from matplotlib import pyplot as plt

def convert_coco_to_yolo(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
    w = (bbox[2] - bbox[0]) / img_width
    h = (bbox[3] - bbox[1]) / img_height
    return f"{x_center} {y_center} {w} {h}"



def save_segmentation_and_bbox(inst_image_path, inst_json_path, segmentation_label_path, bbox_label_path):

    # get json objects
    object_json = {}
    
    with open(inst_json_path, "r") as file:
        object_json = json.load(file)
        
    label_dict = {}
    labels = list(object_json.keys())
    for label_name in labels:
        object_dict = object_json[label_name]
    
        for object_name in object_dict:
            
            info_dict = object_dict[object_name]
            # print(object_name, info_dict)
    
            label_id = info_dict["label"]
            inst_id = info_dict["inst_id"]
    
            if label_id in label_dict:
                label_dict[label_id].append(inst_id)
    
            else:
                label_dict[label_id] = [inst_id]

        
    
    # read image  
    image = cv2.imread(inst_image_path, cv2.IMREAD_UNCHANGED)
    
    image = image.astype(np.uint8)
    (img_h, img_w) = image.shape[:2]
    print(f"shape: {(img_h, img_w)}")
    unique, counts = np.unique(image, return_counts=True)
    unique_count_dict = dict(zip(unique, counts))
    
    # print("unique_count_dict", unique_count_dict)
    for id in unique_count_dict:
        # condition on the pixes
        top_limit = (img_h*img_w) * 0.7 # 80 % of the data
        top_limit = 700000
        
        lower_limit = 500 # pixels
    
        if unique_count_dict[id] < lower_limit or unique_count_dict[id] > top_limit:
            print(f"removing id: {id}, pixel: {unique_count_dict[id]}")
            continue
    
    
        for label_id in label_dict:
            inst_id_list = label_dict[label_id]
    
            if id in inst_id_list:
                break
    
        
        print(f"id: {id}, label: {labels[label_id]}, pixel: {unique_count_dict[id]}")
        
        inst_image = image.copy()
        mask = (inst_image != id)
        inst_image[mask] = 0
        
        mask = (inst_image == id)
        inst_image[mask] = 200
    
        ret, thresh = cv2.threshold(inst_image,100,255,0)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        if len(cnts) == 0:
            print(cnts)
    
        # TODO: get multiple segmentation
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # print("Area: ", area)
    
        segmentation = c.reshape((-1, 2))
        segmentation_str = f""
    
        for count_1, seg in enumerate(segmentation):
    
            x_cor = format(seg[0] / img_w, '.6f')
            y_cor = format(seg[1] / img_h, '.6f')
    
            if count_1 == 0:
                segmentation_str += f"{x_cor}"
                segmentation_str += f" {y_cor}"
            else:
                segmentation_str += f" {x_cor}"
                segmentation_str += f" {y_cor}"
    
    
        x,y,w,h = cv2.boundingRect(c) # COCO Bounding box: (x-top left, y-top left,width, height)
        bbox = [x, y, x+w, y+h]
        bbox_yolo = convert_coco_to_yolo(bbox, img_w, img_h)
    
    
        # save segmentation label
        if segmentation_str:
            file_object = open(segmentation_label_path, "a")
            file_object.write(f"{label_id} {segmentation_str}\n")
            file_object.close()
    
        # save bbox label
        if bbox:
            file_object = open(bbox_label_path, "a")
            file_object.write(f"{label_id} {id} {bbox_yolo}\n")
            file_object.close()
    
        # plt.figure(figsize = (6, 4))
        # plt.imshow(inst_image * 50)
        # plt.show()
        # break
            

num_folders = 250
base_path = r"E:\3D+animation\neo_human\results3"

for folder_num in range(1  , num_folders+1):
    iteration_dir = f"{base_path}/{folder_num}"
    inst_image_path = f"{iteration_dir}/instance.png"
    inst_json_path = f"{iteration_dir}/object_json.json"
    segmentation_label_path = f"{iteration_dir}/segmentation.txt"
    bbox_label_path = f"{iteration_dir}/bbox.txt"


    save_segmentation_and_bbox(inst_image_path, inst_json_path, segmentation_label_path, bbox_label_path)

    if not os.path.exists(bbox_label_path):
    # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(bbox_label_path), exist_ok=True)
    # Create the empty bbox.txt file
        with open(bbox_label_path, 'w') as file:
            pass  # This will create an empty file

    with open(bbox_label_path, "r") as file:
        bbox_data = file.read()

  # Read the image
    image_path = os.path.join(iteration_dir, "rgb.png")
    image = cv2.imread(image_path)  

    # Iterate over each line in the bounding box data
    for line in bbox_data.split('\n'):
        if line.strip() == '':
            continue

        # Parse label and coordinates
        # label, id, x_center, y_center, width, height = map(float, line.split())
        # label = int(label)
        # id = int(id)

        try:
            label, id, x_center, y_center, width, height = map(float, line.split())
        # Rest of the processing logic...
        except ValueError:
            print(f"Error parsing line: {line}")

        # Calculate bounding box coordinates
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        
        label_texts = {
            0: f"{id}"
            # Add more mappings as needed
        }

        label_colors = {
            0: (0, 0, 255),  # Example: green for label 0
            # Add more label-color mappings as needed
        }

        # Draw bounding box with a different color for each label
        color = label_colors.get(label, (0, 0, 255))  # Default to blue if label color is not defined
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        label_text = label_texts.get(label, f"Label: {label}")
        cv2.putText(image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with drawn rectangles
    output_image_path = os.path.join(iteration_dir, "bbox.jpg")
    cv2.imwrite(output_image_path, image)

    # Read the image
    image_path = os.path.join(iteration_dir, "bbox.jpg")
    image = cv2.imread(image_path)  

    if not os.path.exists(segmentation_label_path):
    # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(segmentation_label_path), exist_ok=True)
    # Create the empty bbox.txt file
        with open(segmentation_label_path, 'w') as file:
            pass  # This will create an empty file

# Read the segmentation mask
    with open(segmentation_label_path, "r") as file:
        segmentation_mask = file.read()

# Create a blank mask with the same size as the image
    mask = np.zeros_like(image)

# Specify the desired color (in BGR format)
    red_color = (0, 0, 255)

# Iterate over each line in the segmentation mask
    for line in segmentation_mask.split('\n'):
        if line.strip() == '':
            continue

    # Parse label and coordinates
        label, *coords = map(float, line.split())
        label = int(label)

    # Reshape coordinates to form the polygon
        coordinates = np.array(coords, dtype=np.float32).reshape(-1, 2)
        coordinates *= np.array([image.shape[1], image.shape[0]])

    # Draw filled polygon with the specified red color
        cv2.fillPoly(mask, [coordinates.astype(int)], color=red_color)

    # Draw boundary with opacity on the image
        # cv2.polylines(image, [coordinates.astype(int)], isClosed=True, color=red_color, thickness=3)

# Apply opacity to the filled regions
    opacity = 0.4  # Adjust the opacity as needed
    cv2.addWeighted(mask, opacity, image, 1 - opacity, 0, image)

# Save the final image with segmentation mask
    final_output_image_path = os.path.join(iteration_dir, "data2.jpg")
    cv2.imwrite(final_output_image_path, image)


    previous_bounding_boxes = {}
    previous_centroids = {}

# Iterate over each line in the bounding box data
    for line in bbox_data.split('\n'):
        if line.strip() == '':
            continue

    # Parse label and coordinates
        label, id, x_center, y_center, width, height = map(float, line.split())
        label = int(label)
        id = int(id)

    # Calculate bounding box coordinates
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])

    # Calculate centroid
        centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)

    # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # Store current bounding box coordinates and centroid in the dictionary
        if label not in previous_bounding_boxes:
            previous_bounding_boxes[label] = []
            previous_centroids[label] = []

        previous_bounding_boxes[label].append(((x_min, y_min), (x_max, y_max)))
        previous_centroids[label].append(centroid)

    # If the list exceeds length 10, remove the oldest entry
        if len(previous_bounding_boxes[label]) > 10:
            previous_bounding_boxes[label].pop(0)
            previous_centroids[label].pop(0)

# Draw dotted lines connecting centroids across consecutive frames
    for label, centroids in previous_centroids.items():
        for i in range(len(centroids) - 1):
            cv2.line(image, centroids[i], centroids[i + 1], color, 1, cv2.LINE_AA)

# Save the image with drawn rectangles and motion tracking lines
    output_image_path = os.path.join(iteration_dir, "bbox_with_motion_tracking.jpg")
    cv2.imwrite(output_image_path, image)