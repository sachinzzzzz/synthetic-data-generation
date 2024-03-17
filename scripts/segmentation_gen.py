# import os
# import shutil

# def copy_bbox_to_datapoints(source_dir, dest_dir):
#     # Check if source directory exists
#     if not os.path.isdir(source_dir):
#         print(f"Source directory '{source_dir}' does not exist.")
#         return
    
#     # Create destination directory if it doesn't exist
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     temp = 1
#     # Iterate through directories inside the source directory
#     for dir_name in os.listdir(source_dir):
#         dir_path = os.path.join(source_dir, dir_name)
        
#         # Check if it's a directory
#         if os.path.isdir(dir_path):
#             # Look for bbox.jpg in the current directory
#             bbox_path = os.path.join(dir_path,'data_with_tracking.jpg')

#             # Check if bbox.jpg exists in the current directory
#             if os.path.isfile(bbox_path):
#                 # Copy bbox.jpg to the destination directory
#                 dest_file_path = os.path.join(dest_dir, f"{dir_name}_bbox.jpg")
#                 shutil.copy(bbox_path, dest_file_path)
#                 print(f"bbox.jpg from '{dir_name}' copied to data_points successfully.")

# # Example usage:
# source_directory = r'E:\3D+animation\neo_human\results'  # Assuming this is your source directory path
# destination_directory = r'E:\3D+animation\neo_human\results\datapoints2'  # Assuming this is your destination directory path
# copy_bbox_to_datapoints(source_directory, destination_directory)



# import cv2
# import bpy
# import bpycv
# import random
# import numpy as np
# import os
# import json
# import math
# from math import radians
# import cv2, imutils
# from matplotlib import pyplot as plt

# # #"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" -noaudio -b -P "E:/3D+animation/neo_human/scripts/gen.py"

# #load blender file
# blender_file_path = r"E:\3D+animation\neo_human\model_one.blend"
# bpy.ops.wm.open_mainfile(filepath=blender_file_path)


# #set rendering parameters
# output_path = r"E:\3D+animation\neo_human\results"
# #bpy.context.scene.render.image_settings.file_format = 'PNG'
# # bpy.context.scene.render.filepath = output_path


# #render setting
# bpy.context.scene.render.engine = 'BLENDER_EEVEE'  
# # select cycles and set gpu
# #bpy.context.scene.render.engine = 'CYCLES' ; bpy.context.scene.cycles.device = 'GPU'  
# #set samples 
# bpy.context.scene.cycles.samples = 128

# #render the scene
# #bpy.ops.render.render(write_still=True)



# def visualize_instance(instance_mask, include_instance_ids):
#     colored_mask = np.zeros_like(result["image"])
#     json_path = r"E:\3D+animation\neo_human\human.json"
#     opacity=0.5

#     unique_instance_ids = np.unique(instance_mask)
#     for instance_id in unique_instance_ids:
#         if instance_id == 0 or instance_id not in include_instance_ids:
#                 continue
#     mask = (instance_mask == instance_id)
#     color = [0, 0, 255]
#     color_with_opacity = color + [int(opacity * 255)]

#     for obj_name, obj_data in data["Human"].items():
#             if obj_data["inst_id"] == instance_id:
#                 obj_data["color"] = color
#                 with open(json_path, "w") as file:
#                     json.dump(data, file, indent=4)

#     colored_mask[mask] = color

#     return colored_mask

# def convert_coco_to_yolo(bbox, img_width, img_height):
#     x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
#     y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
#     w = (bbox[2] - bbox[0]) / img_width
#     h = (bbox[3] - bbox[1]) / img_height
#     return f"{x_center} {y_center} {w} {h}"

# def draw_motion_tracking(image, previous_bbox_centers, current_bbox_centers, color):
#     for prev_center, curr_center in zip(previous_bbox_centers, current_bbox_centers):
#         cv2.line(image, tuple(map(int, prev_center)), tuple(map(int, curr_center)), color, 2)
#     return image

# def save_segmentation_and_bbox(inst_image_path, inst_json_path, segmentation_label_path, bbox_label_path):

#     # get json objects
#     object_json = {}
    
#     with open(inst_json_path, "r") as file:
#         object_json = json.load(file)
        
#     label_dict = {}
#     labels = list(object_json.keys())
#     for label_name in labels:
#         object_dict = object_json[label_name]
    
#         for object_name in object_dict:
            
#             info_dict = object_dict[object_name]
#             # print(object_name, info_dict)
    
#             label_id = info_dict["label"]
#             inst_id = info_dict["inst_id"]
    
#             if label_id in label_dict:
#                 label_dict[label_id].append(inst_id)
    
#             else:
#                 label_dict[label_id] = [inst_id]

        
    
#     # read image  
#     image = cv2.imread(inst_image_path, cv2.IMREAD_UNCHANGED)
    
#     image = image.astype(np.uint8)
#     (img_h, img_w) = image.shape[:2]
#     print(f"shape: {(img_h, img_w)}")
#     unique, counts = np.unique(image, return_counts=True)
#     unique_count_dict = dict(zip(unique, counts))
    
#     # print("unique_count_dict", unique_count_dict)
#     for id in unique_count_dict:
#         # condition on the pixes
#         top_limit = (img_h*img_w) * 0.7 # 80 % of the data
#         top_limit = 700000
        
#         lower_limit = 500 # pixels
    
#         if unique_count_dict[id] < lower_limit or unique_count_dict[id] > top_limit:
#             print(f"removing id: {id}, pixel: {unique_count_dict[id]}")
#             continue
    
    
#         for label_id in label_dict:
#             inst_id_list = label_dict[label_id]
    
#             if id in inst_id_list:
#                 break
    
        
#         print(f"id: {id}, label: {labels[label_id]}, pixel: {unique_count_dict[id]}")
        
#         inst_image = image.copy()
#         mask = (inst_image != id)
#         inst_image[mask] = 0
        
#         mask = (inst_image == id)
#         inst_image[mask] = 200
    
#         ret, thresh = cv2.threshold(inst_image,100,255,0)
#         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
    
#         if len(cnts) == 0:
#             print(cnts)
    
#         # TODO: get multiple segmentation
#         c = max(cnts, key=cv2.contourArea)
#         area = cv2.contourArea(c)
#         # print("Area: ", area)
    
#         segmentation = c.reshape((-1, 2))
#         segmentation_str = f""
    
#         for count_1, seg in enumerate(segmentation):
    
#             x_cor = format(seg[0] / img_w, '.6f')
#             y_cor = format(seg[1] / img_h, '.6f')
    
#             if count_1 == 0:
#                 segmentation_str += f"{x_cor}"
#                 segmentation_str += f" {y_cor}"
#             else:
#                 segmentation_str += f" {x_cor}"
#                 segmentation_str += f" {y_cor}"
    
    
#         x,y,w,h = cv2.boundingRect(c) # COCO Bounding box: (x-top left, y-top left,width, height)
#         bbox = [x, y, x+w, y+h]
#         bbox_yolo = convert_coco_to_yolo(bbox, img_w, img_h)
    
    
#         # save segmentation label
#         if segmentation_str:
#             file_object = open(segmentation_label_path, "a")
#             file_object.write(f"{label_id} {segmentation_str}\n")
#             file_object.close()
    
#         # save bbox label
#         if bbox:
#             file_object = open(bbox_label_path, "a")
#             file_object.write(f"{label_id} {bbox_yolo}\n")
#             file_object.close()
    
#         # plt.figure(figsize = (6, 4))
#         # plt.imshow(inst_image * 50)
#         # plt.show()
#         # break
            
# output_dir = r"E:\3D+animation\neo_human\results"

# #loading json
# json_path = r"E:\3D+animation\neo_human\human.json"
# with open(json_path, 'r') as json_file:
#     data = json.load(json_file)

# data["Human"] = {}    

# counter = 1
# Human = 0  
         
# include = ["Human"]            
# for obj in bpy.data.objects:
#     if obj.type == "MESH" and obj.name in include:
#         obj["inst_id"] = counter
#         obj_name =  obj.name
#         obj_id = counter
#         counter += 1

#         if "Human" in obj_name:
#             data["Human"][obj_name] = {
#             "label" : Human,
#             "color" : [],
#             "inst_id" : obj_id 

#         } 
#     else :
#         obj["inst_id"] = 0  

# with open(json_path, "w") as file:
#      json.dump(data, file, indent=4)                    


# include_id = [bpy.data.objects[name]["inst_id"] for name in include]

# start_frame = 1
# end_frame = 200

# # Initialize previous bbox centers for motion tracking
# prev_bbox_centers = []

# for frame in range(start_frame, end_frame + 1):
#     iteration_dir = os.path.join(output_dir, str(frame))
#     os.makedirs(iteration_dir, exist_ok=True)
#     bpy.context.scene.frame_set(frame)
#     # bpy.context.scene.render.filepath = os.path.join(output_path, f"{frame}.png")
#     # bpy.ops.render.render(write_still=True)

#     json_path = r"E:\3D+animation\neo_human\human.json"
#     with open(json_path, 'r') as json_file:
#         data = json.load(json_file)
    

#     inst_path = os.path.join(iteration_dir, "instance.png")
#     depth_path = os.path.join(iteration_dir, "depth.png")
#     rgb_path = os.path.join(iteration_dir, "rgb.png")
#     vis_path = os.path.join(iteration_dir, "vis.png")
#     color_inst_path = os.path.join(iteration_dir, "color_inst.png")
#     segmentation_mask_path = os.path.join(iteration_dir, "segmentation.txt")
#     bbox_label_path = os.path.join(iteration_dir, "bbox.txt")

#     # Render and save images
#     result = bpycv.render_data()

#     # Save visualization inst|rgb|depth
#     cv2.imwrite(vis_path, cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR))

#     # Save RGB image
#     cv2.imwrite(rgb_path, result["image"][..., ::-1])

#         # Save instance map as 16-bit png
#     cv2.imwrite(inst_path, np.uint16(result["inst"]))

#     # Convert depth units from meters to millimeters and save as 16-bit png
#     depth_in_mm = result["depth"] * 1000
#     cv2.imwrite(depth_path, np.uint16(depth_in_mm))

#     instance_mask = result["inst"]
#     visualize = visualize_instance(instance_mask, include_id)
#     cv2.imwrite(color_inst_path, visualize)

#     updated_json_path = os.path.join(iteration_dir, "object_json.json")
#     with open(updated_json_path, "w") as file:
#         json.dump(data, file, indent=4)

#     save_segmentation_and_bbox(inst_path, updated_json_path, segmentation_mask_path, bbox_label_path) 

#     with open(bbox_label_path, "r") as file:
#         bbox_data = file.read()

#   # Read the image
#     image_path = os.path.join(iteration_dir, "rgb.png")
#     image = cv2.imread(image_path)  

#     current_bbox_centers = []  # Initialize current bbox centers

#     # Iterate over each line in the bounding box data
#     for line in bbox_data.split('\n'):
#         if line.strip() == '':
#             continue

#         # Parse label and coordinates
#         label, x_center, y_center, width, height = map(float, line.split())
#         label = int(label)

#         # Calculate bounding box coordinates
#         x_min = int((x_center - width / 2) * image.shape[1])
#         y_min = int((y_center - height / 2) * image.shape[0])
#         x_max = int((x_center + width / 2) * image.shape[1])
#         y_max = int((y_center + height / 2) * image.shape[0])

#         # Calculate bounding box center
#         bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
#         current_bbox_centers.append(bbox_center)  # Append current bbox center

#         label_texts = {
#             0: "person"
#             # Add more mappings as needed
#         }

#         label_colors = {
#             0: (0, 0, 255),  # Example: green for label 0
#             # Add more label-color mappings as needed
#         }

#         # Draw bounding box with a different color for each label
#         color = label_colors.get(label, (0, 0, 255))  # Default to blue if label color is not defined
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

#         label_text = label_texts.get(label, f"Label: {label}")
#         cv2.putText(image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Save the image with drawn rectangles
#     output_image_path = os.path.join(iteration_dir, "bbox.jpg")
#     cv2.imwrite(output_image_path, image)


#      # Read the image
#     image_path = os.path.join(iteration_dir, "bbox.jpg")
#     image = cv2.imread(image_path)  

#     # Read the segmentation mask
#     with open(segmentation_mask_path, "r") as file:
#         segmentation_mask = file.read()

#     # Iterate over each line in the segmentation mask
#     for line in segmentation_mask.split('\n'):
#         if line.strip() == '':
#             continue

#         # Parse label and coordinates
#         label, *coords = map(float, line.split())
#         label = int(label)

#         # Reshape coordinates to form the rectangle
#         coordinates = np.array(coords, dtype=np.float32).reshape(-1, 2)
#         coordinates *= np.array([image.shape[1], image.shape[0]])

#         # Draw rectangle with a different color for each label
#         color = tuple(map(int, np.random.randint(0, 255, 3)))
#         cv2.polylines(image, [coordinates.astype(int)], isClosed=True, color=color, thickness=3)

#     # Save the final image with segmentation mask
#     final_output_image_path = os.path.join(iteration_dir, "data.jpg")
#     cv2.imwrite(final_output_image_path, image)   

#     # Draw motion tracking lines
#     if frame > start_frame:
#         image_path = os.path.join(output_dir, str(frame-1), "color_inst.png")
#         prev_image = cv2.imread(image_path)
#         image_with_lines = draw_motion_tracking(prev_image, prev_bbox_centers, current_bbox_centers, (0, 255, 0))
#         output_image_with_lines_path = os.path.join(iteration_dir, "")
#         cv2.imwrite(output_image_with_lines_path, image_with_lines)

#     # Update previous bbox centers for next frame
#     prev_bbox_centers = current_bbox_centers.copy()

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

def visualize_instance(instance_mask, include_instance_ids):
    colored_mask = np.zeros_like(result["image"])
    json_path = r"E:\3D+animation\neo_human\human.json"

    unique_instance_ids = np.unique(instance_mask)
    for instance_id in unique_instance_ids:
        if instance_id == 0:
                continue
        mask = (instance_mask == instance_id)
        color = [0, 0, 255]

        for obj_name, obj_data in data["Human"].items():
                if obj_data["inst_id"] == instance_id:
                    obj_data["color"] = color
                    with open(json_path, "w") as file:
                        json.dump(data, file, indent=4)

        colored_mask[mask] = color

    return colored_mask

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
            file_object.write(f"{label_id} {bbox_yolo}\n")
            file_object.close()
    
        # plt.figure(figsize = (6, 4))
        # plt.imshow(inst_image * 50)
        # plt.show()
        # break
            
counter = 1
Human = 0

start_frame = 201
end_frame = 250
include = ["Human"]


for frame in range(start_frame, end_frame + 1):
    blender_file_path = r"E:\3D+animation\neo_human\cube.blend"
    bpy.ops.wm.open_mainfile(filepath=blender_file_path)

    output_dir = r"E:\3D+animation\neo_human\results3"
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_dir

# Update the scene after setting output path
    bpy.context.view_layer.update()            
#set active camera
    camera = bpy.data.objects["Camera_R"]   
    bpy.context.scene.camera = camera    

            
    json_path = r"E:\3D+animation\neo_human\human.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    data["Human"] = {}  
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)  

         

# Deselect all objects
# Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    print("All objects deselected.")

# Select all mesh objects except the one named "Human"
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and "Human" not in obj.name:
            obj.select_set(True)
            print(f"Selected object: {obj.name}")

# Switch to edit mode for all selected objects
    bpy.ops.object.mode_set(mode='EDIT')
    print("Switched to edit mode for selected objects.")

    for obj in bpy.data.objects:
        if obj.type == "MESH" and "Human" in obj.name:
            obj["inst_id"] = counter
            obj_name =  obj.name
            obj_id = counter
            counter += 1

            if "Human" in obj_name:
                data["Human"][obj_name] = {
                "label" : Human,
                "color" : [],
                "inst_id" : obj_id 

            } 
        else :
            obj["inst_id"] = 0 

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

    include_id = [bpy.data.objects[name]["inst_id"] for name in include]
    bpy.context.scene.frame_set(frame)
    # bpy.ops.render.render(animation=False, write_still=True)
    iteration_dir = os.path.join(output_dir, str(frame))
    os.makedirs(iteration_dir, exist_ok=True)


    inst_path = os.path.join(iteration_dir, "instance.png")
    depth_path = os.path.join(iteration_dir, "depth.png")
    rgb_path = os.path.join(iteration_dir, "rgb.png")
    vis_path = os.path.join(iteration_dir, "vis.png")
    color_inst_path = os.path.join(iteration_dir, "color_inst.png")
    segmentation_mask_path = os.path.join(iteration_dir, "segmentation.txt")
    bbox_label_path = os.path.join(iteration_dir, "bbox.txt")

    result = bpycv.render_data()
    bpy.ops.wm.quit_blender()
    counter = 1

    cv2.imwrite(vis_path, cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR))

    # Save RGB image
    cv2.imwrite(rgb_path, result["image"][..., ::-1])

        # Save instance map as 16-bit png
    cv2.imwrite(inst_path, np.uint16(result["inst"]))

    # Convert depth units from meters to millimeters and save as 16-bit png
    depth_in_mm = result["depth"] * 1000
    cv2.imwrite(depth_path, np.uint16(depth_in_mm))

    instance_mask = result["inst"]
    visualize = visualize_instance(instance_mask, include_id)
    cv2.imwrite(color_inst_path, visualize)

    updated_json_path = os.path.join(iteration_dir, "object_json.json")
    with open(updated_json_path, "w") as file:
        json.dump(data, file, indent=4)

    


print("Rendering complete.")
