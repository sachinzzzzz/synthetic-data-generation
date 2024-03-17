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
# from bpyheadless import bpy


# read .blend file
# Set the path to your Blender file
blender_file_path = r"E:\3D+animation\neophyte\object\shelf - Copy.blend"

# Load the Blender file
bpy.ops.wm.open_mainfile(filepath=blender_file_path)

# Set the output file path for rendering
output_path = r"C:\Users\masti\OneDrive\Desktop\scripts\render_output.png"

# Set rendering parameters (optional)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = output_path

# Render the scene
bpy.ops.render.render(write_still=True)

# select all the mesh and enter to edit mode
# Deselect all objects first
bpy.ops.object.select_all(action='DESELECT')

# Select all mesh objects
bpy.ops.object.select_by_type(type='MESH')

# Switch to edit mode for all selected mesh objects
for obj in bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')



  #Render settings
# select which render engine -->

# select eevee
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  
# select cycles and set gpu
#bpy.context.scene.render.engine = 'CYCLES' ; bpy.context.scene.cycles.device = 'GPU'  
#set samples 
bpy.context.scene.cycles.samples = 128



# make a object for lightConfig
class LightConfig:
    def __init__(self):
        self.config = {
            "location": (0, 0, 0),
            "rotation_euler": (0, 0, 0),
            "scale": (1, 1, 1),
            "energy": 1.0,
        }


#function for light config
def rotation_light():
    rot_config = LightConfig()
    
    # Randomize the light properties
    rot_config.config["location"] = (random.uniform(-6.5, 7.7), -15.11, 9.2)
    # rot_config.config["rotation_euler"] = (0, random.uniform(-0.95, 1.09), 0)
    # rot_config.config["scale"] = (random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2))
    rot_config.config["energy"] = random.uniform(20, 1000)

    return rot_config



#function for camera config
def generate_random_camera_config():
    camera_height = random.uniform(0.3, 9.5)
    camera_location = [random.uniform(-2, 4), -12, camera_height]
    camera_rotation_euler = [radians(random.uniform(1.3735, 1.7156)), 0, radians(random.uniform(-0.5068, 0.5068))]
    camera_scale = [1, 1, 1]
    camera_lens_angle = radians(58.2)

    return {
        "camera_height": camera_height,
        "camera_location": camera_location,
        "camera_rotation_euler": camera_rotation_euler,
        "camera_scale": camera_scale,
        "camera_lens_angle": camera_lens_angle,
    }



# function to change camera focus point
def random_camera_focus():
    position1 = [0.91, -5.90, 4.88]
    position2 = [4.65, -5.90, 4.88]
    position3 = [-3.06, -5.90, 4.88]
    
    location = random.choice([position1, position2, position3])
    
    return location


# function to write json data
def visualize_instance_mask_exclude(instance_mask, exclude_instance_ids):
    colored_mask = np.zeros_like(result["image"])

    unique_instance_ids = np.unique(instance_mask)
    for instance_id in unique_instance_ids:
        if instance_id == 0 or instance_id in exclude_instance_ids:
            continue  # Skip background and excluded objects

        mask = (instance_mask == instance_id)
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        colored_mask[mask] = color
        
        for obj_name, obj_data in data["products"].items():
            if obj_data["inst_id"] == instance_id:
                obj_data["color"] = color.tolist()
                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)
                    
            
        for obj_name, obj_data in data["testor"].items():
            if obj_data["inst_id"] == instance_id:
                obj_data["color"] = color.tolist()
                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)
                    
        for obj_name, obj_data in data["banner"].items():
            if obj_data["inst_id"] == instance_id:
                obj_data["color"] = color.tolist()
                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)
        
        for obj_name, obj_data in data["shelf"].items():
            if obj_data["inst_id"] == instance_id:
                obj_data["color"] = color.tolist()
                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)
                    
        for obj_name, obj_data in data["brand"].items():
            if obj_data["inst_id"] == instance_id:
                obj_data["color"] = color.tolist()
                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)

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
            


# Specify the output directory
output_dir = "E:/3D+animation/neophyte/object/rendered images/two/"
base_path = "E:/3D+animation/neophyte/object/rendered images/two"





counter = 1

# Number of iterations
num_iterations = 6

num_folders = 6  # Set the number of folders you have

products = 0
shelf = 1
testor = 2
banner = 3
brand =4

json_path = "E:/3D+animation/neophyte/object/object.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

data["products"] = {}
data["shelf"] = {}
data["testor"] = {}
data["brand"] = {}
data["banner"] = {}




for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj["inst_id"] = counter
            obj_name = obj.name
            obj_id = counter
            counter += 1
            
            if "TESTOR" in obj_name:
                data["testor"][obj_name] = {
                "label": testor,
                "color": [],
                "inst_id": obj_id
            }
            elif "_brand_" in obj_name:
                data["brand"][obj_name] = {
                "label": brand,
                "color": [],
                "inst_id": obj_id
                }
            elif "SHELF" in obj_name:
                data["shelf"][obj_name] = {
                "label": shelf,
                "color": [],
                "inst_id": obj_id
                }
            elif "BANNER" in obj_name:
                data["banner"][obj_name] = {
                "label": banner,
                "color": [],
                "inst_id": obj_id
                }
            else:
                data["products"][obj_name] = {
                "label": products,
                "color": [],
                "inst_id": obj_id
            }
        else :
            obj["inst_id"] = 0



# List of instance IDs to exclude in visualization
            exclude_objects = [
 "shelf_light_ck" , "shelf_light_davi", "shelf_light_hugo"
, "shelf_light_boss",
"walldecor2", "wall", "walldecor1"] 
exclude_instance_ids = exclude_instance_ids = [bpy.data.objects[name]["inst_id"] for name in exclude_objects]  # list of excluding ids (currently dummy list)




for iteration in range(1, num_iterations + 1):
    # Create a new folder for each iteration
    iteration_dir = os.path.join(output_dir, str(iteration))
    os.makedirs(iteration_dir, exist_ok=True)
    
    
    json_path = "E:/3D+animation/neophyte/object/object.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        

    

    # Set the counter to 1 for each iteration
    
    #

    
    #

    # Set the output paths for the current iteration
    inst_path = os.path.join(iteration_dir, "instance.png")
    rgb_path = os.path.join(iteration_dir, "rgb.jpg")
    depth_path = os.path.join(iteration_dir, "depth.png")
    vis_path = os.path.join(iteration_dir, "inst_rgb_depth.jpg")
    color_inst_path = os.path.join(iteration_dir, "color_instance.jpg")
    
    #camera config
    random_camera_config = generate_random_camera_config()
    
    bpy.context.scene.camera.location = random_camera_config["camera_location"]
    bpy.context.scene.camera.rotation_euler = random_camera_config["camera_rotation_euler"]
    bpy.context.scene.camera.data.angle = random_camera_config["camera_lens_angle"]
    
    
    #light config
    rot = rotation_light()
    rot_light_obj = bpy.context.scene.objects["Area_rot"]
    # Apply the random light configuration directly in the loop
    rot_light_obj.location = rot.config["location"]
    #rot_light_obj.rotation_euler = rot.config["rotation_euler"]            # rotating light configuration
    # active_light.scale = random_light_config.config["scale"]
    rot_light_obj.data.energy = rot.config["energy"]


    # Render and save images
    result = bpycv.render_data()

    # Save visualization inst|rgb|depth
    cv2.imwrite(vis_path, cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR))

    # Save RGB image
    cv2.imwrite(rgb_path, result["image"][..., ::-1])

    # Save instance map as 16-bit png
    cv2.imwrite(inst_path, np.uint16(result["inst"]))

    # Convert depth units from meters to millimeters and save as 16-bit png
    depth_in_mm = result["depth"] * 1000
    cv2.imwrite(depth_path, np.uint16(depth_in_mm))

    # Visualize instance mask with exclusion and save as an image
    instance_mask = result["inst"]
    visualization = visualize_instance_mask_exclude(instance_mask, exclude_instance_ids)
    cv2.imwrite(color_inst_path, visualization)
    
    # Save a copy of the updated JSON file with a unique name
    updated_json_path = os.path.join(iteration_dir, "object_json.json")
    with open(updated_json_path, "w") as file:
        json.dump(data, file, indent=4)

    # Print status for the current iteration
    print("Iteration {}: Images saved in folder {}".format(iteration, iteration_dir))




for folder_num in range(1, num_folders + 1):
    folder_path = f"{base_path}/{folder_num}"
    inst_image_path = f"{folder_path}/instance.png"
    inst_json_path = f"{folder_path}/object_json.json"
    segmentation_label_path = f"{folder_path}/segmentation.txt"
    bbox_label_path = f"{folder_path}/bbox.txt"

    save_segmentation_and_bbox(inst_image_path, inst_json_path, segmentation_label_path, bbox_label_path)

    # Read the bounding box data
    with open(bbox_label_path, "r") as file:
        bbox_data = file.read()

  # Read the image
    image_path = os.path.join(folder_path, "rgb.jpg")
    image = cv2.imread(image_path)  

    # Iterate over each line in the bounding box data
    for line in bbox_data.split('\n'):
        if line.strip() == '':
            continue

        # Parse label and coordinates
        label, x_center, y_center, width, height = map(float, line.split())
        label = int(label)

        # Calculate bounding box coordinates
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        
        label_texts = {
            0: "Products",
            1: "shelf",
            2: "Testor",
            # Add more mappings as needed
        }

        label_colors = {
            0: (0, 255, 0),  # Example: green for label 0
            1: (255, 0, 0),  # Example: red for label 1
            2: (0, 255, 255),
            # Add more label-color mappings as needed
        }

        # Draw bounding box with a different color for each label
        color = label_colors.get(label, (0, 0, 255))  # Default to blue if label color is not defined
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        label_text = label_texts.get(label, f"Label: {label}")
        cv2.putText(image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with drawn rectangles
    output_image_path = os.path.join(folder_path, "output_image.jpg")
    cv2.imwrite(output_image_path, image)

    
  # Read the image
    image_path = os.path.join(folder_path, "rgb.jpg")
    image = cv2.imread(image_path)  
    # Read the segmentation mask
    with open(segmentation_label_path, "r") as file:
        segmentation_mask = file.read()

    # Iterate over each line in the segmentation mask
    for line in segmentation_mask.split('\n'):
        if line.strip() == '':
            continue

        # Parse label and coordinates
        label, *coords = map(float, line.split())
        label = int(label)

        # Reshape coordinates to form the rectangle
        coordinates = np.array(coords, dtype=np.float32).reshape(-1, 2)
        coordinates *= np.array([image.shape[1], image.shape[0]])

        # Draw rectangle with a different color for each label
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.polylines(image, [coordinates.astype(int)], isClosed=True, color=color, thickness=3)

    # Save the final image with segmentation mask
    final_output_image_path = os.path.join(folder_path, "final_output_image.jpg")
    cv2.imwrite(final_output_image_path, image)