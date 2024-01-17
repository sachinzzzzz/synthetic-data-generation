import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import json
import math
from math import radians


class LightConfig:
    def __init__(self):
        self.config = {
            "location": (0, 0, 0),
            "rotation_euler": (0, 0, 0),
            "scale": (1, 1, 1),
            "energy": 1.0,
        }


def rotation_light():
    rot_config = LightConfig()
    
    # Randomize the light properties
    rot_config.config["location"] = (random.uniform(-6.5, 7.7), -15.11, 9.2)
    # rot_config.config["rotation_euler"] = (0, random.uniform(-0.95, 1.09), 0)
    # rot_config.config["scale"] = (random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2))
    rot_config.config["energy"] = random.uniform(20, 1000)

    return rot_config

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

def random_camera_focus():
    position1 = [0.91, -5.90, 4.88]
    position2 = [4.65, -5.90, 4.88]
    position3 = [-3.06, -5.90, 4.88]
    
    location = random.choice([position1, position2, position3])
    
    return location


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

# Specify the output directory
output_dir = "E:/3D+animation/neophyte/object/rendered images/two/"

counter = 1

# Number of iterations
num_iterations = 5


# List of instance IDs to exclude in visualization
exclude_instance_ids = [1, 3, 5]  # list of excluding ids (currently dummy list)

json_path = "E:/3D+animation/neophyte/object/object.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
        
data["products"] = {}
data["shelf"] = {}
data["testor"] = {}
data["brand"] = {}
data["banner"] = {}


products = 0
shelf = 1
testor = 2
banner = 3
brand =4

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

for iteration in range(1, num_iterations + 1):
    # Create a new folder for each iteration
    iteration_dir = os.path.join(output_dir, str(iteration))
    os.makedirs(iteration_dir, exist_ok=True)
    
    
    json_path = "E:/3D+animation/neophyte/object/object.json"
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        

    

    # Set the counter to 1 for each iteration
    counter = 1
    
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