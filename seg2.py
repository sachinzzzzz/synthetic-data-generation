import cv2
import bpy
import bpycv
import random
import numpy as np
import pandas as pd
import json



#different list for testor
testor_boss = ["bossTESTOR_t1p1_3","bossTESTOR_t1p1_2", "bossTESTOR_t1p1_1", "bossTESTOR_t4p1_2", 
          "bossTESTOR_t4p1", "bossTESTOR_t5p1_2", "bossTESTOR_t5p1_1" ]

testor_hugo = ["hugoTESTOR_t1p2_2", "hugoTESTOR_t1p2_1",
                "hugoTESTOR_t1p1_2", "hugoTESTOR_t1p1_1", "hugoTESTOR_t2p1_1", "hugoTESTOR_t2p1_2", "hugoTESTOR_t3p1_2",
                "hugoTESTOR_t3p1_1", "hugoTESTOR_t4p1", "hugoTESTOR_t5p1"]


testor_davidoff = ["davidoffTESTOR_t1p1", "davidoffTESTOR_t2p1", "davidoffTESTOR_t3p1_1",
                  "davidoffTESTOR_t3p1_2", "davidoffTESTOR_t4p1_1", "davidoffTESTOR_t4p1_2",
                   "davidoffTESTOR_t5p1_2", "davidoffTESTOR_t5p1_1" ]


testor_ck = ["ckTESTOR_t1p1_2", "ckTESTOR_t1p1_1", "ckTESTOR_t2p1", "ckTESTOR_t3p2_1", "ckTESTOR_t3p2_2",
             "ckTESTOR_t3p1_1", "ckTESTOR_t4p1","ckTESTOR_t5p1"]
             
wall = ["walldecor2", "wall", "walldecor1"]
banner= ["BOSS_BANNER", "HUGO_BANNER", "DAVIDOFF_BANNER", "CK_BANNER"]
brand = ["CK_brand_name", "devidoff_brand_name", "hugo_brand_name" , "boss_brand_name"]
shelf = ["BOSS_SHELF", "HUGO_SHELF", "DAVIDOFF_SHELF", "DAVIDOFF_SHELF"]
             
#add all testor
all_testors = []
all_testors.extend(testor_boss)
all_testors.extend(testor_hugo)
all_testors.extend(testor_davidoff)
all_testors.extend(testor_ck)

#load json
#df = pd.read_json("E:/3D+animation/neophyte/object/object.json")
json_path = "E:/3D+animation/neophyte/object/object.json"
with open(json_path, 'r') as json_file:
    data = json.load(json_file)
#//clear exsting data
data["products"] = {}
data["shelf"] = {}
data["testor"] = {}
data["brand"] = {}
data["banner"] = {}

# List of object names to exclude
exclude_objects = [
 "shelf_light_ck" , "shelf_light_davi", "shelf_light_hugo"
, "shelf_light_boss",
"walldecor2", "wall", "walldecor1"] 

## Extend the exclude_objects list with all_testors
#exclude_objects.extend(all_testors)


# result["ycb_meta"] is 6d pose GT
counter = 1
#label
products = 0
shelf = 1
testor = 2
banner = 3
brand =4


for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.name not in exclude_objects:
        obj["inst_id"] = counter
        obj_name = obj.name
        obj_id = counter
        counter += 1
#        obj = bpy.context.active_object

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

#take the inst_id of excluded objects
exclude_instance_ids = [bpy.data.objects[name]["inst_id"] for name in exclude_objects]


#update with object properties except color
with open(json_path, "w") as file:
    json.dump(data, file, indent=4)
        

# render image, instance annotation and depth in one line code
result = bpycv.render_data()
instance_mask = result["inst"]

# write visualization inst|rgb|depth 
cv2.imwrite(
    "(inst|rgb|depth)." + str(counter) + ".jpg", cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR)
)

# save result
cv2.imwrite(
    "rgb.jpg", result["image"][..., ::-1]
)  # transfer RGB image to opencv's BGR

# save instance map as 16 bit png
cv2.imwrite("instance.png", np.uint16(result["inst"]))
# the value of each pixel represents the inst_id of the object

# convert depth units from meters to millimeters
depth_in_mm = result["depth"] * 1000
cv2.imwrite("depth.png", np.uint16(depth_in_mm))  # save as 16bit png

# visualization instance mask, RGB, depth for human
cv2.imwrite("inst_rgb_depth.jpg", result.vis()[..., ::-1])


#create a visulalization seperately for instance mask and update json value
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


# Visualize the instance mask excluding the specified objects
visualization = visualize_instance_mask_exclude(instance_mask, exclude_instance_ids)
# Save the visualization as an image
cv2.imwrite("color_instance.jpg", visualization)
