import bpy
import json


json_file_path = r"E:\3D+animation\neophyte\object\object.json"


with open(json_file_path, "r") as file:
    data = json.load(file)


counter = 1
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj_name = obj.name
        obj_id = counter
        counter += 1

        
        data["products"][obj_name] = {
            "label": counter,
            "color": [],
            "inst_id": obj_id
        }


with open(json_file_path, "w") as file:
    json.dump(data, file, indent=4)


