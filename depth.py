import cv2
import bpy
import bpycv
import random
import numpy as np


# result["ycb_meta"] is 6d pose GT
counter=1
for obj in bpy.data.objects:
        if obj.type == "MESH":
            print(obj)
            obj = bpy.context.active_object
            obj["inst_id"] = counter
            counter=counter+1
# render image, instance annotation and depth in one line code
result = bpycv.render_data()

# write visualization inst|rgb|depth 
cv2.imwrite(
        "(inst|rgb|depth)." + str(counter) + ".jpg", cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR)
    )

# save result
cv2.imwrite(
    "demo-rgb.jpg", result["image"][..., ::-1]
)  # transfer RGB image to opencv's BGR

# save instance map as 16 bit png
cv2.imwrite("demo-inst.png", np.uint16(result["inst"]))
# the value of each pixel represents the inst_id of the object

# convert depth units from meters to millimeters
depth_in_mm = result["depth"] * 1000
cv2.imwrite("demo-depth.png", np.uint16(depth_in_mm))  # save as 16bit png

# visualization instance mask, RGB, depth for human
cv2.imwrite("demo-vis(inst_rgb_depth).jpg", result.vis()[..., ::-1])