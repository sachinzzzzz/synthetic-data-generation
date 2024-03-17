import bpy
import mathutils

def calculate_projected_area(camera_obj, mesh_obj):
    # Get camera data
    cam = camera_obj.data
    
    # Get camera view matrix
    view_matrix = camera_obj.matrix_world.inverted()
    
    # Get mesh vertices in world space
    vertices = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    
    projected_area = 0.0
    
    # Iterate through mesh polygons
    for poly in mesh_obj.data.polygons:
        # Get polygon vertices in world space
        poly_vertices = [vertices[i] for i in poly.vertices]
        
        # Calculate the normal of the polygon
        poly_normal = poly.normal.normalized()
        
        # Check if the polygon is facing the camera
        if poly_normal.dot(view_matrix @ poly_center(poly_vertices) - camera_obj.location) < 0:
            continue
        
        # Calculate the projected area of the polygon
        projected_area += poly_area_projected(poly_vertices, camera_obj)
    
    return projected_area

def poly_center(vertices):
    return sum(vertices, mathutils.Vector()) / len(vertices)

def poly_area_projected(vertices, camera_obj):
    # Get camera data
    cam = camera_obj.data
    
    # Get camera view matrix
    view_matrix = camera_obj.matrix_world.inverted()
    
    # Project polygon vertices onto camera image plane
    projected_vertices = [view_matrix @ v for v in vertices]
    
    # Calculate projected area using Shoelace formula
    area = 0.0
    n = len(projected_vertices)
    for i in range(n):
        j = (i + 1) % n
        area += projected_vertices[i].x * projected_vertices[j].y - projected_vertices[j].x * projected_vertices[i].y
    
    return abs(area) / 2

# Specify the path to the .blend file
blend_file_path = r"E:\3D+animation\untitled.blend"

# Open the .blend file
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# Get the camera and mesh objects from the scene
camera_obj = bpy.context.scene.camera
mesh_obj = bpy.context.scene.objects['Cube']  # Change 'Cube' to the name of your mesh object

# Calculate the projected area
projected_area = calculate_projected_area(camera_obj, mesh_obj)
print("Projected area:", projected_area)

