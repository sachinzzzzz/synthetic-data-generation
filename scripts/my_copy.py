import os
import shutil

def copy_bbox_to_datapoints(source_dir, dest_dir):
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    temp = 1
    # Iterate through directories inside the source directory
    for dir_name in os.listdir(source_dir):
        dir_path = os.path.join(source_dir, dir_name)
        
        # Check if it's a directory
        if os.path.isdir(dir_path):
            # Look for bbox.jpg in the current directory
            bbox_path = os.path.join(dir_path,'data2.jpg')

            # Check if bbox.jpg exists in the current directory
            if os.path.isfile(bbox_path):
                # Copy bbox.jpg to the destination directory
                dest_file_path = os.path.join(dest_dir, f"{dir_name}_bbox.jpg")
                shutil.copy(bbox_path, dest_file_path)
                print(f"bbox.jpg from '{dir_name}' copied to data_points successfully.")

# Example usage:
source_directory = r'E:\3D+animation\neo_human\results3'  # Assuming this is your source directory path
destination_directory = r'E:\3D+animation\neo_human\results\camera_2'  # Assuming this is your destination directory path
copy_bbox_to_datapoints(source_directory, destination_directory)