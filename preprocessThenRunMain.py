import os
import numpy as np
from PIL import Image
import subprocess
from shutil import copyfile
def normalize_image(image_array):
    mean_value = np.mean(image_array)
    normalized_array = image_array / mean_value
    normalized_array = (normalized_array * 255 / np.max(normalized_array)).astype(np.uint8)
    return normalized_array


def process_folder(raw_folder_path, output_base_path):
    # Iterate over each sub-folder in the raw folder
    for sub_folder_name in os.listdir(raw_folder_path):
        sub_folder_path = os.path.join(raw_folder_path, sub_folder_name)

        # Ensure it's a directory
        if os.path.isdir(sub_folder_path):
            output_folder_path = os.path.join(output_base_path, f"{sub_folder_name}.data")

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder_path, exist_ok=True)

            for file_name in os.listdir(sub_folder_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    input_image_path = os.path.join(sub_folder_path, file_name)
                    output_image_path = os.path.join(output_folder_path, file_name)

                    # Open and normalize the image
                    if 'mask' in file_name:
                        copyfile(input_image_path,output_image_path)

                    else:
                        image = Image.open(input_image_path)
                        image_array = np.array(image, dtype=np.float32)
                        normalized_array = normalize_image(image_array)
                        normalized_image = Image.fromarray(normalized_array)

                    # Save the normalized image
                        normalized_image.save(output_image_path)
                        print(f"Normalized image saved to {output_image_path}")


# Define paths
raw_base_path = 'Raw'
output_base_path = 'YOUR_DATA_PATH'

# Process the base folder
process_folder(raw_base_path, output_base_path)

#arguments
session_name='session_test7'

#calls main.py which would generate the normal map
#session_name='session name'
subprocess.run(["python",'source/main.py','--session_name', session_name, '--mode', 'Test', '--test_dir', 'YOUR_DATA_PATH', '--pretrained', 'YOUR_CHECKPOINT_PATH'])

# Define the root folder where all subfolders are located
root_folder = os.path.join(r'C:\Users\Micah\Documents\GitHub\Universal-PS-CVPR2022\output', session_name)


#Loop to copy mask from data path to output folder
for sub_folder_name in os.listdir(output_base_path):
    source_path = os.path.join(output_base_path, sub_folder_name,'mask.png')
    copyfile(source_path,os.path.join(root_folder,sub_folder_name,'mask.png'))

# Iterate over each subfolder in the root folder
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    print(subfolder_path)
    if os.path.isdir(subfolder_path):
        print(subfolder_path)
        normal_image_path = os.path.join(subfolder_path, 'normal.png')
        mask_image_path = os.path.join(subfolder_path, 'mask.png')

        if os.path.isfile(normal_image_path) and os.path.isfile(mask_image_path):
            # Open the normal image
            normal_image = Image.open(normal_image_path)
            # Get the dimensions of the normal image
            normal_width, normal_height = normal_image.size
            # Calculate the coordinates for cropping (keep the right half)
            left = normal_width // 2
            top = 0
            right = normal_width
            bottom = normal_height
            # Crop the normal image
            normal_map = normal_image.crop((left, top, right, bottom))
            # Save the cropped normal image with the new name
            normal_map_path = os.path.join(subfolder_path, 'normal_map.png')
            normal_map.save(normal_map_path)

            # Open the mask image
            mask_image = Image.open(mask_image_path)
            # Resize the mask image to match the dimensions of the normal map
            resized_mask_image = mask_image.resize((normal_map.size))
            # Save the resized mask image
            resized_mask_image.save(mask_image_path)

            #generate depth map
            subprocess.run(['python',r'C:\Users\Micah\Documents\GitHub\bilateral_normal_integration\bilateral_normal_integration_numpy.py','--path',subfolder_path])
            print('Depth Map Done!')
