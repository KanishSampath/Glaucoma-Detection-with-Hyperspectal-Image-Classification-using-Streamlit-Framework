import os
import pandas as pd

# Define the base directory where your images are stored
base_dir = "eyepac-light-v2-512-jpg"  # Update this path as per your system
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Initialize lists to store the data for metadata
metadata = []

# Function to process a directory and generate metadata
def process_directory(directory, folder_name):
    # Loop over all subfolders (classes) in the directory
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        
        # Check if it's a folder
        if os.path.isdir(subfolder_path):
            # Loop over each image in the subfolder
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".jpg"):  # Assuming images are .jpg
                    file_path = os.path.join(subfolder_path, file_name)
                    label = subfolder
                    label_binary = 1 if label == 'RG' else 0  # Set binary label (1 for 'RG', 0 for others)
                    
                    # Prepare the row for the metadata
                    metadata.append({
                        "id": file_name.split('-')[1].split('.')[0],  # Extract ID from the filename
                        "file_name": file_name,
                        "label": label,
                        "label_binary": label_binary,
                        "folder": folder_name,
                        "source_dataset": "EyePACS-TRAIN",
                        "relative_file_type": "jpg",
                        "file_path": file_path.replace(base_dir, "").replace("\\", "/")
                    })

# Process both the training and validation directories
process_directory(train_dir, "train")
process_directory(val_dir, "validation")

# Create a DataFrame from the metadata list
metadata_df = pd.DataFrame(metadata)

# Save the DataFrame to a CSV file
metadata_df.to_csv("new_metadata.csv", index=False)

print("metadata.csv has been created successfully.")
