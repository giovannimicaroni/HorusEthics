import os
import pandas as pd

# current_directory = os.getcwd()

# # Define the additional path to be concatenated
# additional_path = "\RFW_dataset\images\test\test\data"

# Concatenate the current directory with the additional path
root_directory = 'C:/Users/g2317/OneDrive/Documentos/Unicamp/IC/Horus/cameradetection/KANFace/images/static_db'
#não consegui fazer o caminho ser relativo para cada pc

# Initialize a list to hold the data
data = []

# Iterate over each directory in the root directory
for subdir, _, files in os.walk(root_directory):
    for file in files:
        # Create the full path to the file
        file_path = os.path.join(subdir, file)
        full_path = file_path.replace('\\', '/')

        # Append the file path and a placeholder integer (e.g., 0) to the data list
        if '.png' in full_path:
            data.append([full_path, 0])

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=['image_path', 'ethnicity'])

# Save the DataFrame to a CSV file
output_csv_path = 'KANFace.csv'
df.to_csv(output_csv_path, index=False)

print(f"CSV file has been created at {output_csv_path}")
