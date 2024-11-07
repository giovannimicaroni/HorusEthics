import os
import pandas as pd

# current_directory = os.getcwd()

# # Define the additional path to be concatenated
# additional_path = "\RFW_dataset\images\test\test\data"

# Concatenate the current directory with the additional path
root_directory = 'C:/Users/g2317/OneDrive/Documentos/Unicamp/IC/Horus/cameradetection/RFW_dataset/images/test/test/data'
#não consegui fazer o caminho ser relativo para cada pc

# Initialize a list to hold the data
data = []

# Iterate over each directory in the root directory
for subdir, _, files in os.walk(root_directory):
    for file in files:
        # Create the full path to the file
        file_path = os.path.join(subdir, file)
        full_path = file_path.replace('\\', '/')
        if ('African') in file_path:
            int_ethnicity = 0
        elif ('Asian') in file_path:
            int_ethnicity = 1
        elif ('Caucasian') in file_path:
            int_ethnicity = 2
        elif ('Indian') in file_path:
            int_ethnicity = 3
        
        # Append the file path and a placeholder integer (e.g., 0) to the data list
        data.append([full_path, int_ethnicity])

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=['image_path', 'ethnicity'])

# Save the DataFrame to a CSV file
output_csv_path = 'RFW.csv'
df.to_csv(output_csv_path, index=False)

print(f"CSV file has been created at {output_csv_path}")
