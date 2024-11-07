import os
import shutil

def regroup_files_into_subdirectories(directory, files_per_subdir=10):
    # Get the list of all files in the directory
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort the files to ensure consistency (optional)
    # all_files.sort()
    
    # Create subdirectories and move files
    for i in range(0, len(all_files), files_per_subdir):
        # Determine the subdirectory name
        subdir_name = f'subdir_{i//files_per_subdir + 1}'
        subdir_path = os.path.join(directory, subdir_name)
        
        # Create the subdirectory if it doesn't exist
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        # Move the files to the subdirectory
        for file in all_files[i:i + files_per_subdir]:
            shutil.move(os.path.join(directory, file), os.path.join(subdir_path, file))
    
    print("Files have been regrouped into subdirectories.")

# Example usage
cwd = os.getcwd()
directory_path = os.path.join(cwd, 'outputp5cameras/outputp5cameras')  # Replace with your directory path
regroup_files_into_subdirectories(directory_path)
