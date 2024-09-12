import os
import shutil
if __name__ == '__main__':
    current_dir = os.getcwd()
    new_dir_name = f'RFWsummarized'
    new_dir_path = os.path.join(current_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    img_dir = f'/RFW_dataset/images/test/test/data'
    original_dir = 'C:\\Users\\g2317\\OneDrive\\Documentos\\Unicamp\\IC\\Horus\\cameradetection\\RFW_dataset\\images\\test\\test\\data'
    print(new_dir_path)

for root, _, files in os.walk(original_dir):
    for file in files:
        # Full path to the current file
        file_path = os.path.join(root, file)
        
        # Construct the full destination path
        destination_path = os.path.join(new_dir_path, file)
        
        # Copy the file to the destination directory
        shutil.copy(file_path, destination_path)
        print(f"Copied: {file_path} -> {destination_path}")
