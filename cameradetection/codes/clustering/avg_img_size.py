import numpy as np
import os
from PIL import Image

def calculate_average_image_size(directory):
    widths = []
    heights = []
    i = 0
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            i += 1
            # Construct the full path to the image file
            file_path = os.path.join(root, file)
            if '.png' in file_path:
                with Image.open(file_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    print(f'{i} images done')
    
    # Calculate the average width and height using NumPy
    width_array = np.array(widths)
    height_array = np.array(heights)
    average_width = np.mean(width_array)
    average_height = np.mean(height_array)
    
    return average_width, average_height


if __name__ == '__main__':
    avg_w, avg_h = calculate_average_image_size('C:\\Users\\g2317\\OneDrive\\Documentos\\Unicamp\\IC\\Horus\\cameradetection\\KANFace\\images\\static_db')
    print(f'{avg_w} x {avg_h}')