from PIL import Image
import os
from IMG_Dataset import IMG_Dataset

if __name__ == '__main__':
    dataset = IMG_Dataset(csv_file='5camera.csv')
    for i in range (0, len(dataset)):
        image, ethnicitiy = dataset[i]
        img_path = dataset.get_img_path(i)
        width, height = image.size
        cropped_image = image.crop(((width - 1024) // 2, (height - 1024) // 2, (width + 1024) // 2, (height + 1024) // 2))
        # Extract the base name without the extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Create the new filename
        new_filename = f'cropped{base_name}.jpg'
        cropped_image.save(new_filename)