# Renomeando os arquivos para fazer a clusterização
# Para fazer a divisão dos 30%, peguei para train até a pasta indian 991, caucasian 816, asian 818 e african 984

import os
import shutil

source_folder_base = r'C:\Users\emily\Downloads\RFW_dataset\separacao\train\indian' # substituir depois por asian, african e caucasian
destination_folder = r'C:\Users\emily\Downloads\RFW_dataset\separacao\total'

global_photo_counter = 1

# Loop de 00001 até 01000
for i in range(1, 1000):
    # Formata o número da subpasta
    subfolder_name = f'{i:05d}'
    subfolder_path = os.path.join(source_folder_base, subfolder_name)

    # Se existir
    if os.path.exists(subfolder_path):
        for filename in os.listdir(subfolder_path):
            source_file = os.path.join(subfolder_path, filename)
            # Novo nome do arquivo inclui o número da subpasta e o número da foto
            new_filename = f'in{subfolder_name}_{global_photo_counter:04d}.jpg' # in = indian, as = asian, wh = caucasian e bl = african
            destination_file = os.path.join(destination_folder, new_filename)
            shutil.copy2(source_file, destination_file)
            global_photo_counter += 1
   # exemplo: a foto 3 da subpasta 500 indian ficaria in00500_1715 (pois ela foi a foto 1715 a ser processada)

        print(f'Arquivos copiados e renomeados')
    else:
        print(f'Subpasta {subfolder_name} não encontrada')
