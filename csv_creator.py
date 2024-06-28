import csv

# With an input with the total number of images, creates a csv file that adapts the 
# Synthetics-DisCo 3 database to be applicable in Fairface 

def create_csv(n, filename='output.csv'):
    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['img_path'])
        
        # Write the data rows
        for i in range(n + 1): 
            path_string = 'Synthetics-DisCo_sg2_n10k_3/Synthetics-DisCo/sg2_n10k_arc_r14_lang_v1/images_arcface_112x112/' + str(i).zfill(5) + '/reference.png'
            writer.writerow([path_string])

# Get input from the user
n = int(input("Enter the number of images: "))

# Create the CSV file
create_csv(n)

print(f"CSV file created with {n + 1} rows (0 to {n}).")
