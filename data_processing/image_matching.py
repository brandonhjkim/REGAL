# Matching each case with its satellite imagery
# Images were originally scraped by Russ White for DL4SAR but a lot of them got jumbled up. This script is to assign them together! 
import pandas as pd 
import shutil 
import os  

# read in data
df = pd.read_csv('cleaned_mra_with_weather.csv')

# adding UID's to match the image titles 
df['image_id'] = 'MRALabeled' + df['image_id'].astype(str)
df['image_id'] = df['image_id'].str[:-2]

# moving files to destination folder 
destination_folder = 'gis_files'
missing_img = [] 

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for idx, row in df.iterrows():
    source_path = '../unprocessed_data/GIS_data/' + row['image_id'] + '.tif'
        
    if os.path.isfile(source_path):
        filename = os.path.basename(source_path)
        destination_path = os.path.join(destination_folder, filename)

        try:
            # Move the file
            shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Error moving {source_path}: {e}")
    else:
        print(f"File not found: {source_path}")
        missing_img.append(row['image_id'])

remaining_df = df[~df['image_id'].isin(missing_img)]

# saving 
remaining_df.to_csv('mra_wi_img.csv', index = False)






