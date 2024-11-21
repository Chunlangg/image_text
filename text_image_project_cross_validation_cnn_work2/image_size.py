import pandas as pd
from PIL import Image

# Load your CSV file that contains the image paths (change the path as needed)
csv_path = "/Users/chunlan/Research/simple_project_newest /code/image_text/output_final_data.csv"
df = pd.read_csv(csv_path)
print(df.head)

# Function to get the size of the image
def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        return None  # In case the image is not found or cannot be opened

# Assuming there is a column in the CSV with the image file paths, let's call it 'image_path'
df['image_size'] = df['media_path'].apply(get_image_size)
df.to_csv('updated_final_file_with_image_size.csv', index=False)
# Display the first few rows to check the image sizes
print(df.head())
