import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sentence_embeding import extract_text_features
from smat_util import extract_image_features
# from clip_image_embedding import extract_clip_image_features
# from clip_text_embedding import extract_clip_text_features
import torch

class MultimodalDataset(Dataset):

    # constructor

    def __init__(self, csv_file, image_column, text_column, label_column='Category', transform=None):
        # Load the dataset
        self.data = pd.read_csv(csv_file)

        # Remove duplicate rows based on image path and text columns
        self.data = self.data.drop_duplicates(subset=[image_column, text_column]).reset_index(drop=True)

        # Drop rows where the label is 0
        self.data = self.data[self.data[label_column] != 0].reset_index(drop=True)

        self.image_column = image_column
        self.text_column = text_column
        self.label_column = label_column
        self.transform = transform

    # return the data lenght, not the object
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and load the image features

        # get the image path
        image_path = self.data.iloc[idx][self.image_column]

        # get the 
        image_features = extract_image_features(image_path) if pd.notna(image_path) else None
        # image_features_1= extract_clip_image_features(image_path)
        # Get the text and load the text features
        text_data = self.data.iloc[idx][self.text_column]
        text_features = extract_text_features(text_data) if pd.notna(text_data) else None
        # text_features_1= extract_clip_text_features(text_data)
        # Get the label and adjust it from 1-6 to 0-5
        label = self.data.iloc[idx][self.label_column] - 1  # Convert labels 1-6 to 0-5
        #print(f"The label is : {label}")

        #return image_features, text_features, image_features_1,text_features_1, torch.tensor(label)
        return image_features, text_features, torch.tensor(label)
# Function to load the dataset and split into train/test sets
def load_data(csv_file, batch_size=32, test_split=0.2, shuffle=True):
    # Create the dataset from the CSV file
    dataset = MultimodalDataset(csv_file=csv_file, image_column="media_path", text_column="message_text")
    
    # Split the dataset into training and testing sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
