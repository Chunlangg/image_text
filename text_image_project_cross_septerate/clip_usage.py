import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from clip_image_embedding import extract_clip_image_features
from clip_text_embedding import extract_clip_text_features
import torch.nn as nn

# Define the custom dataset to load from CSV file
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, device="cpu"):
        # Read the CSV file using pandas
        self.data = pd.read_csv(csv_file)
        self.device = device

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path, text, and category (label) from CSV file
        media_path = self.data.iloc[idx]['media_path']
        message_text = self.data.iloc[idx]['message_text']
        category = self.data.iloc[idx]['Category'] - 1  # Change to 0-based index (0-4 instead of 1-5)
        
        # Return the image path, message text, and category as label
        return media_path, message_text, torch.tensor(category, dtype=torch.long)

# Define the classifier model
class MultimodalClassifierWithCLIPAttention(nn.Module):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, num_classes, device="cpu"):  # 
        super(MultimodalClassifierWithCLIPAttention, self).__init__()

        # Device for computation
        self.device = device
        
        # Fully connected layers to reduce CLIP embeddings from 512 to 256 dimensions
        self.fc_image = nn.Linear(512, 256)
        self.fc_text = nn.Linear(512, 256)

        # Attention layer to weigh image and text features
        self.attention_weights = nn.Linear(512, 2)  # Two weights: one for image, one for text

        # Final classification layer
        self.fc_fusion = nn.Linear(256, num_classes)

    def forward(self, image_path, text_batch):
        # Extract image features using external function
        image_features = extract_clip_image_features(image_path, self.device)

        # Extract text features using external function
        text_features = extract_clip_text_features(text_batch, self.device)

        # Process image and text features through fully connected layers
        image_output = torch.relu(self.fc_image(image_features))
        text_output = torch.relu(self.fc_text(text_features))

        # Concatenate image and text features along the feature dimension
        combined_features = torch.cat((image_output, text_output), dim=1)  # Shape: [batch_size, 512]

        # Apply attention mechanism to weigh image and text features
        attention_scores = torch.softmax(self.attention_weights(combined_features), dim=1)
        image_weight, text_weight = attention_scores[:, 0], attention_scores[:, 1]

        # Expand weights to match the feature size
        image_weight = image_weight.unsqueeze(1)  # Shape: [batch_size, 1]
        text_weight = text_weight.unsqueeze(1)    # Shape: [batch_size, 1]

        # Apply the attention weights to image and text features
        fused_output = image_weight * image_output + text_weight * text_output  # Shape: [batch_size, 256]

        # Final classification layer
        output = self.fc_fusion(fused_output)

        return output

# Example usage:
if __name__ == "__main__":
    # Set device to CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model with 5 output classes (corresponding to categories 1-5)
    num_classes = 5  # Category ranges from 1 to 5
    model = MultimodalClassifierWithCLIPAttention(num_classes=num_classes, device=device)

    # Load the dataset from a CSV file
    dataset = MultimodalDataset(csv_file='output_final_data.csv', device=device)

    # Create a DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Iterate over the DataLoader for training
    for media_path, message_text, category in dataloader:
        # Forward pass
        output = model(media_path[0], message_text)  # media_path[0] to convert list to string
        loss = criterion(output, category)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}, Output shape: {output.shape}")  # Output shape should be [1, 5]
