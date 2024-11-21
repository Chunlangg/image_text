import torch
import clip
from PIL import Image

# Load the CLIP model and the preprocessing function
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

# Set the model to evaluation mode
clip_model.eval()

def extract_clip_image_features(image_path, device="cpu"):
    """
    Extract image features using CLIP's image encoder.

    Args:
    - image_path (str): Path to the image file.
    - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - image_features (torch.Tensor): The extracted image features.
    """
    # Open and preprocess the image
    img = Image.open(image_path)
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Move model to device
    clip_model.to(device)

    # Extract image features using CLIP's image encoder
    with torch.no_grad():  # Disable gradient calculation
        image_features = clip_model.encode_image(img_tensor)
        
        # Optionally normalize the image features (CLIP generally normalizes embeddings)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features

# Example usage:
# image_path = "example_image.jpg"
# image_features = extract_clip_image_features(image_path, device="cpu")
# print(f"Image features shape: {image_features.shape}")  # Expected output shape: [1, 512]
