
import clip
import torch
# import preprocessing

# Load CLIP model and tokenizer
max_length=77
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)  # Load on CPU by default

def extract_clip_text_features(text_batch, device=device):
    """
    Extract text features using CLIP's text encoder.
    Args:
    - text_batch (list of str): List of sentences to extract features from.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
    - text_features (torch.Tensor): Encoded text features with shape [batch_size, 512].
    """
    # Tokenize the input text using CLIP's built-in tokenizer

    truncated_text_batch = [text[:max_length] for text in text_batch]

    inputs = clip.tokenize(truncated_text_batch,context_length=max_length).to(device)  # Tokenize and move to the specified device
    # Move the model to the device
    clip_model.to(device)
    
    # Disable gradient calculation during inference
    with torch.no_grad():
        # Encode the text inputs using CLIP's text encoder
        text_features = clip_model.encode_text(inputs)
        
        # Optionally normalize the features (CLIP usually normalizes embeddings)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features

# # Example usage
# text_batch = ["This is an example sentence.", "CLIP handles text and images together."]
# device = "cuda" if torch.cuda.is_available() else "cpu"
# text_embeddings = extract_clip_text_features(text_batch, device)

# print(f"Text embeddings shape: {text_embeddings.shape}")




