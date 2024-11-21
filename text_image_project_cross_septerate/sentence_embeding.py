from transformers import BertTokenizer, BertModel
import torch

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_features(text_batch):
    inputs = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        text_features = outputs.last_hidden_state.mean(dim=1)
        text_features=text_features.squeeze(1)

        #print(f"the size of the text_feature: {text_features}")
    return text_features
