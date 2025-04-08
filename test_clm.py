import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


def load_tokenizer():
    """Load the MoLFormer tokenizer for chemical SMILES sequences."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", 
                                              add_prefix_space=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = '<pad>'
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.mask_token = '<unk>'
    return tokenizer


def load_model():
    """Load the MoLFormer model for chemical embeddings."""
    print("Loading model...")
    model = AutoModel.from_pretrained("ibm-research/MoLFormer-XL-both-10pct",
                                      trust_remote_code=True)
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
    return model


def make_last_n_blocks_trainable(model, n=3):
    """Make the last n transformer blocks trainable, keeping the rest frozen."""
    print(f"Making the last {n} blocks trainable...")
    
    # Freeze all parameters first (already done in load_model, but being explicit)
    for param in model.parameters():
        param.requires_grad = False
        
    # Make the last n layers trainable
    for param in model.encoder.layer[-n:].parameters():
        param.requires_grad = True
    
    # Count and display trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
          f"({100 * trainable_params / all_params:.2f}%)")
    
    return model


def tokenize_smiles(tokenizer, smiles, max_length=128):
    """Tokenize a SMILES sequence for input to the model."""
    print(f"Tokenizing SMILES: {smiles}")
    encoded = tokenizer(smiles, 
                       max_length=max_length,
                       padding='max_length',
                       truncation=True,
                       return_tensors="pt",
                       add_special_tokens=True)
    
    print(f"Input shape: {encoded['input_ids'].shape}")
    return encoded


def get_embeddings(model, encoded_input, output_dim=768):
    """Get embeddings from the model for the encoded input."""
    print("Getting embeddings from model...")
    
    with torch.no_grad():
        features = model(input_ids=encoded_input['input_ids'],
                        attention_mask=encoded_input['attention_mask'])
        
        # Shape [batch_size, sequence_length, embedding_dim]
        features_permuted = features.last_hidden_state
    
    return features_permuted


def main():
    # Sample SMILES sequence
    smiles = "c1nc(c2c(n1)n(cn2)[C@H]3C@@HO)N"
    
    # Step 1: Load tokenizer
    tokenizer = load_tokenizer()
    
    # Step 2: Load model
    model = load_model()
    
    # Step 3: Make last 3 blocks trainable
    model = make_last_n_blocks_trainable(model, n=3)
    
    # Step 4: Tokenize SMILES sequence
    encoded_input = tokenize_smiles(tokenizer, smiles)
    
    # Step 5: Get embeddings
    embeddings = get_embeddings(model, encoded_input)
    
    # Step 6: Print embeddings shape
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Optional: Print a small slice of the embeddings to verify content
    print("\nSample of embeddings (first 3 tokens, first 5 dimensions):")
    print(embeddings[0, :3, :5])


if __name__ == "__main__":
    main()
