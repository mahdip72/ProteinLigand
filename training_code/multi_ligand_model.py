
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn

# Placeholder
# TODO: Refine to include masking, layer norm, feedforward, etc.
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Cross-attention mechanism where protein representation attends to ligand embeddings.

        Args:
            embed_dim (int): Dimension of embeddings (should match ESM2 output size).
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, protein_repr, ligand_repr):
        """
        Applies cross-attention where protein queries ligand information.

        Args:
            protein_repr (Tensor): Output from ESM2 (batch, seq_len, hidden_size).
            ligand_repr (Tensor): Ligand embedding (batch, 1, hidden_size).

        Returns:
            Tensor: Cross-attended protein representation.
        """
        attn_output, _ = self.cross_attention(protein_repr, ligand_repr, ligand_repr)
        return attn_output

class TransformerHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        """
        Transformer encoder with interleaved self-attention and cross-attention layers.

        Args:
            embed_dim (int): Model embedding size (should match ESM2 output size).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.cross_attention = CrossAttentionModule(embed_dim, num_heads)

        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True))
            self.layers.append(self.cross_attention)  # Interleaved Cross-Attention

    def forward(self, protein_repr, ligand_repr):
        """
        Forward pass through transformer encoder.

        Args:
            protein_repr (Tensor): Protein sequence representation.
            ligand_repr (Tensor): Ligand embedding.

        Returns:
            Tensor: Refined protein representation.
        """
        for layer in self.layers:
            protein_repr = layer(protein_repr)  # Apply self-attention on protein representation
            protein_repr = self.cross_attention(protein_repr, ligand_repr)  # Cross-attention with ligand

        return protein_repr  # Final refined representation

class LigandPredictionModel(nn.Module):
    def __init__(self, configs, num_labels=1):
        """
        Fine-tuning model for post-translational modification prediction.

        Args:
        Args:
            configs: Contains model configurations like
                     - model.model_name (str)
                     - model.hidden_size (int)
                     - model.freeze_backbone (bool, optional)
                     - model.dropout_rate (float, optional)
            num_labels (int): The number of output labels (default: 2 for binary classification).
        """
        super().__init__()

        # 1. Read from configs
        base_model_name = configs.model.model_name
        hidden_size = configs.model.hidden_size
        freeze_backbone = configs.model.freeze_backbone
        freeze_embeddings = configs.model.freeze_embeddings
        freeze_backbone_layers = configs.model.freeze_backbone_layers
        classifier_dropout_rate = configs.model.classifier_dropout_rate
        backbone_dropout_rate = configs.model.backbone_dropout_rate
        num_ligands = configs.num_ligands
        ligand_embedding_dim = configs.ligand_embedding_size

        # 2. Load the pretrained transformer
        config = AutoConfig.from_pretrained(base_model_name)
        config.torch_dtype = "bfloat16"
        config.classifier_dropout = classifier_dropout_rate
        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)

        # 3. Freeze backbone if requested
        if freeze_backbone:
            if freeze_embeddings:
                # Freeze all layers (including embeddings)
                for param in self.base_model.parameters():
                    param.requires_grad = False
                # Unfreeze layers
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i >= freeze_backbone_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                        # add dropout to unfrozen backbone layers
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)
                        # print("Unfreezing layer", i+i)
            else:
                # Freeze requested layers and leave embeddings
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i < freeze_backbone_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)

        # 4. Ligand Embedding Table
        # self.ligand_embedding = nn.Embedding(num_embeddings=num_ligands, embedding_dim=ligand_embedding_dim)
        self.ligand_embedding = nn.Embedding(num_embeddings=num_ligands, embedding_dim=hidden_size) # Testing with matching embedding and hidden size

        # 5. Transformer Head with Cross-Attention
        num_heads = configs.transformer_head.num_heads
        num_layers = configs.transformer_head.num_layers
        self.transformer_head = TransformerHead(embed_dim=hidden_size, num_heads=num_heads, num_layers=num_layers)

        # 6. Add classifier on top
        # self.classifier = nn.Linear(encoder_output_size, num_labels)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask,ligand_idx):
        """
        Forward pass for the Ligand prediction model.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask for padding.
            ligand_idx (Tensor): Index of the ligand type.

        Returns:
            Tensor: Logits for each residue in the input sequence.
        """
        # 1. Get Protein Representation from ESM2
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        protein_repr = outputs.last_hidden_state # Shape: [batch_size, seq_len, hidden_size]
        # 2. Retrieve Ligand Embedding
        ligand_repr = self.ligand_embedding(ligand_idx).unsqueeze(1) # Shape: [batch_size, 1, hidden_size]
        # 3. Pass Through Transformer Encoder (Cross-Attention via Memory)
        transformer_output = self.transformer_head(protein_repr, ligand_repr)
        logits = self.classifier(transformer_output)
        return logits

    def num_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def prepare_model(configs):
    """
    Prepares the ESM2 model and tokenizer based on given configurations.

    Args:
        configs (dict): A dictionary containing model configurations.
            Example keys:
                - "model_name" (str): The name of the ESM model to load (e.g., "facebook/esm2_t12_35M_UR50D").
                - "hidden_size" (int): The hidden size of the model

    Returns:
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model loaded with the specified configurations.
    """
    # Extract configurations
    model_name = configs.model.model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = LigandPredictionModel(configs)

    print(f"Loaded model: {model_name}")
    print(f"Model has {model.num_parameters():,} trainable parameters")

    return tokenizer, model

if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")

    from box import Box
    import yaml

    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_configs = Box(config_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = prepare_model(test_configs)
    model.to(device)

    # Define a sample amino acid sequence
    # randomly generated protein sequence, not really sure if it's valid
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEY"
    labels = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.long)


    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = labels.to(device)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        logits = model(**inputs)
        print(f"Logits shape: {logits.shape}")  # Shape: [batch_size, sequence_length, num_labels]
        # Print the logits tensor
        # print("Logits values:")
        # print(logits)

    # print(model)