
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn

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
        dim_feedforward = configs.transformer_head.dim_feedforward
        dropout = configs.transformer_head.dropout

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

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
        # 1. Get protein representation
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        protein_repr = outputs.last_hidden_state
        # 2. Retrieve ligand embedding
        ligand_repr = self.ligand_embedding(ligand_idx).unsqueeze(1)
        # 3. Pass through transformer
        transformer_output = self.transformer_decoder(
            tgt=protein_repr,
            memory=ligand_repr
        )
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
        ligand_idx = torch.tensor([0]).to(device)

        # Step 1: Input IDs and Tokens
        print("\n[1] Tokenized Input IDs:")
        print(inputs["input_ids"])
        print("\n[1.1] Tokens:")
        print(tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).tolist()))

        # Step 2: Get ESM2 hidden states
        outputs = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        print(f"\n[2] ESM2 Output Shape: {outputs.last_hidden_state.shape}")

        # Step 3: Ligand embedding
        ligand_repr = model.ligand_embedding(ligand_idx).unsqueeze(1)
        print(f"\n[3] Ligand Embedding Shape: {ligand_repr.shape}")
        print(f"[3.1] Ligand Embedding Vector: {ligand_repr.squeeze(1).cpu().numpy()}")

        # Step 4: Transformer Decoder (cross-attention)
        transformer_output = model.transformer_decoder(
            tgt=outputs.last_hidden_state,
            memory=ligand_repr
        )
        print(f"\n[4] Transformer Decoder Output Shape: {transformer_output.shape}")

        # Step 5: Final Classification
        logits = model.classifier(transformer_output)
        print(f"\n[5] Final Logits Shape: {logits.shape}")
        print(f"[5.1] Final Logits (Sample):\n{logits.squeeze(0)[:10]}")  # Show first 10 tokens for readability

    # print(model)