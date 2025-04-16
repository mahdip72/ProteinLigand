import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from multi_ligand_dataset import idx2ligand

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
        dtype = configs.model.dtype
        freeze_backbone = configs.model.freeze_backbone
        freeze_embeddings = configs.model.freeze_embeddings
        num_unfrozen_layers = configs.model.num_unfrozen_layers
        classifier_dropout_rate = configs.model.classifier_dropout_rate
        backbone_dropout_rate = configs.model.backbone_dropout_rate
        esm_to_decoder_dropout_rate = configs.model.last_state_dropout_rate
        num_ligands = configs.num_ligands
        ligand_names = configs.ligands
        # If true, use chemical encoder for ligand representation, else use embedding table
        self.use_chemical_encoder = configs.model.use_chemical_encoder
        # noise added to the ESM2 protein representation
        self.noise_std = configs.noise_std

        # 2. Load the pretrained transformer
        config = AutoConfig.from_pretrained(base_model_name)
        config.torch_dtype = dtype
        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)
        num_total_layers = len(self.base_model.encoder.layer)
        # option to load structure-aware model
        structure_aware = configs.model.structure_aware
        if structure_aware:
            checkpoint_path = "/home/dc57y/data/checkpoint_60.pth"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')["model_state_dict"]

            # remove '_orig_mod.protein_encoder.model.' from the keys
            checkpoint = {
                k.replace('_orig_mod.protein_encoder.model.', ''): v
                for k, v in checkpoint.items()
            }

            load_report = self.base_model.load_state_dict(checkpoint, strict=False)
            print("Loaded structure-aware weights")
            # print("Load report:", load_report)

        # 3. Freeze backbone if requested
        if freeze_backbone:
            if freeze_embeddings:
                # Freeze all layers (including embeddings)
                for param in self.base_model.parameters():
                    param.requires_grad = False
                # Unfreeze layers
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i >= num_total_layers - num_unfrozen_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                        # add dropout to unfrozen backbone layers
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)
            else:
                # Freeze requested layers and leave embeddings
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i < num_total_layers - num_unfrozen_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)

        # Final dropout layer for the last hidden state
        self.encoder_to_decoder_dropout = nn.Dropout(esm_to_decoder_dropout_rate)

        # 4. Ligand Embedding Table // Chemical Encoder
        if not self.use_chemical_encoder: # Use embedding table
            # Potentially could experiment with variable embedding_dim size
            print("Using embedding table for ligand representation")
            self.ligand_embedding = nn.Embedding(num_embeddings=num_ligands, embedding_dim=hidden_size)
        else: # Use chemical encoder
            print("Using chemical encoder for ligand representation")
            self.smiles_tokenizer = AutoTokenizer.from_pretrained(
                "ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
            self.smiles_tokenizer.pad_token = "<pad>"
            self.smiles_tokenizer.bos_token = "<s>"
            self.smiles_tokenizer.eos_token = "</s>"
            self.smiles_tokenizer.mask_token = "<unk>"
            clm_config = AutoConfig.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
            clm_config.torch_dtype = dtype
            clm_hidden_dropout = configs.model.clm_hidden_dropout_rate if configs.model.clm_hidden_dropout_rate else 0.0
            clm_embedding_dropout = configs.model.clm_embedding_dropout_rate if configs.model.clm_embedding_dropout_rate else 0.0
            clm_config.hidden_dropout_prob = clm_hidden_dropout
            clm_config.embedding_dropout_prob = clm_embedding_dropout
            self.smiles_model = AutoModel.from_pretrained(
                "ibm-research/MoLFormer-XL-both-10pct",
                config=clm_config,
                trust_remote_code=True
            )

            # Freeze everything by default (including embeddings)
            for param in self.smiles_model.parameters():
                param.requires_grad = False

            last_n = configs.model.chemical_encoder_num_unfrozen_layers if hasattr(configs.model, "chemical_encoder_num_unfrozen_layers") else 0
            for param in self.smiles_model.encoder.layer[-last_n:].parameters():
                param.requires_grad = True
            self.clm_max_length = configs.model.clm_max_length
            clm_output_dim = configs.model.clm_output_dim
            self.ligand_to_smiles = configs.ligand_to_smiles
            self.idx_to_ligand = idx2ligand(ligand_names)
            # Add projector to match hidden size
            self.projector = nn.Linear(clm_output_dim, hidden_size)
            self.proj_layernorm = nn.LayerNorm(hidden_size)

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
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout_rate)
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
        # protein_repr = outputs.last_hidden_state
        protein_repr = self.encoder_to_decoder_dropout(outputs.last_hidden_state)

        # Experimental: Adding noise to the protein representation as regularization
        if self.training:
            noise = torch.randn_like(protein_repr) * self.noise_std
            protein_repr = protein_repr + noise

        # 2. Retrieve ligand representation
        if self.use_chemical_encoder:
            ligand_names = [self.idx_to_ligand[i.item()] for i in ligand_idx]
            smiles_batch = [self.ligand_to_smiles[name] for name in ligand_names]
            encoded = self.smiles_tokenizer(smiles_batch,
                                            max_length=self.clm_max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors="pt",
                                            add_special_tokens=True).to(input_ids.device)
            ligand_hidden = self.smiles_model(**encoded).last_hidden_state
            ligand_repr = self.proj_layernorm(self.projector(ligand_hidden))
        else:
            ligand_repr = self.ligand_embedding(ligand_idx).unsqueeze(1)
        # 3. Pass through transformer
        if self.use_chemical_encoder:
            memory_key_padding_mask = encoded["attention_mask"] == 0  # pass in padding mask
            transformer_output = self.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr,
                memory_key_padding_mask=memory_key_padding_mask
            )
        else:
            transformer_output = self.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr
            )
        normalized = self.norm(transformer_output)
        dropout_output = self.dropout(normalized)
        logits = self.classifier(dropout_output)
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
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEY"
    labels = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.long)

    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=False)
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
        protein_repr = outputs.last_hidden_state
        print(f"\n[2] ESM2 Protein Representation Shape: {protein_repr.shape}")

        # Step 3: Ligand Representation
        if model.use_chemical_encoder:
            # Map ligand_idx to SMILES
            ligand_name = model.idx_to_ligand[ligand_idx.item()]
            smiles = model.ligand_to_smiles[ligand_name]
            print(f"\n[3] Ligand Name: {ligand_name}")
            print(f"[3.1] SMILES: {smiles}")

            # Tokenize SMILES
            encoded = model.smiles_tokenizer([smiles], max_length=model.clm_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
            ligand_hidden = model.smiles_model(**encoded).last_hidden_state
            ligand_repr = model.projector(ligand_hidden)

            print(f"[3.2] CLM Embedding Shape: {ligand_repr.shape}")

            memory_key_padding_mask = encoded["attention_mask"] == 0
            transformer_output = model.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr,
                memory_key_padding_mask=memory_key_padding_mask
            )

        else:
            ligand_repr = model.ligand_embedding(ligand_idx).unsqueeze(1)
            print(f"\n[3] Ligand Embedding Shape: {ligand_repr.shape}")
            print(f"[3.1] Ligand Embedding Vector: {ligand_repr.squeeze(1).cpu().numpy()}")

            transformer_output = model.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr
            )

        # Step 4: Transformer Decoder (cross-attention)
        print(f"\n[4] Transformer Decoder Output Shape: {transformer_output.shape}")

        # Step 5: Final Classification
        logits = model.classifier(transformer_output)
        print(f"\n[5] Final Logits Shape: {logits.shape}")
        print(f"[5.1] Final Logits (Sample):\n{logits.squeeze(0)[:10]}")  # Show first 10 tokens for readability

    # print(model)