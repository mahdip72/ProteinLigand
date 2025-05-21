import argparse
import torch
from huggingface_hub import hf_hub_download
from unimol2.models.unimol2 import UniMol2Model  # type: ignore
from unimol2.data.molecule_dataset import smi2_graph_features  # type: ignore
from rdkit.Chem import GetPeriodicTable

# Predefined model configurations
MODEL_CONFIGS = {
    "84M": {
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 768,
        "encoder_attention_heads": 48,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "max_seq_len": 1024,
    },
    "164M": {
        "encoder_layers": 24,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 768,
        "encoder_attention_heads": 48,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "max_seq_len": 1024,
    },
    "310M": {
        "encoder_layers": 32,
        "encoder_embed_dim": 1024,
        "encoder_ffn_embed_dim": 1024,
        "encoder_attention_heads": 64,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "max_seq_len": 1024,
    },
    "570M": {
        "encoder_layers": 32,
        "encoder_embed_dim": 1536,
        "encoder_ffn_embed_dim": 1536,
        "encoder_attention_heads": 96,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "max_seq_len": 1024,
    },
    "1.1B": {
        "encoder_layers": 64,
        "encoder_embed_dim": 1536,
        "encoder_ffn_embed_dim": 1536,
        "encoder_attention_heads": 96,
        "pair_embed_dim": 512,
        "pair_hidden_dim": 64,
        "max_seq_len": 1024,
    },
}

REPO_ID = "dptech/Uni-Mol2"


def get_model_args(size: str) -> argparse.Namespace:
    """
    Create and return a model argument namespace for the specified UniMol2 model size.

    Parameters:
        size (str): Key identifying model size in MODEL_CONFIGS (e.g., "84M", "164M", "310M").

    Returns:
        argparse.Namespace: Namespace with architecture hyperparameters for UniMol2Model.
    """
    cfg = MODEL_CONFIGS[size]
    return argparse.Namespace(
        encoder_layers=cfg["encoder_layers"],
        encoder_embed_dim=cfg["encoder_embed_dim"],
        encoder_ffn_embed_dim=cfg["encoder_ffn_embed_dim"],
        encoder_attention_heads=cfg["encoder_attention_heads"],
        pair_embed_dim=cfg["pair_embed_dim"],
        pair_hidden_dim=cfg["pair_hidden_dim"],
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        emb_dropout=0.0,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len=cfg["max_seq_len"],
        post_ln=False,
        masked_token_loss=0.0,
        masked_dist_loss=0.0,
        masked_coord_loss=0.0,
        masked_coord_dist_loss=0.0,
        mode="infer",
        gaussian_std_width=1.0,
        gaussian_mean_start=0.0,
        gaussian_mean_stop=9.0,
        droppath_prob=0.0,
    )


def load_model(size: str, device: torch.device) -> UniMol2Model:
    """
    Download and load a pretrained UniMol2 model checkpoint by size, map to device, and set to evaluation mode.

    Parameters:
        size (str): Model size key matching MODEL_CONFIGS (e.g., "310M").
        device (torch.device): Target device for the model (CPU or CUDA device).

    Returns:
        UniMol2Model: Loaded model instance ready for inference.
    """
    args = get_model_args(size)
    model = UniMol2Model(args)
    filename = f"modelzoo/{size}/checkpoint.pt"
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    model.to(device)
    return model


def featurize(smiles, device: torch.device) -> dict:
    """
    Convert one or multiple SMILES strings into batched graph feature tensors for UniMol2Model.

    This routine calls smi2_graph_features for each SMILES, collects atom and edge features,
    stacks them into batch dimension, and builds model inputs:
      - node/atom features and masks
      - edge attributes and spatial relations
      - src_token: atomic numbers tensor (batch, N)
      - src_pos: placeholder positions tensor (batch, N, 3)

    Parameters:
        smiles (str or list of str): Single SMILES or list of SMILES molecules.
        device (torch.device): Device to place the returned tensors on.

    Returns:
        dict: Dictionary of batched torch.Tensors keyed by model input names.
    """
    # support single SMILES or batch of SMILES
    if isinstance(smiles, (list, tuple)):
        feats = [smi2_graph_features(s) for s in smiles]
        batched = {}
        pt = GetPeriodicTable()
        # keys except atomic tokens
        keys = [k for k in feats[0].keys() if k != 'atoms_token']
        for k in keys:
            tensors = [torch.tensor(f[k]).unsqueeze(0).to(device) for f in feats]
            batched[k] = torch.cat(tensors, dim=0)
        tokens = []
        for f in feats:
            nums = [pt.GetAtomicNumber(sym) for sym in f['atoms_token']]
            tokens.append(torch.LongTensor(nums).unsqueeze(0).to(device))
        batched['src_token'] = torch.cat(tokens, dim=0)
        B, N = batched['src_token'].size()
        batched['src_pos'] = torch.zeros(B, N, 3, device=device)
        return batched
    # single SMILES
    feat = smi2_graph_features(smiles)
    batched = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in feat.items() if k != 'atoms_token'}
    pt = GetPeriodicTable()
    nums = [pt.GetAtomicNumber(sym) for sym in feat['atoms_token']]
    batched['src_token'] = torch.LongTensor(nums).unsqueeze(0).to(device)
    N = batched['src_token'].size(1)
    batched['src_pos'] = torch.zeros(1, N, 3, device=device)
    return batched

