import argparse
import torch
from huggingface_hub import hf_hub_download
from unimol2.models.unimol2 import UniMol2Model  # type: ignore
from unimol2.data.molecule_dataset import smi2_graph_features  # type: ignore
from rdkit.Chem import GetPeriodicTable

def _compute_padded_node_count(atom_counts):
    """
    Compute uniform node count for batching using the same block rounding as the finetune collater.

    Args:
        atom_counts (List[int]): Number of atoms (nodes) per molecule.
    Returns:
        int: Padded node count = (max_count + 1 + 3)//4*4 - 1
    """
    raw_max = max(atom_counts)
    return (raw_max + 1 + 3) // 4 * 4 - 1


def _pad_batch_features(feats, max_N):
    """
    Pad each molecule's feature dict in feats to the uniform node dimension max_N.
    Handles 1D, 2D, and 3D feature arrays per node or edge.
    """
    import numpy as _np
    padded = []
    for f in feats:
        pf = {}
        n = len(f['atoms_token'])
        for k, v in f.items():
            if k == 'atoms_token':
                pf[k] = v
                continue
            arr = _np.array(v)
            # pad per dimensionality
            if arr.ndim == 1:
                pf[k] = _np.pad(arr, (0, max_N - arr.shape[0]), constant_values=0)
            elif arr.ndim == 2:
                d0, d1 = arr.shape
                # square or bias
                if d0 == d1 or d0 == n + 1:
                    t0 = max_N + (1 if d0 == n + 1 else 0)
                    t1 = max_N + (1 if d1 == n + 1 else 0)
                    pf[k] = _np.pad(arr, ((0, t0 - d0), (0, t1 - d1)), constant_values=0)
                else:
                    pf[k] = _np.pad(arr, ((0, max_N - d0), (0, 0)), constant_values=0)
            elif arr.ndim == 3:
                p0, p1 = max_N - arr.shape[0], max_N - arr.shape[1]
                pf[k] = _np.pad(arr, ((0, p0), (0, p1), (0, 0)), constant_values=0)
            else:
                pf[k] = arr
        padded.append(pf)
    return padded

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

    This function handles variable-length molecules by:
      1. Extracting raw features per SMILES via smi2_graph_features.
      2. Computing a uniform node count (`max_N`) using block rounding (_compute_padded_node_count).
      3. Padding all per-molecule feature arrays to (`max_N`) with `_pad_batch_features`.
      4. Stacking padded tensors across the batch dimension.

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
        # 1. raw features per molecule
        feats = [smi2_graph_features(s) for s in smiles]
        # 2. compute batch-wide padded node count
        atom_counts = [len(f['atoms_token']) for f in feats]
        max_N = _compute_padded_node_count(atom_counts)
        # 3. pad all molecule features uniformly
        feats = _pad_batch_features(feats, max_N)
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
            # pad atomic numbers to max_N with pad index 0
            pad_len = max_N - len(nums)
            if pad_len > 0:
                nums = nums + [0] * pad_len
            tokens.append(torch.LongTensor(nums).unsqueeze(0).to(device))
        batched['src_token'] = torch.cat(tokens, dim=0)
        B = len(tokens)
        # src_pos is padded to (batch, max_N, 3)
        batched['src_pos'] = torch.zeros(B, max_N, 3, device=device)
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

