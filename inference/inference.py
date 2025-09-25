import os, csv, json, yaml
import torch
import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from model.utils import load_configs, load_checkpoint
from model.model import prepare_model
from contextlib import nullcontext

# -------------------------
# Helpers
# -------------------------
def resolve_autocast_dtype(configs, device):
    if device.type != "cuda":
        return None
    use_mp = getattr(configs, "use_mixed_precision", False)
    if not use_mp:
        return None
    mp = str(getattr(configs, "mixed_precision_dtype", "")).lower()
    if mp in ("bf16", "bfloat16", "bfloat"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("Warning: bf16 not supported on this GPU.")
        return None
    if mp in ("fp32", "float32", "single"):
        return torch.float32
    return None

# -------------------------
# Dataset
# -------------------------
class InferenceDataset(Dataset):
    """
    Expects a dataframe with columns configured in YAML:
      - lig_name_col (str)         required
      - prot_seq_col (str)         required
      - smiles_col (str)           optional (only for Stage 3; model will ignore if not needed)
    """
    def __init__(self, df, tokenizer, max_length, lig_name_col, prot_seq_col, smiles_col=None):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.lig_col = lig_name_col
        self.seq_col = prot_seq_col
        self.smiles_col = smiles_col

        for col in [self.lig_col, self.seq_col]:
            if col not in self.df.columns:
                raise ValueError(f"Input CSV must include '{col}'. Found: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        lig = str(row[self.lig_col])
        smiles = None
        if self.smiles_col and self.smiles_col in row and pd.notna(row[self.smiles_col]):
            smiles = row[self.smiles_col]

        enc = self.tok(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sequence_len": len(seq),
            "ligand_name": lig,
            "smiles": smiles,
        }

# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def run_inference(model, tokenizer, device, df, configs):
    seq_len        = int(getattr(configs, "sequence_length", 512))
    batch_size     = int(getattr(configs, "batch_size", 8))
    pred_threshold = float(getattr(configs, "prediction_threshold", 0.5))
    lig_name_col   = getattr(configs, "lig_name_col", "ligand_name")
    prot_seq_col   = getattr(configs, "prot_seq_col", "protein_sequence")
    smiles_col     = getattr(configs, "smiles_col", None)
    out_probs      = bool(getattr(configs, "output_binding_probs", True))
    stage3         = bool(getattr(configs, "stage_3", False))
    ac_dtype = resolve_autocast_dtype(configs, device)

    dataset = InferenceDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=seq_len,
        lig_name_col=lig_name_col,
        prot_seq_col=prot_seq_col,
        smiles_col=smiles_col
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ligand2idx = None
    if not stage3:
        train_ligands = list(getattr(configs, "ligands", []))
        if not train_ligands:
            raise ValueError("Stage 2 inference requires 'configs.ligands' to be set.")
        ligand2idx = {lig: i for i, lig in enumerate(sorted(set(train_ligands)))}

        unknown = set(df[lig_name_col].unique()) - set(ligand2idx.keys())
        if unknown:
            raise ValueError(
                "Stage 2 checkpoint requires all ligands to be known. "
                f"Unknown ligand names in input: {sorted(list(unknown))}. "
                "Use a Stage 3 checkpoint or filter your input."
                "Refer to example_data.csv for the list of the 166 supported ligands."
            )

    n = len(df)
    all_pred_json    = [""] * n
    all_pos_indices  = [""] * n
    all_probs_json   = [""] * n if out_probs else None

    model.eval()
    offset = 0
    for batch in tqdm.tqdm(loader, desc="Inference"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_lens       = batch["sequence_len"]
        lig_names      = batch["ligand_name"]
        smiles_list    = batch.get("smiles", None)

        if stage3:
            ligand_idx = torch.full((input_ids.size(0),), -1, dtype=torch.long, device=device)

            # Prefer SMILES from CSV; if missing, try model.ligand_to_smiles
            if smiles_list is None:
                smiles_list = [None] * len(lig_names)

            if hasattr(model, "ligand_to_smiles") and isinstance(model.ligand_to_smiles, dict):
                smiles_fallback = [
                    s if isinstance(s, str) and len(s) > 0 else model.ligand_to_smiles.get(name, None)
                    for s, name in zip(smiles_list, lig_names)
                ]
            else:
                smiles_fallback = smiles_list

            if any(s is None for s in smiles_fallback):
                missing_for = [name for s, name in zip(smiles_fallback, lig_names) if s is None]
                raise ValueError(
                    "Stage 3 requires SMILES per ligand. "
                    f"Missing SMILES for ligands: {sorted(set(missing_for))}. "
                    "Provide SMILES in the input CSV."
                )

            ligand_smiles = smiles_fallback

        else:
            idxs = [ligand2idx[name] for name in lig_names]
            ligand_idx   = torch.tensor(idxs, dtype=torch.long, device=device)
            ligand_smiles = None

        ac_ctx = (
            autocast(device_type="cuda", dtype=ac_dtype)
            if (device.type == "cuda" and ac_dtype is not None)
            else nullcontext()
        )
        with ac_ctx:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ligand_idx=ligand_idx,
                ligand_smiles=ligand_smiles,
            )

        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > pred_threshold).to(torch.int32)
        preds_np = preds.cpu().numpy()
        probs_np = probs.to(torch.float32).cpu().numpy()

        for b in range(len(seq_lens)):
            L = int(seq_lens[b])
            pred_list = preds_np[b][:L].tolist()
            all_pred_json[offset + b] = json.dumps(pred_list)
            pos_idx = [i for i, val in enumerate(pred_list) if val == 1]
            all_pos_indices[offset + b] = json.dumps(pos_idx)
            if out_probs:
                probs_list = [float(f"{p:.4f}") for p in probs_np[b][:L]]
                all_probs_json[offset + b] = json.dumps(probs_list)

        offset += len(seq_lens)

    out_df = df.copy()
    out_df["predictions"] = all_pred_json
    out_df["positive_indices"] = all_pos_indices
    if out_probs:
        out_df["binding_probabilities"] = all_probs_json

    return out_df

# -------------------------
# Main
# -------------------------
def main():
    with open("./configs/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    configs = load_configs(cfg_dict)

    dev_str = getattr(configs, "device_type", "cuda")
    device = torch.device(dev_str if torch.cuda.is_available() and "cuda" in dev_str else "cpu")

    tokenizer, model = prepare_model(configs)

    ckpt = getattr(configs, "checkpoint_path", None)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"Loading checkpoint: {ckpt}")
    _ = load_checkpoint(ckpt, model)

    model.to(device)

    input_csv = getattr(configs, "input_csv_path", None)
    output_csv = getattr(configs, "output_csv_path", None)
    if not input_csv or not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not output_csv:
        raise ValueError("Config must define 'output_csv_path'")

    df = pd.read_csv(input_csv)

    out_df = run_inference(model, tokenizer, device, df, configs)

    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    main()
