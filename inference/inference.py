#!/usr/bin/env python3
import os
import yaml
import torch
import tqdm
import pandas as pd
import json, csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from model.utils import load_configs, load_checkpoint
from model.model import prepare_model


# =========================
# USER SETTINGS (edit these)
# =========================
ZERO_SHOT        = True # Set to True if using Stage 3 checkpoint, False if using Stage 2 checkpoint
CONFIG_PATH      = "./configs/config.yaml"
# CHECKPOINT_PATH  = "./placeholder_checkpoint.pth"
CHECKPOINT_PATH  = "/mnt/pixstor/data/dc57y/2025-08-12__13-19-32__MOLFORMER_1280_ESM2_ZERO_SHOT/checkpoints/stage_3.pth"

INPUT_CSV        = "./data/example_data.csv"     # must have: ligand_name, protein_sequence; optional: SMILES
OUTPUT_CSV       = "./data/output.csv"

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 4
PRED_THRESHOLD   = 0.5

# If None, will use configs.train_settings.max_sequence_length or 512
MAX_LENGTH       = None

# CSV column names (change if your file uses different headers)
COL_LIGAND       = "ligand_name"
COL_SEQUENCE     = "protein_sequence"
COL_SMILES       = "SMILES"      # optional

BINDING_PROBS = True # TODO


# =========================
# Dataset
# =========================
class InferenceDataset(Dataset):
    """
    Expects a dataframe with columns:
      - ligand_name (str)           required
      - protein_sequence (str)      required
      - SMILES (str)                optional
    """
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length

        for col in [COL_LIGAND, COL_SEQUENCE]:
            if col not in self.df.columns:
                raise ValueError(f"Input CSV must include '{col}' column. Found: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = str(row[COL_SEQUENCE])
        lig = str(row[COL_LIGAND])
        smiles = row[COL_SMILES] if (COL_SMILES in row and pd.notna(row[COL_SMILES])) else None

        enc = self.tok(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_len": len(seq),
            "ligand_name": lig,
            "smiles": smiles,
        }


# =========================
# Inference
# =========================
@torch.no_grad()
def run_inference(model, tokenizer, device, df, output_csv_path,
                  pred_threshold=0.5, batch_size=256, max_length=512, configs=None):
    dataset = InferenceDataset(df, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Mixed precision from configs if available
    mixed_precision = None
    if configs is not None:
        if getattr(configs, "use_mixed_precision", False):
            mixed_precision = getattr(configs, "mixed_precision_dtype", None)

    model.eval()
    all_pred_str = [""] * len(df)
    all_pos_indices = [""] * len(df)
    all_probs_str = [""] * len(df)

    offset = 0
    for batch in tqdm.tqdm(loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_lens = batch["sequence_len"]
        smiles_list = batch.get("smiles", None)

        ligand_idx = torch.full((input_ids.size(0),), -1, dtype=torch.long, device=device)

        mp = str(mixed_precision).lower() if mixed_precision is not None else None
        if mp in ("fp16", "float16"):
            ac_dtype = torch.float16
        elif mp in ("bf16", "bfloat16"):
            ac_dtype = torch.bfloat16
        elif mp in ("fp32", "float32"):
            ac_dtype = torch.float32
        else:
            ac_dtype = None

        with autocast(device_type=device.type, dtype=ac_dtype):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ligand_idx=ligand_idx,
                ligand_smiles=[s if isinstance(s, str) else None for s in smiles_list] if smiles_list is not None else None,
            )

        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > pred_threshold).long().cpu().numpy()
        probs_np = probs.to(torch.float32).cpu().numpy()
        probs_rounded = np.round(probs_np, 4)

        for b in range(len(seq_lens)):
            L = int(seq_lens[b])

            pred_list = preds[b][:L].tolist()
            all_pred_str[offset + b] = json.dumps(pred_list)

            pos_idx = [i for i, val in enumerate(preds[b][:L]) if val == 1]
            all_pos_indices[offset + b] = json.dumps(pos_idx)

            probs_list = [float(f"{p:.4f}") for p in probs_rounded[b][:L]]
            all_probs_str[offset + b] = json.dumps(probs_list)

        offset += len(seq_lens)

    out_df = df.copy()
    out_df["predictions"] = all_pred_str
    out_df["positive_indices"] = all_pos_indices
    out_df["binding_probabilities"] = all_probs_str

    out_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved predictions to: {output_csv_path}")


def main():
    with open(CONFIG_PATH, "r") as f:
        cfg_dict = yaml.safe_load(f)
    configs = load_configs(cfg_dict)

    if hasattr(configs, "sequence_length"):
        max_len = configs.sequence_length
    else:
        raise ValueError("Config must define 'sequence_length'.")

    tokenizer, model = prepare_model(configs)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    _ = load_checkpoint(CHECKPOINT_PATH, model)

    device = torch.device(DEVICE)
    model.to(device)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    run_inference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        df=df,
        output_csv_path=OUTPUT_CSV,
        pred_threshold=PRED_THRESHOLD,
        batch_size=BATCH_SIZE,
        max_length=max_len,
        configs=configs,
    )


if __name__ == "__main__":
    main()
