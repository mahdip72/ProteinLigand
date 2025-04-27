import torch
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd
from transformers import AutoTokenizer
import os

def build_ligand2idx(ligand_names):
    """
    Takes a list of ligand names and returns {ligand_name: index} dictionary.
    """
    return {ligand: idx for idx, ligand in enumerate(sorted(set(ligand_names)))}

def idx2ligand(ligand_names):
    """
    Takes a list of ligand names and returns {index: ligand_name} dictionary.
    """
    return {idx: ligand for idx, ligand in enumerate(sorted(set(ligand_names)))}


def read_fasta(file_path):
    """
    Reads a FASTA file and returns a dictionary mapping sequence IDs to sequences.
    """
    sequences = {}
    with open(file_path, "r") as file:
        current_id = None
        current_seq = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)  # Save last sequence
    return sequences


class LigandDataset(Dataset):
    def __init__(self, ligand_list, data_root, tokenizer, ligand2idx, split="train", max_length=512, subset_size=None, use_precompiled_data = False, use_dynamic_SMILES=False):
        """
        Args:
            ligand_list (list): List of ligand names like ["ADP", "ATP", ...]
            data_root (str): Root directory containing ligand folders. Note: File paths are automatically generated and expect a strict naming convention
            tokenizer: Hugging Face tokenizer.
            ligand2idx (dict): Mapping from ligand name to index.
            split (str): One of 'train', 'eval', or 'test'.
            max_length (int): Max sequence length.
            subset_size (int): If provided, randomly samples this many entries from the dataset.
            use_precompiled_data (bool): If True, uses precompiled data CSV file instead of FASTA files. Used for PLINDER
            use_dynamic_SMILES (bool): If True, uses dynamic SMILES representation based on the given dataset
        """

        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ligand2idx = ligand2idx

        if not use_precompiled_data:
            for ligand in ligand_list:
                ligand_dir = os.path.join(data_root, ligand)
                seq_file = os.path.join(ligand_dir, f"{split}_set.fasta")
                label_file = os.path.join(ligand_dir, f"{split}_labels.fasta")

                sequences = read_fasta(seq_file)
                labels = read_fasta(label_file)

                for seq_id in sequences:
                    if seq_id in labels and len(sequences[seq_id]) == len(labels[seq_id]):
                        self.data.append({
                            "sequence": sequences[seq_id],
                            "label": labels[seq_id],
                            "ligand": ligand,
                            "ligand_idx": ligand2idx[ligand]
                        })

        else:
            csv_map = {
                # "train": "top_10_train.csv",
                # "eval": "top_10_val.csv",
                # "test": "top_10_test.csv",
                "train": "all_ligands_100_threshold_train.csv",
                "eval": "all_ligands_100_threshold_val.csv",
                "test": "all_ligands_100_threshold_test.csv"
            }
            # csv_dir = os.path.join(data_root, "top_10")
            csv_dir = os.path.join(data_root, "PLINDER_181")
            csv_path = os.path.join(csv_dir, csv_map[split])
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                ligand = row["ligand_unique_ccd_code"]
                if ligand in ligand2idx:
                    sequence = row["protein_sequence"]
                    label_str = row["binding_sites_labels"]
                    if isinstance(label_str, str):
                        label_str = label_str.strip('"')
                        if len(sequence) == len(label_str):
                            sample = {
                                "sequence": sequence,
                                "label": label_str,
                                "ligand": ligand,
                                "ligand_idx": ligand2idx[ligand]
                            }
                            if use_dynamic_SMILES:
                                sample["smiles"] = row["ligand_rdkit_canonical_smiles"]
                            self.data.append(sample)

        if subset_size is not None:
            import random
            self.data = random.sample(self.data, subset_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Tokenize sequence
        inputs = self.tokenizer(entry["sequence"], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.max_length, add_special_tokens=False)

        # Convert label string to tensor
        labels = torch.tensor([int(c) for c in entry["label"]], dtype=torch.long)
        # Pad or truncate labels to max_length
        if len(labels) < self.max_length:
            labels = torch.cat([labels, torch.zeros(self.max_length - len(labels), dtype=torch.long)])
        else:
            labels = labels[:self.max_length]

        batch = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "ligand_idx": entry["ligand_idx"],
        }

        # If SMILES representation is needed (For PLINDER dataset where some ligands have different SMILES)
        if "smiles" in entry:
            batch["smiles"] = entry["smiles"]

        return batch

    def get_amino_acid_token_ids(self):
        vocab = self.tokenizer.get_vocab()
        special_ids = self.tokenizer.all_special_ids if hasattr(self.tokenizer, "all_special_ids") else [
            self.tokenizer.pad_token_id, self.tokenizer.mask_token_id]
        return [v for k, v in vocab.items() if v not in special_ids]

    # For testing
    def print_tokenizer_vocab(self):
        vocab = self.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # sort by token ID
        for token, token_id in sorted_vocab:
            print(f"{token_id:>4}: {token}")


def prepare_dataloaders(configs, debug=False, debug_subset_size=None, use_precompiled_data = False):
    """
    Prepares DataLoaders for training, validation, and testing based on configurations.

    Args:
        configs: Configuration object containing file paths and DataLoader settings.

    Returns:
        dict: A dictionary containing DataLoaders for "train", "valid", and "test".
    """
    from transformers import AutoTokenizer

    if use_precompiled_data:
        print("Using PLINDER dataset")
    else:
        print("Using BioLIP dataset")

    model_name = configs.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloaders = {}
    ligand_list = configs.ligands
    ligand2idx = build_ligand2idx(ligand_list)
    data_root = configs.data_root
    use_dynamic_SMILES = configs.use_dynamic_SMILES

    # Prepare train, test, and valid DataLoaders
    if hasattr(configs, 'train_settings'):
        batch_size = configs.train_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        shuffle = configs.train_settings.shuffle
        num_workers = configs.train_settings.num_workers

        train_dataset = LigandDataset(
            ligand_list,
            data_root,
            tokenizer,
            ligand2idx,
            split="train",
            max_length=max_length,
            subset_size=debug_subset_size if debug else None,
            use_precompiled_data=use_precompiled_data,
            use_dynamic_SMILES=use_dynamic_SMILES
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    if hasattr(configs, 'valid_settings'):
        batch_size = configs.valid_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        num_workers = configs.valid_settings.num_workers

        valid_dataset = LigandDataset(
            ligand_list,
            data_root,
            tokenizer,
            ligand2idx,
            split="eval",
            max_length=max_length,
            subset_size=debug_subset_size if debug else None,
            use_precompiled_data=use_precompiled_data,
            use_dynamic_SMILES=use_dynamic_SMILES
        )
        dataloaders["valid"] = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    if hasattr(configs, 'test_settings'):
        batch_size = configs.test_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        num_workers = configs.test_settings.num_workers

        test_dataset = LigandDataset(
            ligand_list,
            data_root,
            tokenizer,
            ligand2idx,
            split="test",
            max_length=max_length,
            subset_size=debug_subset_size if debug else None,
            use_precompiled_data=use_precompiled_data,
            use_dynamic_SMILES=use_dynamic_SMILES
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloaders

if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")
    import yaml
    from box import Box

    # Load configurations from YAML
    config_file_path = "configs/config.yaml"
    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    configs = Box(config_data)
    use_precompiled_data = configs.use_plinder_dataset
    # Prepare DataLoaders
    dataloaders = prepare_dataloaders(configs, use_precompiled_data=use_precompiled_data)

    # Access DataLoaders
    train_loader = dataloaders.get("train", None)
    valid_loader = dataloaders.get("valid", None)
    test_loader = dataloaders.get("test", None)

    print("Finished preparing DataLoaders")

    if train_loader:
        print(f"Number of samples in train_loader: {len(train_loader.dataset)}")
        # print(f"Number of batches in train_loader: {len(train_loader)}")

    if valid_loader:
        print(f"Number of samples in valid_loader: {len(valid_loader.dataset)}")
        # print(f"Number of batches in valid_loader: {len(valid_loader)}")

    if test_loader:
        print(f"Number of samples in test_loader: {len(test_loader.dataset)}")
        # print(f"Number of batches in test_loader: {len(test_loader)}")

    from collections import Counter

    # Testing
    if train_loader:
        sample = next(iter(train_loader))  # Get first batch
        input_ids = sample["input_ids"][0]
        labels = sample["labels"][0]
        ligand_idx = sample["ligand_idx"][0]

        # Decode tokenized input back to amino acid characters
        tokenizer = AutoTokenizer.from_pretrained(configs.model.model_name)
        decoded = tokenizer.convert_ids_to_tokens(input_ids[:20])  # First 20 tokens

        print("\n=== Sample from train_loader ===")
        print("Decoded Tokens (first 20):", " ".join(decoded))
        print("Labels (first 20):", labels[:20].tolist())
        print("Ligand Index:", ligand_idx.item())
        print("================================\n")

        # Count ligand distribution
        ligand_counter = Counter()
        for batch in train_loader:
            ligands = batch["ligand_idx"]
            for l in ligands:
                ligand_counter[int(l)] += 1

        print("=== Ligand Distribution in train_loader ===")
        ligand_list = configs.ligands
        print(ligand_list)
        ligand2idx = build_ligand2idx(ligand_list)
        idx2ligand = idx2ligand(ligand_list)
        for ligand, count in ligand_counter.items():
            ligand_name = idx2ligand[ligand]
            print(f"{ligand_name} (index {ligand}): {count} samples")
        print("==========================================\n")

        train_loader.dataset.print_tokenizer_vocab()