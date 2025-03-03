import torch
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd
from transformers import AutoTokenizer

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
    def __init__(self, seq_file, label_file, tokenizer, max_length=512, subset_size=None):
        """
        Args:
            seq_file (str): Path to the fasta sequence file.
            label_file (str): Path to the fasta label file.
            tokenizer: Tokenizer from Hugging Face (e.g., ESM2 tokenizer).
            max_length (int): Maximum length for tokenized sequences.
        """

        # read and combine both files to create a paired list of the data
        self.sequences = read_fasta(seq_file)
        self.labels = read_fasta(label_file)
        self.data = []
        for seq_id in self.sequences:
            if seq_id in self.labels and len(self.sequences[seq_id]) == len(self.labels[seq_id]):
                self.data.append((seq_id, self.sequences[seq_id], self.labels[seq_id]))

        # If we only want a small subset to test quickly:
        if subset_size is not None:
            import random
            # Randomly sample 'subset_size' rows from dataset
            self.data = random.sample(self.data, subset_size)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_id, sequence, label_str = self.data[idx]
        # Tokenize sequence
        inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=self.max_length, add_special_tokens=True)
        # Convert label string to tensor
        labels = torch.tensor([int(c) for c in label_str], dtype=torch.long)
        # Pad or truncate labels to max_length
        if len(labels) < self.max_length:
            labels = torch.cat([labels, torch.zeros(self.max_length - len(labels), dtype=torch.long)])
        else:
            labels = labels[:self.max_length]

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

def prepare_dataloaders(configs, debug = False, debug_subset_size=None):
    """
    Prepares DataLoaders for training, validation, and testing based on configurations.

    Args:
        configs: Configuration object containing file paths and DataLoader settings.

    Returns:
        dict: A dictionary containing DataLoaders for "train", "valid", and "test".
    """
    from transformers import AutoTokenizer

    model_name = configs.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloaders = {}

    data_type = configs.data_type
    base_path = f"./data/{data_type}"  # Base directory for the dataset

    # Prepare paths for train, test, and valid DataLoaders
    if hasattr(configs, 'train_settings'):
        train_seq_file_path = f"{base_path}/train_set.fasta"
        train_label_file_path = f"{base_path}/train_labels.fasta"

    if hasattr(configs, 'valid_settings'):
        valid_seq_file_path = f"{base_path}/eval_set.fasta"
        valid_label_file_path = f"{base_path}/eval_labels.fasta"

    if hasattr(configs, 'test_settings'):
        test_seq_file_path = f"{base_path}/test_set.fasta"
        test_label_file_path = f"{base_path}/test_labels.fasta"


    # Prepare train, test, and valid DataLoaders
    if hasattr(configs, 'train_settings'):
        # use default names, or config names if no default names
        train_seq_file = train_seq_file_path or configs.train_settings.train_seq_path
        train_label_file = train_label_file_path or configs.train_settings.train_label_path
        batch_size = configs.train_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        shuffle = configs.train_settings.shuffle
        num_workers = configs.train_settings.num_workers

        train_dataset = LigandDataset(
            train_seq_file,
            train_label_file,
            tokenizer,
            max_length=max_length,
            subset_size=debug_subset_size if debug else None
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    if hasattr(configs, 'valid_settings'):
        valid_seq_file = valid_seq_file_path or configs.valid_settings.valid_seq_path
        valid_label_file = valid_label_file_path or configs.valid_settings.valid_label_path
        batch_size = configs.valid_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        num_workers = configs.valid_settings.num_workers

        valid_dataset = LigandDataset(
            valid_seq_file,
            valid_label_file,
            tokenizer,
            max_length=max_length
        )
        dataloaders["valid"] = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    if hasattr(configs, 'test_settings'):
        test_seq_file = test_seq_file_path or configs.test_settings.test_seq_path
        test_label_file = test_label_file_path or configs.test_settings.test_label_path
        batch_size = configs.test_settings.batch_size
        max_length = configs.train_settings.max_sequence_length
        num_workers = configs.test_settings.num_workers

        test_dataset = LigandDataset(
            test_seq_file,
            test_label_file,
            tokenizer,
            max_length=max_length
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloaders


def analyze_data(configs):
    """
    Analyzes ligand-binding frequency of amino acids across all datasets,
    calculating both raw binding counts and binding proportion.

    Args:
        configs: Configuration object containing paths to sequence and label FASTA files.

    Returns:
        dict: Sorted dictionary of amino acids with their binding counts and binding proportion.
    """
    from collections import Counter
    # Define file paths for train, valid, and test datasets

    data_type = configs.data_type
    base_path = f"./data/{data_type}"  # Base directory for the dataset
    train_seq_file_path = f"{base_path}/train_set.fasta"
    train_label_file_path = f"{base_path}/train_labels.fasta"
    valid_seq_file_path = f"{base_path}/eval_set.fasta"
    valid_label_file_path = f"{base_path}/eval_labels.fasta"
    test_seq_file_path = f"{base_path}/test_set.fasta"
    test_label_file_path = f"{base_path}/test_labels.fasta"

    file_paths = [
        (train_seq_file_path, train_label_file_path),
        (valid_seq_file_path, valid_label_file_path),
        (test_seq_file_path, test_label_file_path)
    ]

    # Dictionaries to store counts
    binding_counts = Counter()  # Counts of how often each amino acid binds
    total_counts = Counter()    # Counts of how often each amino acid appears

    # Iterate over all datasets (train, valid, test)
    for seq_file, label_file in file_paths:
        if seq_file and label_file:  # Ensure paths are provided
            # Read FASTA sequences and labels
            sequences = read_fasta(seq_file)
            labels = read_fasta(label_file)

            # Process each sequence
            for seq_id in sequences:
                if seq_id in labels and len(sequences[seq_id]) == len(labels[seq_id]):
                    sequence = sequences[seq_id]
                    label = labels[seq_id]  # "0001000100" (binary string)

                    # Count amino acids in the dataset
                    total_counts.update(sequence)

                    # Count amino acids at ligand-binding positions
                    for i, (aa, bind) in enumerate(zip(sequence, label)):
                        if bind == "1":  # If this position is ligand-binding
                            binding_counts[aa] += 1

    # Compute binding proportions (binding count / total count)
    binding_proportions = {
        aa: binding_counts[aa] / total_counts[aa] if total_counts[aa] > 0 else 0
        for aa in total_counts
    }

    # Calculate total binding proportion across all amino acids
    total_binding_count = sum(binding_counts.values())
    total_amino_acid_count = sum(total_counts.values())
    total_binding_proportion = total_binding_count / total_amino_acid_count if total_amino_acid_count > 0 else 0

    # Sort amino acids by binding proportion (highest proportion first)
    sorted_binding = sorted(binding_proportions.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print("\nAmino Acid Binding Proportions (Sorted by Binding Frequency Proportion):")
    print(f"{'Amino Acid':<15}{'Binding Count':<15}{'Total Count':<15}{'Binding Proportion'}")
    print("-" * 50)
    for aa, proportion in sorted_binding:
        print(f"{aa:<15}{binding_counts[aa]:<15}{total_counts[aa]:<15}{proportion:.4f}")

    print("\nTotal Binding Proportion (Overall Across All Amino Acids):")
    print(
        f"{total_binding_count} binding sites out of {total_amino_acid_count} total residues ({total_binding_proportion:.4f})")

    return sorted_binding

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

    # Prepare DataLoaders
    dataloaders = prepare_dataloaders(configs)

    # Access DataLoaders
    train_loader = dataloaders.get("train", None)
    valid_loader = dataloaders.get("valid", None)
    test_loader = dataloaders.get("test", None)

    print("Finished preparing DataLoaders")

    if train_loader:
        print(f"Number of samples in train_loader: {len(train_loader.dataset)}")
        print(f"Number of batches in train_loader: {len(train_loader)}")

    if valid_loader:
        print(f"Number of samples in valid_loader: {len(valid_loader.dataset)}")
        print(f"Number of batches in valid_loader: {len(valid_loader)}")

    if test_loader:
        print(f"Number of samples in test_loader: {len(test_loader.dataset)}")
        print(f"Number of batches in test_loader: {len(test_loader)}")

    # analyze_data(configs)