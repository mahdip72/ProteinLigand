from box import Box
from pathlib import Path
import datetime
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(model, configs):
    optimizer_config = configs.optimizer
    optimizer_name = optimizer_config.name.lower()
    weight_decouple = optimizer_config.weight_decouple

    if optimizer_name == 'adam':
        if weight_decouple:
            print("Using AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config.lr,
                betas=(optimizer_config.beta_1, optimizer_config.beta_2),
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            print("Using Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_config.lr,
                betas=(optimizer_config.beta_1, optimizer_config.beta_2),
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay
            )
    elif optimizer_name == 'sgd':
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=optimizer_config.nesterov
        )
    return optimizer


def get_scheduler(optimizer, configs):
    scheduler_config = configs.scheduler
    scheduler_name = scheduler_config.name.lower()

    if scheduler_name == 'cosine_annealing':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.T_max,
            eta_min=scheduler_config.eta_min
        )
    elif scheduler_name == 'cosine_annealing_warm_restarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.T_0,
            T_mult=scheduler_config.T_mult,
            eta_min=scheduler_config.eta_min
        )
    elif scheduler_name == 'cosine_annealing_sequential':
        # First cosine scheduler
        first_cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs.train_settings.num_epochs // 2,
            eta_min=scheduler_config.eta_min_first
        )
        print("First cosine scheduler with eta_min: ", scheduler_config.eta_min_first, " and starting learning rate of: ", optimizer.param_groups[0]["lr"])
        # Second cosine scheduler
        second_cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs.train_settings.num_epochs // 2,
            eta_min=scheduler_config.eta_min_second
        )
        print("Second cosine scheduler with eta_min: ", scheduler_config.eta_min_second, " and starting learning rate of: ", scheduler_config.eta_min_first)
        # SequentialLR combining all three
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[first_cosine_scheduler, second_cosine_scheduler],
            milestones=[configs.train_settings.num_epochs // 2]
        )

    elif scheduler_name == 'multistep_lr':
        milestones = [
            int(configs.train_settings.num_epochs / 3),
            int(2 * configs.train_settings.num_epochs / 3)
        ]

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler

def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer

def prepare_saving_dir(configs, config_file_path, save_to_data=False):
    """
    Prepare a directory for saving training results.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Path to the configuration file.
        save_to_data (bool, optional): If True, save in Hellbender 500GB ~/data instead of the normal path. Defaults to False.

    Returns:
        tuple: (result_path, checkpoint_path) - The paths where results and checkpoints will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Optionally label the run_id with the dataset name
    if configs.data_type:
        run_id += f"__{configs.data_type}"

    # Determine the base save path
    if save_to_data:
        base_path = os.path.expanduser("~/data")  # Expands ~/data to absolute path
    else:
        base_path = os.path.abspath(configs.result_path)

    # Create the result directory and checkpoint subdirectory.
    result_path = os.path.join(base_path, run_id)
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    print("Created saving directory: ", result_path)

    return result_path, checkpoint_path

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, checkpoint_path):
    """
    Save a checkpoint of the model, optimizer, scheduler, and scaler.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch
    }
    checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved at {checkpoint_file}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load a checkpoint to restore the model, optimizer, scheduler, and scaler states.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to restore.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to restore.
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler to restore.

    Returns:
        int: The epoch to resume training from.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state restored.")

    # Restore optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state restored.")

    # Restore scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state restored.")

    # Restore scaler state if provided
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("Scaler state restored.")

    # Return the epoch to resume from
    epoch = checkpoint.get('epoch', -1)
    print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}.")

    return epoch

def load_configs(config):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)

    return tree_config

def visualize_predictions(model, dataloader, device, num_sequences=5):
    """
    Visualize model predictions on the validation dataset.

    Args:
        model: The trained model.
        dataloader: DataLoader for the validation/test dataset.
        device: The device (CPU or GPU) where the model is loaded.
        num_sequences: Number of sequences to visualize.
    """
    model.eval()

    sequences = []
    ground_truths = []
    predictions = []

    print("'P' indicates a positive label/prediction, '.' indicates a negative label/prediction, and '!' indicates a difference between the ground truth and the prediction.")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(sequences) >= num_sequences:
                break
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=inputs, attention_mask=attention_mask)
            preds = torch.sigmoid(logits) > 0.5

            for seq, label, pred, mask in zip(inputs, labels, preds, attention_mask):
                if len(sequences) >= num_sequences:
                    break

                seq = seq.cpu().numpy()
                label = label.cpu().numpy()
                pred = pred.cpu().numpy()
                mask = mask.cpu().numpy()

                valid_idx = mask.astype(bool)
                sequences.append(seq[valid_idx])
                ground_truths.append(label[valid_idx])
                predictions.append(pred[valid_idx])

    for i in range(len(sequences)):
        seq = sequences[i]
        truth = ground_truths[i]
        pred = predictions[i]

        decoded_seq = dataloader.dataset.tokenizer.decode(seq, skip_special_tokens=True)

        pred_str = ' '.join('P' if p else '.' for p in pred)
        truth_str = ' '.join('P' if t else '.' for t in truth)
        true_positives = int(sum((t == 1 and p == 1) for t, p in zip(truth, pred)))
        false_positives = int(sum((t == 0 and p == 1) for t, p in zip(truth, pred)))
        false_negatives = int(sum((t == 1 and p == 0) for t, p in zip(truth, pred)))

        if true_positives > 0:
            f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        else:
            f1_score = 0.0

        print(f"\nSequence {i + 1}:")
        print(f"  Sequence:      {decoded_seq}")
        print(f"  Ground Truth:  {truth_str}")
        print(f"  Predictions:   {pred_str}")
        differences = ' '.join('!' if t != p else ' ' for t, p in zip(truth, pred))
        print(f"  Differences:   {differences}")
        print(
            f"  Length: {len(seq)}, F1 Score: {f1_score:.2f}, "
            f"Correctly Predicted Positives: {true_positives}, "
            f"False Positives: {false_positives}, "
            f"False Negatives: {false_negatives}"
        )
        
import torch

def apply_random_masking(input_ids, mask_token_id, amino_acid_token_ids, mask_prob=0.05):
    """
    Apply ESM-style masked language modeling:
    - Randomly mask `mask_prob`% of tokens in the sequence (excluding padding).
    - Of the selected tokens:
        - 80% replaced with [MASK]
        - 10% replaced with a random amino acid token
        - 10% left unchanged
    Args:
        input_ids (Tensor): [batch_size, seq_len]
        mask_token_id (int): Token ID for [MASK]
        amino_acid_token_ids (List[int]): Valid token IDs for amino acids (excluding special tokens)
        mask_prob (float): Probability of masking a token (default 0.15, like ESM)
    Returns:
        Tensor: masked input_ids
    """
    device = input_ids.device
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_selector = (rand < mask_prob) & (input_ids != 0)

    masked_input = input_ids.clone()
    rand_for_strategy = torch.rand_like(input_ids, dtype=torch.float)

    mask_mask = (rand_for_strategy < 0.8) & mask_selector
    masked_input[mask_mask] = mask_token_id

    rand_mask = ((rand_for_strategy >= 0.8) & (rand_for_strategy < 0.9)) & mask_selector
    random_tokens = torch.randint(len(amino_acid_token_ids), rand_mask.shape, device=device)
    masked_input[rand_mask] = torch.tensor(amino_acid_token_ids, device=device)[random_tokens[rand_mask]]

    # Leave 10% unchanged (no need to modify masked_input)

    return masked_input
