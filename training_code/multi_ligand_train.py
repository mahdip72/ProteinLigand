import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer, get_scheduler, save_checkpoint, \
    load_checkpoint, visualize_predictions, apply_random_masking
from multi_ligand_dataset import prepare_dataloaders
from multi_ligand_model import prepare_model
import tqdm
import torchmetrics
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score as sk_f1
import copy

def calculate_loss(logits, labels, ligand_idx = None, smoothed_pos_weight=None, device='cuda', alpha=0.9, use_focal_loss=False, gamma=2.0,
                   label_smoothing=0.0, **kwargs):
    """
    Calculates the loss using either weighted BCE or focal loss.

    Args:
        logits (Tensor): Logits output from the model.
        labels (Tensor): Ground truth labels.
        ligand_idx (Tensor): Index of ligand to use.
        smoothed_pos_weight (float, optional): Smoothed positive class weight. Defaults to None.
        device (str): Device to use (e.g., 'cuda' or 'cpu').
        alpha (float): Weight smoothing parameter for dynamic class balancing. Defaults to 0.9.
        use_focal_loss (bool): If True, use focal loss instead of weighted BCE loss. Defaults to False.
        gamma (float): Focusing parameter for focal loss. Defaults to 2.0.
        label_smoothing (float): Amount of label smoothing to apply (0.0 = none).

    Returns:
        Tensor: The calculated loss.
        float: Updated smoothed positive weight.
    """
    configs = kwargs.get('configs', None)
    # ========================
    # 1. Dynamic Positive Weight
    # ========================

    # positive_count = labels.sum()
    # negative_count = labels.numel() - positive_count
    #
    # # Calculate current positive weight
    # if positive_count > 0 and negative_count > 0:
    #     current_pos_weight = negative_count / positive_count
    # else:
    #     current_pos_weight = 1.0
    #
    # # Smooth the positive weight
    # if smoothed_pos_weight is None:
    #     smoothed_pos_weight = current_pos_weight
    # else:
    #     smoothed_pos_weight = smoothed_pos_weight
    #     weight_divisor = configs.pos_weight_divisor if configs and hasattr(configs, 'pos_weight_divisor') else 1
    #     max_weight = configs.max_pos_weight if configs and hasattr(configs, 'max_pos_weight') else 1
    #
    #     smoothed_weight = alpha * smoothed_pos_weight + (1 - alpha) * current_pos_weight
    #     smoothed_pos_weight = min(smoothed_weight, 1)
    #     # smoothed_pos_weight = smoothed_weight
    #     # Experimental: Cap the smoothed positive weight at a very small value
    #     # smoothed_pos_weight = max(smoothed_weight/weight_divisor, 1)
    #     # else:
    #     #     smoothed_pos_weight = min(smoothed_weight, max_weight)
    #
    # pos_weight_tensor = torch.as_tensor(smoothed_pos_weight, dtype=torch.float, device=device)
    # pos_weight_tensor = pos_weight_tensor.clone().detach()
    #
    # # ========================
    # # 2. Apply Label Smoothing
    # # ========================
    # # label_smoothing in [0.0, 1.0], e.g. 0.1 => "0.9 for positives, 0.1 for negatives"
    # if label_smoothing > 0.0:
    #     eps = label_smoothing
    #     # smoothed_labels = labels.float() * (1.0 - eps) + 0.5 * eps
    #
    #     # Experimental smooth less for negative class since the positive weight can be very large
    #     smoothed_labels = labels.float() * (1.0 - eps) + 0.01 * eps
    #
    #     # Experimental: only smooth for the positive class to prevent overconfident false positives
    #     # smoothed_labels = labels.float() * (1.0 - eps)
    #
    # else:
    #     smoothed_labels = labels.float()

    smoothed_labels = labels.float()
    # commenting out dynamic positive weight for now. Reminder to uncomment line 98 and above code later

    # ========================
    # 3. Compute Weighted BCE
    # ========================
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.view(-1),
        smoothed_labels.view(-1),
        # pos_weight=pos_weight_tensor,
        reduction='none'
    )
    bce_loss = bce_loss.view_as(logits)

    # ========================
    # 4. Dataset Weighting
    # ========================
    if configs and hasattr(configs, 'ligands') and hasattr(configs, 'dataset_weight') and ligand_idx is not None:
        ligand_names = configs.ligands
        dataset_weight = configs.dataset_weight

        # Get weights for each sample in the batch
        batch_ligand_weights = torch.tensor([
            dataset_weight.get(ligand_names[idx.item()], 1.0)
            for idx in ligand_idx
        ], device=logits.device)

        weight_matrix = batch_ligand_weights.unsqueeze(1).unsqueeze(2).expand_as(logits)
    else:
        weight_matrix = torch.ones_like(logits)

    # ========================
    # 5. Optional Focal Loss
    # ========================
    if use_focal_loss:
        pt = torch.exp(-bce_loss)
        focal_term = ((1 - pt) ** gamma)
        loss_matrix = focal_term * bce_loss * weight_matrix
    else:
        loss_matrix = bce_loss * weight_matrix

    loss = loss_matrix.mean()
    return loss, smoothed_pos_weight


def training_loop(model, trainloader, optimizer, epoch, device, scaler, scheduler, train_writer=None, grad_clip_norm=1,
                  alpha=0.9, gamma=2.0, label_smoothing=0.0, verbose=False, **kwargs):
    """
    Training loop for fine-tuning the model on the Ligand dataset.

    Args:
        model: The model to train.
        trainloader: DataLoader for the training dataset.
        optimizer: Optimizer for model parameters.
        epoch: Current epoch number.
        device: Device to run training on (CPU/GPU).
        scaler: Gradient scaler for mixed precision training.
        scheduler: Learning rate scheduler.
        train_writer: TensorBoard writer for logging.
        grad_clip_norm: Gradient clipping value.
        verbose: (bool): If True, logs individual and macro ligand F1 scores in TensorBoard. // (Makes training around twice as slow)
        kwargs: Additional parameters.
    """
    accuracy = torchmetrics.Accuracy(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")

    accuracy.to(device)
    f1_score.to(device)
    model.train()
    running_loss = 0.0
    smoothed_pos_weight = None

    # for keeping track of individual ligand predictions
    if verbose:
        ligand_preds_labels = defaultdict(list)

    # For logging within each epoch instead of just once every epoch since each epoch takes a long time, logging 100 times per epoch
    log_interval = max(len(trainloader) // 100, 1)
    mixed_precision = kwargs.get("configs", {}).get("train_settings", {}).get("mixed_precision", "no")

    # For random masking
    amino_acid_token_ids = trainloader.dataset.get_amino_acid_token_ids()

    for i, batch in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}", leave=False):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ligand_idx = batch["ligand_idx"].to(device)
        
        configs = kwargs.get("configs", None) 
        mask_prob = configs.mask_prob if configs and hasattr(configs, "mask_prob") else 0

        num_epochs = configs.train_settings.num_epochs if configs and hasattr(configs, "num_epochs") else 1
        # Experimenting with decaying masking
        mask_prob = mask_prob * (1 - epoch / num_epochs)
        # Experimenting with curriculum-style learning with increasing masking
        # mask_prob = mask_prob * (epoch / num_epochs)

        inputs = apply_random_masking(
            inputs,
            mask_token_id=configs.mask_token_id,
            amino_acid_token_ids=amino_acid_token_ids,
            mask_prob=mask_prob
        )
        optimizer.zero_grad()

        # TESTING
        if epoch == 0 and i == 0:
            print("Original:", batch["input_ids"][0][:30])
            print("Masked:", inputs[0][:30])

        # Mixed precision training
        if mixed_precision == "fp16":
            autocast_dtype = torch.float16
        elif mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None
        with autocast(device_type=device.type, dtype=autocast_dtype):
            outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx)
            logits = outputs

            loss, smoothed_pos_weight = calculate_loss(
                logits=logits,
                labels=labels,
                ligand_idx=ligand_idx,
                smoothed_pos_weight=smoothed_pos_weight,
                device=device,
                alpha=alpha,
                use_focal_loss=False,
                gamma=gamma,
                label_smoothing=label_smoothing,
                **kwargs
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long()
        predictions = predictions.squeeze(-1)
        batch_size, seq_len = predictions.shape
        accuracy.update(predictions.view(-1), labels.view(-1))
        f1_score.update(predictions.view(-1), labels.view(-1))

        if verbose:
            for b in range(batch_size):
                for t in range(seq_len):
                    if attention_mask[b][t]:
                        lig = int(ligand_idx[b].item())
                        ligand_preds_labels[lig].append((
                            predictions[b][t].item(),
                            labels[b][t].item()
                        ))

        if train_writer and (i + 1) % log_interval == 0:
            train_writer.add_scalar("Gradient_Norm", grad_norm, epoch * len(trainloader) + i)
            train_writer.add_scalar("Batch_Loss", loss.item(), epoch * len(trainloader) + i)
            train_writer.add_scalar("Running_Accuracy", accuracy.compute().cpu().item(), epoch * len(trainloader) + i)
            train_writer.add_scalar("Running_F1_Score", f1_score.compute().cpu().item(), epoch * len(trainloader) + i)

    avg_train_loss = running_loss / len(trainloader)
    epoch_acc = accuracy.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu().item()

    accuracy.reset()
    f1_score.reset()

    if verbose:
        ligand_f1s = {}
        all_f1s = []
        for lig_id, pairs in ligand_preds_labels.items():
            if pairs:
                y_pred, y_true = zip(*pairs)
                f1 = sk_f1(y_true, y_pred, zero_division=0)
                ligand_f1s[lig_id] = f1
                all_f1s.append(f1)

        macro_f1 = np.mean(all_f1s) if all_f1s else 0.0

    if train_writer:
        train_writer.add_scalar("Loss", avg_train_loss, epoch)
        train_writer.add_scalar("Accuracy", epoch_acc, epoch)
        train_writer.add_scalar("F1_Score", epoch_f1, epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_writer.add_scalar("Learning_Rate", lr, epoch)


    if verbose and train_writer:
        configs = kwargs.get("configs", None)
        ligand_names = configs.ligands if configs and hasattr(configs, "ligands") else []
        ligand2name = {i: name for i, name in enumerate(ligand_names)}
        for lig_id, f1 in ligand_f1s.items():
            lig_name = ligand2name.get(lig_id, f"Ligand_{lig_id}")
            train_writer.add_scalar("Macro_F1", macro_f1, epoch)
            train_writer.add_scalar(f"Ligand_F1/{lig_name}", f1, epoch)

    if verbose:
        print(f"Training Accuracy: {100 * epoch_acc:.2f}%, All-Sites F1: {epoch_f1:.4f}, Macro F1: {macro_f1:.4f}")
    else:
        print(f"Training Accuracy: {100 * epoch_acc:.2f}%, All-Sites F1: {epoch_f1:.4f}")

    return avg_train_loss, epoch_f1


def validation_loop(model, testloader, epoch, device, valid_writer=None, alpha=0.9, gamma=2.0, label_smoothing=0.0, verbose=True, **kwargs):
    """
    Validation loop to evaluate the model on the test/validation dataset.

    Args:
        model: The model to evaluate.
        testloader: DataLoader for the test/validation dataset.
        epoch: Current epoch number.
        device: Device to run validation on (CPU/GPU).
        valid_writer: TensorBoard writer for logging.
        verbose: Prints and logs individual ligand F1 scores if True.
        kwargs: Additional parameters.
    """
    accuracy = torchmetrics.Accuracy(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")

    accuracy.to(device)
    f1_score.to(device)

    model.eval()
    valid_loss = 0.0
    smoothed_pos_weight = None

    ligand_preds_labels = defaultdict(list)

    for i, batch in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc=f"Validation Epoch {epoch + 1}",
                              leave=False):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ligand_idx = batch["ligand_idx"].to(device)

        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx)
                logits = outputs

                loss, smoothed_pos_weight = calculate_loss(
                    logits=logits,
                    labels=labels,
                    ligand_idx=ligand_idx,
                    smoothed_pos_weight=smoothed_pos_weight,
                    device=device,
                    alpha=alpha,
                    use_focal_loss=False,
                    gamma=gamma,
                    label_smoothing=label_smoothing,
                    **kwargs
                )
                valid_loss += loss.item()

            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long()
            predictions = predictions.squeeze(-1)
            batch_size, seq_len = predictions.shape
            accuracy.update(predictions.view(-1), labels.view(-1))
            f1_score.update(predictions.view(-1), labels.view(-1))

            for b in range(batch_size):
                for t in range(seq_len):
                    if attention_mask[b][t]:
                        lig = int(ligand_idx[b].item())
                        ligand_preds_labels[lig].append((
                            predictions[b][t].item(),
                            labels[b][t].item()
                        ))

    avg_valid_loss = valid_loss / len(testloader)
    valid_acc = accuracy.compute().cpu().item()
    valid_f1 = f1_score.compute().cpu().item()

    accuracy.reset()
    f1_score.reset()

    ligand_f1s = {}
    all_f1s = []
    for lig_id, pairs in ligand_preds_labels.items():
        if pairs:
            y_pred, y_true = zip(*pairs)
            f1 = sk_f1(y_true, y_pred, zero_division=0)
            ligand_f1s[lig_id] = f1
            all_f1s.append(f1)

    macro_f1 = np.mean(all_f1s) if all_f1s else 0.0

    if valid_writer:
        valid_writer.add_scalar("Loss", avg_valid_loss, epoch)
        valid_writer.add_scalar("Accuracy", valid_acc, epoch)
        valid_writer.add_scalar("F1_Score", valid_f1, epoch)
        valid_writer.add_scalar("Macro_F1", macro_f1, epoch)

    print(f"Epoch: {epoch}, Validation Accuracy: {100 * valid_acc:.2f}%, All-Sites F1 Score: {valid_f1:.4f}, Macro F1: {macro_f1:.4f}")

    if verbose:
        configs = kwargs.get("configs", None)
        ligand_names = configs.ligands if configs and hasattr(configs, "ligands") else []
        ligand2name = {i: name for i, name in enumerate(ligand_names)}
        for lig_id, f1 in ligand_f1s.items():
            lig_name = ligand2name.get(lig_id, f"Ligand_{lig_id}")
            print(f"  {lig_name}: F1 = {f1:.4f}")
            if valid_writer:
                valid_writer.add_scalar(f"Ligand_F1/{lig_name}", f1, epoch)

    return avg_valid_loss, valid_f1, macro_f1


def evaluation_loop(model, testloader, device, log_confidences=False, alpha=0.9, gamma=2.0, label_smoothing=0.0, **kwargs):
    """
    Test loop to evaluate the model on the test dataset with detailed analytics.

    Args:
        model: The trained model to evaluate.
        testloader: DataLoader for the test dataset.
        device: Device to run evaluation on (CPU/GPU).
        log_confidences (bool): Whether to log confidence scores of incorrect predictions.
        kwargs: Additional parameters.
    """
    from sklearn.metrics import confusion_matrix

    accuracy = torchmetrics.Accuracy(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")

    accuracy.to(device)
    f1_score.to(device)
    precision.to(device)
    recall.to(device)

    model.eval()
    test_loss = 0.0

    all_labels = []
    all_predictions = []
    incorrect_confidences = [] if log_confidences else None
    false_positive_confidences = [] if log_confidences else None
    false_negative_confidences = [] if log_confidences else None
    smoothed_pos_weight = None

    from collections import defaultdict
    ligand_preds_labels = defaultdict(list)

    for i, batch in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc="Testing Model"):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ligand_idx = batch["ligand_idx"].to(device)

        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx)
                logits = outputs

                loss, smoothed_pos_weight = calculate_loss(
                    logits=logits,
                    labels=labels,
                    ligand_idx=ligand_idx,
                    smoothed_pos_weight=smoothed_pos_weight,
                    device=device,
                    alpha=alpha,
                    use_focal_loss=False,
                    gamma=gamma,
                    label_smoothing=label_smoothing,
                    **kwargs
                )
                test_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)  # Convert to binary labels
            labels_flat = labels.cpu().numpy().flatten()

            batch_size, seq_len = logits.shape[:2]
            for b in range(batch_size):
                ligand = int(ligand_idx[b].item())
                for t in range(seq_len):
                    if attention_mask[b][t]:
                        pred = predictions[b * seq_len + t]
                        true = labels_flat[b * seq_len + t]
                        ligand_preds_labels[ligand].append((pred, true))

            # Store incorrect prediction confidence scores only if log_confidences is True
            if log_confidences:
                for prob, pred, true_label in zip(probabilities, predictions, labels_flat):
                    if pred != true_label:  # Misclassified samples
                        incorrect_confidences.append(prob)
                        if pred == 1 and true_label == 0:  # False Positive
                            false_positive_confidences.append(prob)
                        elif pred == 0 and true_label == 1:  # False Negative
                            false_negative_confidences.append(prob)

            # Update Metrics
            accuracy.update(torch.tensor(predictions), torch.tensor(labels.cpu().numpy().flatten()))
            f1_score.update(torch.tensor(predictions), torch.tensor(labels.cpu().numpy().flatten()))
            precision.update(torch.tensor(predictions), torch.tensor(labels.cpu().numpy().flatten()))
            recall.update(torch.tensor(predictions), torch.tensor(labels.cpu().numpy().flatten()))

            # Collect predictions and labels for further analysis
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy().flatten())

    # Compute Metrics
    avg_test_loss = test_loss / len(testloader)
    test_acc = accuracy.compute().cpu().item()
    test_f1 = f1_score.compute().cpu().item()
    test_precision = precision.compute().cpu().item()
    test_recall = recall.compute().cpu().item()

    accuracy.reset()
    f1_score.reset()
    precision.reset()
    recall.reset()

    # Confusion Matrix
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

    # Log Detailed Analytics
    print("\n=== Test Results ===")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {100 * test_acc:.2f}%")
    print(f"All-Sites F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"True Positives: {tp}, False Positives: {fp}")
    print(f"True Negatives: {tn}, False Negatives: {fn}")
    print(f"Fraction of Positive Predictions: {np.sum(all_predictions) / len(all_predictions):.4f}")

    # Log Confidence Scores of Incorrect Predictions if enabled
    if log_confidences and incorrect_confidences:
        avg_confidence = np.mean(incorrect_confidences)
        avg_fp_confidence = np.mean(false_positive_confidences) if false_positive_confidences else 0
        avg_fn_confidence = np.mean(false_negative_confidences) if false_negative_confidences else 0

        print(f"Average Confidence of Incorrect Predictions: {avg_confidence:.4f}")
        print(f"Average Confidence of False Positives: {avg_fp_confidence:.4f}")
        print(f"Average Confidence of False Negatives: {avg_fn_confidence:.4f}")
        print(f"Top 10 Incorrect Confidence Scores: {sorted(incorrect_confidences, reverse=True)[:10]}")

    print("\n=== F1 Score Per Ligand ===")
    configs = kwargs.get("configs", None)
    if configs and hasattr(configs, "ligands"):
        ligand_names = configs.ligands
    else:
        ligand_names = []
    ligand2name = {i: name for i, name in enumerate(ligand_names)}

    ligand_f1s = []
    for ligand_idx, pairs in ligand_preds_labels.items():
        preds, labels = zip(*pairs)
        f1 = sk_f1(labels, preds, zero_division=0)
        ligand_f1s.append(f1)
        name = ligand2name.get(ligand_idx, str(ligand_idx))
        print(f"{name}: F1 = {f1:.4f} ({len(pairs)} classification points)")

    if ligand_f1s:
        macro_f1 = np.mean(ligand_f1s)
        print(f"\nMacro F1 Score (Average across {len(ligand_f1s)} ligands): {macro_f1:.4f}")
    else:
        print("\nMacro F1 Score: N/A (no ligand data)")

    return {
        "loss": avg_test_loss,
        "accuracy": test_acc,
        "f1_score": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "incorrect_confidence_scores": incorrect_confidences if log_confidences else None
    }


def main(dict_config, config_file_path):

    # Flags for convenience
    train = True
    on_hellbender = True
    save_best_checkpoint_only = True
    use_checkpoint = False
    visualize = False
    test = True

    if use_checkpoint:
        load_checkpoint_path = "/home/dc57y/data/2025-04-06__23-37-46__TOP-10/checkpoints/checkpoint_epoch_6.pth"
    else:
        load_checkpoint_path = None

    configs = load_configs(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    dataloaders = prepare_dataloaders(configs)
    # dataloaders = prepare_dataloaders(configs, debug=True, debug_subset_size=50)

    trainloader = dataloaders["train"]
    validloader = dataloaders["valid"]
    testloader = dataloaders["test"]

    print("Finished preparing dataloaders")

    _, model = prepare_model(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Finished preparing model, using device:", device)

    optimizer = get_optimizer(model, configs)
    scheduler = get_scheduler(optimizer, configs)
    num_epochs = configs.train_settings.num_epochs
    grad_clip_norm = configs.train_settings.grad_clip_norm
    alpha = configs.train_settings.alpha
    gamma = configs.train_settings.gamma
    label_smoothing = configs.train_settings.label_smoothing
    checkpoint_every = configs.checkpoints_every
    scaler = GradScaler()
    start_epoch = 0

    if not load_checkpoint_path:
        print("Training without using any checkpoints")

    if load_checkpoint_path:
        print(f"Loading checkpoint from {load_checkpoint_path}...")
        # start_epoch = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, scaler) + 1
        # start_epoch = load_checkpoint(load_checkpoint_path, model=model, optimizer=None, scheduler=scheduler, scaler=scaler) + 1
        start_epoch = load_checkpoint(load_checkpoint_path, model) + 1
        start_epoch = 0

        if visualize:
            print("Visualizing predictions on test dataset")
            visualize_predictions(model, testloader, device, num_sequences=10)
            return

    if save_best_checkpoint_only:
        best_f1 = -1.0
        best_model_state = None
        best_epoch = -1

    if train:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path, save_to_data=on_hellbender)
        train_writer, valid_writer = prepare_tensorboard(result_path)

        for epoch in range(start_epoch, num_epochs):
            training_loss, training_f1 = training_loop(model, trainloader, optimizer, epoch, device, scaler, scheduler,
                                                       train_writer=train_writer, grad_clip_norm=grad_clip_norm,
                                                       alpha=alpha,
                                                       gamma=gamma, label_smoothing=label_smoothing,
                                                       configs=configs)

            valid_loss, _, macro_f1 = validation_loop(model, validloader, epoch, device, valid_writer=valid_writer,
                                                   alpha=alpha, gamma=gamma, configs=configs,
                                                   label_smoothing=label_smoothing,)
            scheduler.step()
            if save_best_checkpoint_only:
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_model_state = {
                        'model': copy.deepcopy(model.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict()),
                        'scheduler': copy.deepcopy(scheduler.state_dict()),
                        'scaler': copy.deepcopy(scaler.state_dict()),
                        'epoch': epoch
                    }
            else:
                if (epoch + 1) % checkpoint_every == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, checkpoint_path)
        if save_best_checkpoint_only and best_f1 > -1:
            model.load_state_dict(best_model_state['model'])
            optimizer.load_state_dict(best_model_state['optimizer'])
            scheduler.load_state_dict(best_model_state['scheduler'])
            scaler.load_state_dict(best_model_state['scaler'])
            save_checkpoint(model, optimizer, scheduler, scaler, best_model_state['epoch'], checkpoint_path)
        print(f"Best Validation Macro F1 Score: {best_f1:.4f}")

    if test:
        print("Testing model on test dataset")
        if use_checkpoint:
            load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, scaler)
            results = evaluation_loop(model, testloader, device, log_confidences=False, alpha=alpha, gamma=gamma,
                                      label_smoothing=label_smoothing, configs=configs)
        else:
            print("Evaluating test dataset with the model at epoch ", best_model_state['epoch'])
            # model.load_state_dict(best_model_state['model'])
            results = evaluation_loop(model, testloader, device, log_confidences=True, alpha=alpha,
                                  gamma=gamma, label_smoothing=label_smoothing, configs=configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune the Ligand model')
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
