import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir, prepare_tensorboard, get_optimizer, get_scheduler, save_checkpoint, \
    load_checkpoint, visualize_predictions, apply_random_masking
from multi_ligand_dataset import prepare_dataloaders, build_ligand2idx
from multi_ligand_model import prepare_model
import tqdm
import torchmetrics
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score as sk_f1
import copy

def calculate_loss(logits, labels, ligands = None, smoothed_pos_weight=None, device='cuda', alpha=0.9, use_focal_loss=False, gamma=2.0,
                   label_smoothing=0.0, **kwargs):
    """
    Calculates the loss using either weighted BCE or focal loss.

    Args:
        logits (Tensor): Logits output from the model.
        labels (Tensor): Ground truth labels.
        ligands (Tensor): Names of the ligands in the batch.
        smoothed_pos_weight (float, optional): Smoothed positive class weight. Defaults to None.
        device (str): Device to use (e.g., 'cuda' or 'cpu').
        alpha (float): Weight smoothing parameter for dynamic class balancing. Defaults to 0.9.
        use_focal_loss (bool): If True, use focal loss instead of weighted BCE loss. Defaults to False.
        gamma (float): Focusing parameter for focal loss. Defaults to 2.0.
        label_smoothing (float): Amount of label smoothing to apply (0.0 = none).
        kwargs: Additional parameters, including configs.

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
    if label_smoothing > 0.0:
        eps = label_smoothing
        smoothed_labels = labels.float() * (1.0 - eps) + 0.5 * eps

        # Experimental smooth less for negative class since the positive weight can be very large
        # smoothed_labels = labels.float() * (1.0 - eps) + 0.01 * eps

        # Experimental: only smooth for the positive class to prevent overconfident false positives
        # smoothed_labels = labels.float() * (1.0 - eps)

    else:
        smoothed_labels = labels.float()

    # smoothed_labels = labels.float()
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
    # 4. Optional Dataset Weighting and Binding-Ratio Weighting
    # ========================

    use_pos_weighting = configs.use_positive_weighting_for_binding_sites
    use_dataset_weighting = configs.use_dataset_weighting
    if configs and hasattr(configs, 'ligands') and hasattr(configs, 'dataset_weight') and ligands is not None:
        ligand_names = configs.ligands
        if use_dataset_weighting and hasattr(configs, 'dataset_weight'):
            dataset_weight = configs.dataset_weight_plinder if configs.dataset_format == "PLINDER" else configs.dataset_weight
        else:
            dataset_weight = {}
        pos_weight_dict = configs.pos_dataset_weight_plinder if configs.dataset_format == "PLINDER" else configs.pos_dataset_weight

        batch_sample_weights = torch.tensor(
            [dataset_weight.get(name, 1.0) for name in ligands],
            device=logits.device
        )
        if use_pos_weighting and pos_weight_dict:
            batch_token_pos_weights = torch.tensor(
                [pos_weight_dict.get(name, 1.0) for name in ligands],
                device=logits.device
            )
            token_mask = labels.float()  # shape: [B, L]
            pos_token_weights = batch_token_pos_weights[:, None] * token_mask + (1.0 - token_mask) # TODO: Behaving weirdly - needs to be debugged
            pos_token_weights = pos_token_weights.unsqueeze(2)  # shape: [B, L, 1]
        else:
            pos_token_weights = torch.ones_like(logits, dtype=torch.float)

        weight_matrix = batch_sample_weights.unsqueeze(1).unsqueeze(2).expand_as(logits)  # [B, 1, 1] → [B, L, 1]
        weight_matrix = weight_matrix * pos_token_weights  # final shape: [B, L, 1]
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
        verbose: (bool): If True, logs individual and macro ligand F1 scores in TensorBoard. // (Makes training around twice as slow, use for debugging)
        kwargs: Additional parameters.
    """
    accuracy = torchmetrics.Accuracy(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    specificity = torchmetrics.Specificity(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")
    auroc = torchmetrics.AUROC(task="binary")

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.to(device)

    model.train()
    running_loss = 0.0
    smoothed_pos_weight = None

    configs = kwargs.get("configs", None)

    # for keeping track of individual ligand predictions
    if verbose:
        ligand_preds_labels = defaultdict(list)

    # For logging within each epoch instead of just once every epoch since each epoch takes a long time, logging 100 times per epoch
    log_interval = max(len(trainloader) // 100, 1)
    mixed_precision = configs.train_settings.mixed_precision if configs and hasattr(configs, "train_settings") else None

    # For random masking
    amino_acid_token_ids = trainloader.dataset.get_amino_acid_token_ids()

    for i, batch in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}", leave=False):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ligand_idx = batch["ligand_idx"].to(device)
        ligand_names = batch["ligand"]
        # If using dynamic ligand SMILES, get the ligand SMILES from the batch
        ligand_smiles = batch.get("smiles", None)


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

        if mixed_precision == "fp16":
            autocast_dtype = torch.float16
        elif mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None
        with autocast(device_type=device.type, dtype=autocast_dtype):
            outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx,
                            ligand_smiles=ligand_smiles)
            logits = outputs
            loss, smoothed_pos_weight = calculate_loss(
                logits=logits,
                labels=labels,
                ligands=ligand_names,
                smoothed_pos_weight=smoothed_pos_weight,
                device=device,
                alpha=alpha,
                use_focal_loss=True,
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

        flat_preds = predictions.view(-1)
        flat_labels = labels.view(-1)
        accuracy.update(flat_preds, flat_labels)
        f1_score.update(flat_preds, flat_labels)
        precision.update(flat_preds, flat_labels)
        recall.update(flat_preds, flat_labels)
        specificity.update(flat_preds, flat_labels)
        mcc.update(flat_preds, flat_labels)
        auroc.update(flat_preds.float(), flat_labels)

        if verbose:
            for b in range(batch_size):
                for t in range(seq_len):
                    if attention_mask[b][t]:
                        ligand_name = batch["ligand"][b]
                        ligand_preds_labels[ligand_name].append((
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
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_specificity = specificity.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu().item()
    epoch_auroc = auroc.compute().cpu().item()

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.reset()

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
        train_writer.add_scalar("Precision", epoch_precision, epoch)
        train_writer.add_scalar("Recall", epoch_recall, epoch)
        train_writer.add_scalar("Specificity", epoch_specificity, epoch)
        train_writer.add_scalar("MCC", epoch_mcc, epoch)
        train_writer.add_scalar("AUROC", epoch_auroc, epoch)


    if verbose and train_writer:
        train_writer.add_scalar("Macro_F1", macro_f1, epoch)
        for ligand_name, f1 in ligand_f1s.items():
            train_writer.add_scalar(f"Ligand_F1/{ligand_name}", f1, epoch)

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
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    specificity = torchmetrics.Specificity(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")
    auroc = torchmetrics.AUROC(task="binary")

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.to(device)

    model.eval()
    valid_loss = 0.0
    smoothed_pos_weight = None

    configs = kwargs.get("configs", None)

    if verbose:
        ligand_preds_labels = defaultdict(list)

    mixed_precision = configs.train_settings.mixed_precision if configs and hasattr(configs, "train_settings") else None

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc=f"Validation Epoch {epoch + 1}",
                                  leave=False):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            ligand_idx = batch["ligand_idx"].to(device)
            ligand_names = batch["ligand"]
            ligand_smiles = batch.get("smiles", None)

            if mixed_precision == "fp16":
                autocast_dtype = torch.float16
            elif mixed_precision == "bf16":
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = None
            with autocast(device_type=device.type, dtype=autocast_dtype):
                outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx,
                                ligand_smiles=ligand_smiles)
                logits = outputs

                loss, smoothed_pos_weight = calculate_loss(
                    logits=logits,
                    labels=labels,
                    ligands=ligand_names,
                    smoothed_pos_weight=smoothed_pos_weight,
                    device=device,
                    alpha=alpha,
                    use_focal_loss=True,
                    gamma=gamma,
                    label_smoothing=label_smoothing,
                    **kwargs
                )
                valid_loss += loss.item()

            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long()
            predictions = predictions.squeeze(-1)
            batch_size, seq_len = predictions.shape

            flat_preds = predictions.view(-1)
            flat_labels = labels.view(-1)
            accuracy.update(flat_preds, flat_labels)
            f1_score.update(flat_preds, flat_labels)
            precision.update(flat_preds, flat_labels)
            recall.update(flat_preds, flat_labels)
            specificity.update(flat_preds, flat_labels)
            mcc.update(flat_preds, flat_labels)
            auroc.update(flat_preds.float(), flat_labels)

            if verbose:
                for b in range(batch_size):
                    for t in range(seq_len):
                        if attention_mask[b][t]:
                            ligand_name = batch["ligand"][b]
                            ligand_preds_labels[ligand_name].append((
                                predictions[b][t].item(),
                                labels[b][t].item()
                            ))

    avg_valid_loss = valid_loss / len(testloader)
    valid_acc = accuracy.compute().cpu().item()
    valid_f1 = f1_score.compute().cpu().item()
    valid_precision = precision.compute().cpu().item()
    valid_recall = recall.compute().cpu().item()
    valid_specificity = specificity.compute().cpu().item()
    valid_mcc = mcc.compute().cpu().item()
    valid_auroc = auroc.compute().cpu().item()

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.reset()

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

    if valid_writer:
        valid_writer.add_scalar("Loss", avg_valid_loss, epoch)
        valid_writer.add_scalar("Accuracy", valid_acc, epoch)
        valid_writer.add_scalar("F1_Score", valid_f1, epoch)
        valid_writer.add_scalar("Macro_F1", macro_f1, epoch)
        valid_writer.add_scalar("Precision", valid_precision, epoch)
        valid_writer.add_scalar("Recall", valid_recall, epoch)
        valid_writer.add_scalar("Specificity", valid_specificity, epoch)
        valid_writer.add_scalar("MCC", valid_mcc, epoch)
        valid_writer.add_scalar("AUROC", valid_auroc, epoch)

    print(f"Epoch: {epoch}, Validation Accuracy: {100 * valid_acc:.2f}%, All-Sites F1 Score: {valid_f1:.4f}, Macro F1: {macro_f1:.4f}")

    if verbose:
        for lig_name, f1 in ligand_f1s.items():
            # print(f"  {lig_name}: F1 = {f1:.4f}") // # TODO: Commenting out for now
            if valid_writer:
                valid_writer.add_scalar(f"Ligand_F1/{lig_name}", f1, epoch)

    return avg_valid_loss, valid_f1, macro_f1, valid_precision, valid_recall, valid_specificity, valid_mcc, valid_auroc


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
    specificity = torchmetrics.Specificity(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")
    auroc = torchmetrics.AUROC(task="binary")

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.to(device)

    model.eval()
    test_loss = 0.0

    configs = kwargs.get("configs", None)

    all_labels = []
    all_predictions = []
    incorrect_confidences = [] if log_confidences else None
    false_positive_confidences = [] if log_confidences else None
    false_negative_confidences = [] if log_confidences else None
    smoothed_pos_weight = None

    mixed_precision = configs.train_settings.mixed_precision if configs and hasattr(configs, "train_settings") else None

    ligand_preds_labels = defaultdict(list)

    with torch.inference_mode():
        for i, batch in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc="Testing Model"):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            ligand_idx = batch["ligand_idx"].to(device)
            ligand_names = batch["ligand"]
            ligand_smiles = batch.get("smiles", None)

            if mixed_precision == "fp16":
                autocast_dtype = torch.float16
            elif mixed_precision == "bf16":
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = None
            with autocast(device_type=device.type, dtype=autocast_dtype):
                outputs = model(input_ids=inputs, attention_mask=attention_mask, ligand_idx=ligand_idx,
                                ligand_smiles=ligand_smiles)
                logits = outputs

                loss, smoothed_pos_weight = calculate_loss(
                    logits=logits,
                    labels=labels,
                    ligands=ligand_names,
                    smoothed_pos_weight=smoothed_pos_weight,
                    device=device,
                    alpha=alpha,
                    use_focal_loss=True,
                    gamma=gamma,
                    label_smoothing=label_smoothing,
                    **kwargs
                )
                test_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits.float()).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)  # Convert to binary labels
            labels_flat = labels.float().cpu().numpy().flatten()

            batch_size, seq_len = logits.shape[:2]
            for b in range(batch_size):
                ligand = batch["ligand"][b]
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

            flat_preds = torch.tensor(predictions, device=device).view(-1)
            flat_labels = labels.view(-1)
            accuracy.update(flat_preds, flat_labels)
            f1_score.update(flat_preds, flat_labels)
            precision.update(flat_preds, flat_labels)
            recall.update(flat_preds, flat_labels)
            specificity.update(flat_preds, flat_labels)
            mcc.update(flat_preds, flat_labels)
            auroc.update(flat_preds.float(), flat_labels)

            # Collect predictions and labels for further analysis
            all_predictions.extend(predictions)
            all_labels.extend(labels.float().cpu().numpy().flatten())

    # Compute Metrics
    avg_test_loss = test_loss / len(testloader)
    test_acc = accuracy.compute().cpu().item()
    test_f1 = f1_score.compute().cpu().item()
    test_precision = precision.compute().cpu().item()
    test_recall = recall.compute().cpu().item()
    test_specificity = specificity.compute().cpu().item()
    test_mcc = mcc.compute().cpu().item()
    test_auroc = auroc.compute().cpu().item()

    for metric in [accuracy, f1_score, precision, recall, specificity, mcc, auroc]:
        metric.reset()

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
    print(f"Specificity: {test_specificity:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {test_mcc:.4f}")
    print(f"AUROC: {test_auroc:.4f}")
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

    ligand_f1s = []
    for name, pairs in ligand_preds_labels.items():
        preds, labels = zip(*pairs)
        f1 = sk_f1(labels, preds, zero_division=0)
        ligand_f1s.append(f1)
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
        "specificity": test_specificity,
        "mcc": test_mcc,
        "auroc": test_auroc,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "incorrect_confidence_scores": incorrect_confidences if log_confidences else None
    }

def make_inferences(model, tokenizer, data_path, device, ligand, ligand2idx, max_length, prediction_threshold=0.5, **kwargs):
    """
    Make inferences on the test dataset using a trained checkpoint.

    Args:
        model: The trained model to use for inference.
        tokenizer: Tokenizer for the model.
        data_path (str): Path to the input data file.
        device: Device to run inference on (CPU/GPU).
        ligand: The ligand type to consider when making inferences.
        ligand2idx: Dictionary mapping ligands to indices.
        max_length (int): Maximum sequence length to consider for labels.
        prediction_threshold (float): Threshold for making binary predictions.
        kwargs: Additional parameters, including configs
    """

    import pandas as pd
    df = pd.read_csv(data_path)

    print("Preview of the input data:")
    print(df.head())
    # print length of the dataframe
    print("Length of the input data:", len(df))

    sequences = df['Amino Acid Sequence'].tolist()
    predictions = [""] * len(sequences)

    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) <= max_length]
    valid_sequences = [sequences[i] for i in valid_indices]
    model.eval()
    batch_size = 512
    ligand_idx = ligand2idx[ligand]

    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, len(valid_sequences), batch_size), desc="Running inference"):
            batch_seqs = valid_sequences[i:i + batch_size]
            encodings = tokenizer(batch_seqs, return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=max_length, add_special_tokens=False)
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            ligand_idx_batch = torch.full((len(batch_seqs),), ligand_idx, dtype=torch.long, device=device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, ligand_idx=ligand_idx_batch)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > prediction_threshold).long().cpu().numpy()

            for j, pred in enumerate(preds):
                seq_len = len(batch_seqs[j])
                pred_str = ''.join(str(x) for x in pred[:seq_len])
                predictions[valid_indices[i + j]] = pred_str

    # Append predictions
    df[f"{ligand}_Predictions"] = predictions
    # output_path = data_path.replace(".csv", f"_{ligand}_predictions.csv")
    output_path = data_path
    df.to_csv(output_path, index=False)
    return output_path


def main(dict_config, config_file_path):

    configs = load_configs(dict_config)
    # Flags for convenience

    general_settings = configs.general_settings
    train = general_settings.train if hasattr(general_settings, "train") else True
    on_hellbender = general_settings.on_hellbender if hasattr(general_settings, "on_hellbender") else True
    save_best_checkpoint = general_settings.save_best_checkpoint if hasattr(general_settings, "save_best_checkpoint") else True
    save_intermediate_checkpoints = general_settings.save_intermediate_checkpoints if hasattr(general_settings, "save_intermediate_checkpoints") else False
    use_checkpoint = general_settings.use_checkpoint if hasattr(general_settings, "use_checkpoint") else False
    visualize = general_settings.visualize if hasattr(general_settings, "visualize") else False
    test = general_settings.test if hasattr(general_settings, "test") else True
    inference = general_settings.inference if hasattr(general_settings, "inference") else False

    if use_checkpoint:
        if general_settings.checkpoint_path:
            load_checkpoint_path = general_settings.checkpoint_path
        else:
            load_checkpoint_path = "/home/dc57y/data/2025-06-05__18-40-14__1280_ZERO_SHOT_570_UNIMOL/checkpoints/checkpoint_epoch_6.pth"
    else:
        load_checkpoint_path = None


    if configs.use_fix_seed and isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    dataset_format = configs.dataset_format
    dataloaders = prepare_dataloaders(configs, data_format=dataset_format)
    # dataloaders = prepare_dataloaders(configs, debug=True, debug_subset_size=50)

    trainloader = dataloaders["train"]
    validloader = dataloaders["valid"]
    # Handling multiple test loaders for case of zero-shot dataset
    test_split_names = getattr(configs.test_settings, "test_splits", [])
    testloaders = []
    if test_split_names:
        testloaders = {name: dataloaders[name] for name in test_split_names if name in dataloaders}
        if len(testloaders) > 0:
            testloader = testloaders[list(testloaders.keys())[0]] # default
    else:
        testloader = dataloaders["test"]

    print("Finished preparing dataloaders")

    tokenizer, model = prepare_model(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.is_available() and hasattr(torch, "compile") and configs.model.use_compile:
        print("Compiling model with torch.compile()")
        model = torch.compile(model)

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
        start_epoch = load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, scaler) + 1
        # start_epoch = load_checkpoint(load_checkpoint_path, model=model, optimizer=None, scheduler=scheduler, scaler=scaler) + 1
        # start_epoch = load_checkpoint(load_checkpoint_path, model) + 1
        # start_epoch = 0

        if visualize:
            print("Visualizing predictions on test dataset")
            visualize_predictions(model, testloader, device, num_sequences=10, configs=configs)
            return

    if save_best_checkpoint:
        best_f1 = -1.0
        best_model_state = None
        best_epoch = -1

    if train:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path, save_to_data=on_hellbender)
        train_writer, valid_writer = prepare_tensorboard(result_path)

        for epoch in range(start_epoch, num_epochs):
            training_loss, training_f1 = training_loop(model, trainloader, optimizer, epoch, device, scaler, scheduler,
                                                       train_writer=train_writer, grad_clip_norm=grad_clip_norm,
                                                       alpha=alpha,gamma=gamma,
                                                       label_smoothing=label_smoothing, verbose=False,
                                                       configs=configs)

            valid_loss, _, macro_f1, precision, recall, specificity, mcc, auroc = validation_loop(model, validloader, epoch, device, valid_writer=valid_writer,
                                                   alpha=alpha, gamma=gamma,
                                                   label_smoothing=label_smoothing,configs=configs)
            scheduler.step()
            if save_best_checkpoint:
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    best_specificity = specificity
                    best_mcc = mcc
                    best_auroc = auroc
                    best_model_state = {
                        'model': copy.deepcopy(model.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict()),
                        'scheduler': copy.deepcopy(scheduler.state_dict()),
                        'scaler': copy.deepcopy(scaler.state_dict()),
                        'epoch': epoch
                    }
            if save_intermediate_checkpoints:
                if (epoch + 1) % checkpoint_every == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, checkpoint_path)
        if save_best_checkpoint and best_f1 > -1:
            model.load_state_dict(best_model_state['model'])
            optimizer.load_state_dict(best_model_state['optimizer'])
            scheduler.load_state_dict(best_model_state['scheduler'])
            scaler.load_state_dict(best_model_state['scaler'])
            save_checkpoint(model, optimizer, scheduler, scaler, best_model_state['epoch'], checkpoint_path)
        print(f"Best Validation Macro F1 Score: {best_f1:.4f} at epoch {best_epoch + 1}/{num_epochs}")
        print(f"Other metrics: Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, Specificity: {best_specificity:.4f}, MCC: {best_mcc:.4f}, AUROC: {best_auroc:.4f}")

    if test:
        print("Testing model on test dataset")
        if use_checkpoint and not train:
            load_checkpoint(load_checkpoint_path, model, optimizer, scheduler, scaler)
        if len(testloaders) > 0:
            for name, testloader in testloaders.items():
                print(f"Evaluating {name} dataset")
                results = evaluation_loop(model, testloader, device, log_confidences=False, alpha=alpha,
                                          gamma=gamma, label_smoothing=label_smoothing, configs=configs)
                # For convenience, saving important test results to CSV
                import csv
                from pathlib import Path
                results["model_name"] = configs.model.model_name
                results["hidden_size"] = configs.model.hidden_size
                results["test_set_name"] = name
                results["fix_seed"] = configs.fix_seed
                log_path = Path("logs/ablation_results.csv")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results.keys())
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(results)
        else:
            results = evaluation_loop(model, testloader, device, log_confidences=False, alpha=alpha, gamma=gamma,
                                      label_smoothing=label_smoothing, configs=configs)

    if inference:
        print("Making inferences on given dataset")
        inference_file_path = "/home/dc57y/ProteinLigand/training_code/data/uniref50_chunk_3.csv"
        ligand = "ZN"
        max_length = 768
        ligand2idx = build_ligand2idx(configs.ligands)
        prediction_threshold = 0.6
        make_inferences(model, tokenizer, inference_file_path, device, ligand, ligand2idx, max_length, prediction_threshold=prediction_threshold, configs=configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune the Ligand model')
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config.yaml')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
