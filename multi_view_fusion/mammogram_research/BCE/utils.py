# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re
import time
import copy
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from moe import MoEViewFusion

""" 
Utility functions used by many of the fusion models. 
"""

def extract_images_by_abnormality_id(data_map, view_type):
    """
    Extracts all 'image' values from a nested map where 'image_view'
    matches the pattern 'image_view_<abnormality_id>' for each entity.

    Args:
        data_map (dict): A dictionary, each key is an entity_id
                         and each value is an entity dictionary.
                         Each entity dictionary has a 'images' member,
                         which is a set of tuples/lists, each containing
                         'image_view', 'image', and 'file_path'.
                         Each entity has an 'abnormality_id' member.

    Returns:
        list: A list of all 'image' values that match the criteria.
    """
    all_matching_images = []

    for idx, entity_data in enumerate( data_map):
        abnormality_id = entity_data['abnormality id']

        # Ensure abnormality_id exists and is not None
        if abnormality_id is None:
            print(f"Warning: Entity '{entity_data}' is missing 'abnormality_id'. Skipping.")
            continue

        expected_image_view_prefix = f"{view_type}_{abnormality_id}"

        if len(entity_data['images']) == 4:
            for image in entity_data['images']:
            # Need to have the 2 CC images and 2 MLO images
                if image['image_view'] == expected_image_view_prefix and image['file_path'].endswith('1-1.dcm'):
                    all_matching_images.append(image['image'])
                    if len(all_matching_images) > idx + 1:
                        print(f"Image file name issue: error on : index {idx} and path {image['file_path']}")
    
    return all_matching_images

def print_performance_metrics(tp, fp,tn,fn):
    """
    Prints performance metrics calculated from TP, FP, TN and FN

    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives
        fn (int): False Negatives
    Returns:
      
    """
    print(f"Accuracy:  {(tp+tn)/(tp+ fp+tn+fn):.4f}")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:  {recall:.4f}")
    print(f"Specificity:  {tn/(tn+fn):.4f}")
    print(f"F1-Score:  {2*(precision*recall)/(precision+recall):.4f}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")


def get_performance_metrics(tp, fp, tn, fn, view_name, roc_auc=None):
    """
    Calculates and returns performance metrics, including AUC if provided.

    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives
        fn (int): False Negatives
        view_name (str): Name of the view/model for labeling.
        roc_auc (float, optional): The AUC score from the ROC curve. Defaults to None.

    Returns:
        dict: A dictionary containing calculated performance metrics.
    """
    metrics = {}
    metrics['Name'] = view_name

    # Calculate basic metrics, handling potential division by zero
    total_samples = (tp + fp + tn + fn)
    metrics['Accuracy'] = (tp + tn) / total_samples if total_samples > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity : True Negatives / (True Negatives + False Positives)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['Specificity'] = specificity

    # F1-Score calculation
    if (precision + recall) > 0:
        metrics['F1-Score'] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics['F1-Score'] = 0

    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn

    # Include AUC if it was provided
    if roc_auc is not None:
        metrics['AUC'] = roc_auc
    else:
        # If AUC is not provided, set it to NaN 
        metrics['AUC'] = float('nan') 

    return metrics
    
# This adaptation was used for debugging for MOE Gated Fusion which did not give better results
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, patience, scheduler=None):
    """
        Train the model on the train dataset with validation and early stopping.
    Args:
        model (MultiView): trained multi view model
        train_loader (DataLoader): train dataset loader
        val_loader (DataLoader): validation dataset loader
        test_loader (DataLoader): test dataset loader (for final evaluation, not early stopping)
        criterion () : loss function
        optimizer () :
        device () : device to use
        num_epochs (int): number of epochs to run for
        patience (int): patience to use in the EarlyStopping
    """
    # Consolidate dataloaders for easy iteration
    dataloaders = {'train': train_loader, 'val': val_loader} # Early stopping based on 'val'

    print("Starting training with validation and early stopping.")
    since = time.time() # To measure training time

    best_val_loss = float('inf') # Initialize with a very high value for validation loss
    epochs_no_improvement = 0
    best_model_state = None

    # These will store predictions and labels from the TEST set for the best model found during training
    best_model_test_predictions = []
    all_test_labels = []

    # Store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Collect all test labels once, as they don't change
    for _, labels in test_loader:
        all_test_labels.extend(labels.cpu().numpy())

    # Flag to check if the current model is an MoEViewFusion instance
    is_moe_model = isinstance(model, MoEViewFusion)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Iterate over training and validation phases
        for phase in ['train', 'val']:
            if phase =='train':
                model.train() 
                current_dataloader = dataloaders['train']
                # if (epoch+1) % 5 == 0: # Print every 5 epochs
                print(f"\nEpoch {epoch+1}/{num_epochs} - Training Phase")
                # Initialize for this epoch
                if is_moe_model:
                    gate_weights_history_epoch = []
                    gate_raw_scores_history_epoch = []
            else: # phase == 'val'
                model.eval() 
                current_dataloader = dataloaders['val'] # Use validation dataloader
                # if (epoch+1) % 5 == 0: # Print every 5 epochs 
                print(f"\nEpoch {epoch+1}/{num_epochs} - Validation Phase")

            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Debugging: Lists to store gradient norms for analysis per epoch
            if is_moe_model:
                gating_network_grads = {name: [] for name, param in model.gating_network.named_parameters()}


            for batch_idx, (views, labels) in enumerate(current_dataloader):
                
                views = [v.to(device) for v in views]
                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients
                with torch.set_grad_enabled(phase=='train'): # Only enable grad for training phase
                    if is_moe_model:
                        combined_logits, gate_raw_scores, gate_weights = model(views) 
                        outputs = combined_logits 
                    else: 
                        outputs = model(views)
                    # Binary Cross entropy loss will apply LogSoftMax to the raw logits (outputs)
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                    
                    if phase == 'train':
                        loss.backward()
                        # --- DEBUGGING: Check Gradients of Gating Network ---
                        # the gradients disappeared and so this debugging was included
                        if is_moe_model:
                            # Store for epoch summary
                            gate_weights_history_epoch.append(gate_weights.mean(dim=0).detach().cpu().numpy())
                            gate_raw_scores_history_epoch.append(gate_raw_scores.mean(dim=0).detach().cpu().numpy())

                            # Print at specific batches for real-time observation
                            if batch_idx == 0 or batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
                                print(f"  [DEBUG] Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader) - 1}")
                                for name, param in model.gating_network.named_parameters():
                                    if param.grad is not None:
                                        grad_norm = param.grad.norm().item()
                                        gating_network_grads[name].append(grad_norm) # Store for summary
                                        if grad_norm < 1e-9: # Check for near-zero gradients
                                            print(f"    WARNING: Gradient for {name} is near zero ({grad_norm:.8f})!")
                                        else:
                                            print(f"    {name} grad norm: {grad_norm:.8f}")
                                    else:
                                        print(f"    WARNING: Gradient for {name} is None!")
                                
                                print(f"    Gating Raw Scores (mean): {gate_raw_scores.mean(dim=0).detach().cpu().numpy()}")
                                print(f"    Gating Raw Scores (min/max): {gate_raw_scores.min().item():.4f}/{gate_raw_scores.max().item():.4f}")
                                print(f"    Gating Weights (mean): {gate_weights.mean(dim=0).detach().cpu().numpy()}")
                                print(f"    Gating Weights (min/max): {gate_weights.min().item():.4f}/{gate_weights.max().item():.4f}")

                        optimizer.step()

                running_loss += loss.item() * labels.size(0)               
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long().squeeze()
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples

            print(f"{phase.capitalize()} Results: Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # Store metrics for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)
            elif phase == 'val':
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_accuracy)

            # Early stopping based on validation loss
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improvement = 0 # Reset counter
                    # Save the model state dict if the validation loss is better
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"Validation loss improved to {best_val_loss:.4f}. Saving model weights.")
                else:
                    epochs_no_improvement += 1
                    print(f"Validation loss did not improve. Patience: {epochs_no_improvement}/{patience}")

        # --- DEBUGGING: Summary of gradients and gate outputs for the epoch ---
        if is_moe_model:
            print(f"--- Epoch {epoch+1} Gating Network Gradient Summary ---")
            for name, grads_list in gating_network_grads.items():
                if grads_list:
                    print(f"  {name} Avg Grad Norm: {np.mean(grads_list):.8f}, Max Grad Norm: {np.max(grads_list):.8f}")
                else:
                    print(f"  No gradients recorded for {name} in this epoch's debug prints (check batch_idx conditions).")

            if gate_weights_history_epoch:
                avg_gate_weights_epoch = np.mean(gate_weights_history_epoch, axis=0)
                avg_gate_raw_scores_epoch = np.mean(gate_raw_scores_history_epoch, axis=0)
                print(f"  Epoch Average Gating Raw Scores: {avg_gate_raw_scores_epoch}")
                print(f"  Epoch Average Gating Weights: {avg_gate_weights_epoch}")
            else:
                print("  No gating output history recorded for this epoch.")

        if scheduler is not None:
            # Step the learning rate scheduler AFTER the training and validation phases of the current epoch
            scheduler.step()
            # Print current learning rate to observe its decay
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")

        # Check for early stopping after both train and val phases in an epoch
        if epochs_no_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
            break

    # Load the best model state found during training (based on validation loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state was saved. This might happen if num_epochs is 0 or patience is too low.")

    is_moe_model = isinstance(model, MoEViewFusion)

    # After training, get predictions for the test set using the best model
    model.eval() 
    with torch.no_grad():
        for views, labels in test_loader:
            views = [v.to(device) for v in views]
            if is_moe_model:
                combined_logits, gate_raw_scores, gate_weights = model(views) 
                outputs = combined_logits 
            else: 
                outputs = model(views)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long().squeeze()
            best_model_test_predictions.extend(predicted.cpu().numpy())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Loss: {best_val_loss:.4f}')

    # Plotting training vs validation accuracy and loss
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, best_model_state, best_model_test_predictions, all_test_labels

def show_confusion_matrix(cm, view_type):
    """
        Displays confusion matrix using sns.
    Args:
        cm (confusion_matrix)
    """
    # Visualize the confusion matrix 
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {view_type}')
    plt.show()

def evaluate_model(model, test_loader, device):
    """
        Evaluates the model on the test dataset.

    Args:
        model (MultiView): trained multi view model
        test_loader (DataLoader): test dataset loader
        device () : device to use        
    """
    model.eval() # Set model to evaluation mode
    print("\nStarting evaluation...")
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = [] # To store probabilities for ROC/AUC
    is_moe_model = isinstance(model, MoEViewFusion)
    with torch.no_grad(): # Disable gradient calculation for inference
        for views, labels in test_loader:
            views = [v.to(device) for v in views]
            labels = labels.to(device)
            if is_moe_model:
                combined_logits, gate_raw_scores, gate_weights = model(views) 
                outputs = combined_logits 
            else: 
                outputs = model(views)
            probs = torch.sigmoid(outputs)      # Apply sigmoid to get probabilities
            predicted = (probs > 0.5).long().squeeze(1)  # Threshold and squeeze

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy()) 

    accuracy = correct_predictions / total_samples

    print(f"Evaluation Complete. Test Accuracy: {accuracy:.4f}")
    
    # Calculate the TN, FP, FN and TP, and use to print out the Confusion matrix 
    cm = confusion_matrix(all_labels, all_predictions)
    show_confusion_matrix(cm, "Multi-View Model")
    tn, fp, fn, tp = cm.ravel()
    print("In evaluate_model tp, fp, tn, fn, ", tp, fp, tn, fn)
    # --- Add AUC and ROC Curve Calculation ---
    all_probabilities_np = np.array(all_probabilities).flatten()
    all_labels_np = np.array(all_labels)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probabilities_np)
    
    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    results = get_performance_metrics(tp, fp, tn, fn, "Multi-View Model", roc_auc=roc_auc)

    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return accuracy, results #, roc_auc


def print_single_view_results_table(view1, view2):
    """
        Prints a single table of the performance results of the 2 Single View models
    """
    print(f"{'Metric':<15} {view1['Name']:>10} {view2['Name']:>10}")
    print(f"{'-'*15} {'-'*10} {'-'*10}")

    for metric_name in view1.keys(): # Assuming both dicts have same keys in same order
        if metric_name != 'Name':
            value1 = view1[metric_name]
            value2 = view2[metric_name]
            if isinstance(value1, float):
                print(f"{metric_name:<15} {value1:>10.4f} {value2:>10.4f}")
            else:
                print(f"{metric_name:<15} {value1:>10} {value2:>10}")

def compare_performance_all_views(view1, view2, multiview):
    """
        Prints a single table of the performance results of the 3  models
    """
    view1_name = view1['Name']
    view2_name = view2['Name']
    view3_name = multiview['Name']

    data_for_df = {
        view1_name: {},
        view2_name : {},
        view3_name : {}
    }

    # Populate data for each view
    for metric_name, value in view1.items():
        if metric_name != 'Name':
            data_for_df[view1_name][metric_name] = value

    for metric_name, value in view2.items():
        if metric_name != 'Name':
            data_for_df[view2_name][metric_name] = value

    for metric_name, value in multiview.items():
        if metric_name != 'Name':
            data_for_df[view3_name][metric_name] = value

    # Create the initial DataFrame
    df = pd.DataFrame.from_dict(data_for_df)
    print(df)