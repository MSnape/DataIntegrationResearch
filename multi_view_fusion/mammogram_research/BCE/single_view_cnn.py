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
from typing import Literal
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import local code
from utils import extract_images_by_abnormality_id, show_confusion_matrix, print_performance_metrics, get_performance_metrics

class SingleViewCNN:
    """
    Single View Model which uses a pretrained CNN (ResNet-50).
    """
    def __init__(self, train_dataset: Dataset, validation_dataset: Dataset, test_dataset: Dataset, view_type: Literal):
        """
        Initializes the dataset.
        Args:
            train_dataset : Training dataset.
            validation_dataset : Validation dataset.
            test_dataset : Test dataset.
            view_type (Literal): Name of the view type, used for various print statements.
        """
        self.view_type = view_type
        self.torch_dataset_train_view = train_dataset
        self.torch_dataset_val_view = validation_dataset
        self.torch_dataset_test_view = test_dataset
        self.saved_model_path = ""

    def Train(self, num_epochs, patience,  subdirectory: str = None, pos_weight=None):
        """
        Trains Resnet50 CNNs with each view and check how well they classify on the test set for each view

        Args:
            num_epochs : number of epochs to run for
            patience : patience to use in the EarlyStopping

        """
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        FREEZE_FEATURES = True
        self.subdirectory = subdirectory

        train_dataloader = DataLoader(self.torch_dataset_train_view, batch_size=BATCH_SIZE, shuffle=True) 
        val_dataloader = DataLoader(self.torch_dataset_val_view, batch_size=BATCH_SIZE, shuffle=False)  
        test_dataloader = DataLoader(self.torch_dataset_test_view, batch_size=BATCH_SIZE, shuffle=False)  

        dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
        dataset_sizes = {'train': len(self.torch_dataset_train_view),
                         'val': len(self.torch_dataset_val_view),
                         'test': len(self.torch_dataset_test_view)}

        print(f"\nTrain DataLoader has {len(train_dataloader)} batches. Dataset size: {dataset_sizes['train']}")
        print(f"Validation DataLoader has {len(val_dataloader)} batches. Dataset size: {dataset_sizes['val']}")
        print(f"Test DataLoader has {len(test_dataloader)} batches. Dataset size: {dataset_sizes['test']}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if FREEZE_FEATURES:
            for param in model_ft.parameters():
                param.requires_grad = False
            print("Frozen all parameters except the final layer.")

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)
        model_ft = model_ft.to(device)

        print(f"Modified ResNet50 final layer: {model_ft.fc}")

        # Binary Cross Entropy
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Freezing parameters as we do not want to lose the features ResNet-50 has learnt
        if FREEZE_FEATURES:
            optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=LEARNING_RATE)
        else:
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        def train_model(model, criterion, optimizer, scheduler, num_epochs, patience):
            since = time.time()
            best_val_loss = float('inf')  
            best_val_acc = 0.0
            epochs_no_improvement = 0
            best_model_wts = copy.deepcopy(model.state_dict()) # To save the best model weights

            # For plotting
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            # Prepare for collecting test predictions from the best model
            best_model_test_predictions = []
            all_test_labels = []

            # Populate all_test_labels once
            for _, labels in dataloaders['test']:
                all_test_labels.extend(labels.cpu().numpy())

            for epoch in range(num_epochs):
                print(f'Epoch {epoch+1}/{num_epochs}')
                print('-' * 10)

                # Each epoch has a training, validation, and (optional) test phase
                # Early stopping is based on 'val' phase
                for phase in ['train', 'val']: # We will use 'val' for early stopping
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # Set the gradients to zero before starting to do backpropagation
                        optimizer.zero_grad()

                        # Forward
                        # Track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            preds = torch.sigmoid(outputs) >= 0.5
                            preds = preds.int().squeeze(1)  # (batch_size,)
                            loss = criterion(outputs, labels.float().unsqueeze(1))

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)
                       

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # Store metrics for plotting
                    if phase == 'train':
                        train_losses.append(epoch_loss)
                        train_accuracies.append(epoch_acc.item()) # .item() to get scalar from tensor
                    elif phase == 'val':
                        val_losses.append(epoch_loss)
                        val_accuracies.append(epoch_acc.item()) # .item() to get scalar from tensor

                    # Save the model weights if it's the best validation loss
                    if phase == 'val':
                        if epoch_loss < best_val_loss:
                            best_val_loss = epoch_loss
                            best_val_acc = epoch_acc
                            best_model_wts = copy.deepcopy(model.state_dict())
                            epochs_no_improvement = 0 # Reset counter
                            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model weights.")
                        else:
                            epochs_no_improvement += 1
                            print(f"Validation loss did not improve. Patience: {epochs_no_improvement}/{patience}")

                if epochs_no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss.")
                    break

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best Validation Loss: {best_val_loss:.4f}')
            print(f'Best Validation Accuracy: {best_val_acc:.4f}')

            # Load best model weights
            model.load_state_dict(best_model_wts)

            # After training, evaluate on test set with the best model
            # This part is just for getting the predictions from the best model
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['test']:
                    inputs = inputs.to(device)
                    outputs = model(inputs)                   
                    preds = torch.sigmoid(outputs) >= 0.5
                    preds = preds.int().squeeze(1)  # (batch_size,)
                    best_model_test_predictions.extend(preds.cpu().numpy())

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


            return model, best_model_test_predictions, all_test_labels

        print("\nStarting training with validation and early stopping...")
        model_ft, best_model_test_predictions, all_test_labels = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, patience
        )

        def evaluate_model(model, dataloader, criterion, dataset_size, phase_name="Test"):
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            all_probabilities = [] # To store probabilities for ROC/AUC

            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)  
                    preds = (probs >= 0.5).int().squeeze(1)
                    loss = criterion(outputs, labels.float().unsqueeze(1))

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probabilities.extend(probs.cpu().numpy()) # Store probabilities

            total_loss = running_loss / dataset_size
            total_acc = running_corrects.double() / dataset_size
            print(f'Final {phase_name} Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

            cm = confusion_matrix(all_labels, all_preds)
            show_confusion_matrix(cm, self.view_type + f" Single View Model ({phase_name} Set)")
            tn, fp, fn, tp = cm.ravel()

            # --- Add AUC and ROC Curve Calculation ---
            all_probabilities_np = np.array(all_probabilities).flatten()
            all_labels_np = np.array(all_labels)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(all_labels_np, all_probabilities_np)
            
            # Calculate AUC (Area Under the Curve)
            roc_auc = auc(fpr, tpr)
            print(f"ROC AUC: {roc_auc:.4f}")
            performance_results = get_performance_metrics(tp, fp, tn, fn, self.view_type + " SingleView " + phase_name, roc_auc)
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

            return total_loss, total_acc, performance_results

        print("\nFinal evaluation on the test set using the best model (based on validation performance):")
        filename = f"model_{self.view_type}_best_val_model.pth"

        # Determine the full path where the model will be saved
        if self.subdirectory:
            # If a subdirectory is provided, join it with the filename
            # os.path.join handles path separators across different OS
            self.saved_model_path = os.path.join(self.subdirectory, filename)
        else:
            # If no subdirectory is provided, save in the current directory
            self.saved_model_path = f"./{filename}"
        print("Model will be saved to : " + self.saved_model_path)
        torch.save(model_ft.state_dict(), self.saved_model_path) # Save state_dict, not the entire model for flexibility

        total_loss, total_acc, performance_results = evaluate_model(model_ft, dataloaders['test'], criterion, dataset_sizes['test'])

        return best_model_test_predictions, all_test_labels, performance_results

    def evaluate_and_get_logits(self, model, dataloader, criterion, dataset_size, phase_name="Test"):
        """
        Evaluates the model and returns test predictions, actual labels, performance metrics,
        and the raw logits. Needed by some of the fusion models.
        """
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        all_probabilities = []
        all_logits = [] # Store raw logits here

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.int()
                
                # Outputs are raw logits 
                outputs = model(inputs) 
                # Probabilities
                probs = torch.sigmoid(outputs)
                # Binary predictions
                preds = (probs >= 0.5).int()

                running_corrects += torch.sum(preds.squeeze() == labels.int())

                loss = criterion(outputs, labels.float().unsqueeze(1)) 
                running_loss += loss.item() * inputs.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())

        total_loss = running_loss / dataset_size
        total_acc = running_corrects.double() / dataset_size

        print(f'DEBUG : running_corrects {running_corrects} and dataset size {dataset_size}')
        print(f'Final {phase_name} Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

        cm = confusion_matrix(all_labels, all_preds)
        show_confusion_matrix(cm, self.view_type + f" Single View Model ({phase_name} Set)")
        tn, fp, fn, tp = cm.ravel()

        performance_results = get_performance_metrics(tp, fp, tn, fn, self.view_type + " SingleView " + phase_name)

        all_probabilities_np = np.array(all_probabilities).flatten()
        all_labels_np = np.array(all_labels).flatten()

        roc_auc = roc_auc_score(all_labels_np, all_probabilities_np)
        print(f"ROC AUC: {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(all_labels_np, all_probabilities_np)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve ({self.view_type} - {phase_name} Set)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        # Return logits along with other metrics
        return total_loss, total_acc, performance_results, np.array(all_logits).flatten(), all_labels_np

    def get_trained_model_logits(self, data_loader, view_name):
        """
        Loads the best trained model and extracts logits for a given data loader.
        Used for the some of the fusion models.
        """
        if self.saved_model_path != "":
            # Attempt to load from saved path if not in memory
            if os.path.exists(self.saved_model_path):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, 1)
                model_ft.load_state_dict(torch.load(self.saved_model_path, map_location=device))
                model_ft.to(device)
                self.best_trained_model = model_ft
                print(f"Loaded best trained model for {self.view_type} from {self.saved_model_path}")
            else:
                raise RuntimeError(f"Model for {self.view_type} has not been trained or saved model not found at {self.saved_model_path}. Please call Train() first.")

        # Re-use the existing evaluation logic to get logits
        _, _, _, logits, labels = self.evaluate_and_get_logits(
            self.best_trained_model, data_loader, nn.BCEWithLogitsLoss(), len(data_loader.dataset), f"{view_name} Logit Extraction"
        )
        return logits, labels
