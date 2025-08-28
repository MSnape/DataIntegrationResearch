import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset 
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score
import seaborn as sns 

# This method has an MLP as the fusion layer
# --- Define the MLP Fusion Classifier ---
class MLPFusionClassifier(nn.Module):
    """
    A simple Multi-Layer Perceptron to fuse logits from multiple views.
    Takes concatenated logits as input.
    """
    def __init__(self, input_dim, hidden_dim, num_classes=1, dropout_rate=0.2):
        super(MLPFusionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes) # Output single logit for binary classification

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# --- Modified Fusion Model to use MLP ---
class LogitMLPFusionModel:
    """
    Orchestrates the two-stage late fusion model using SingleViewCNN for each view
    and an MLP for non-linear fusion of logits.
    1. (Optional) Trains individual SingleViewCNN branches (CC and MLO) if not provided.
    2. Extracts logits from these trained branches on the training data.
    3. Trains an MLP model on the concatenated logits.
    4. Evaluates the full pipeline on validation/test data.
    """
    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 cc_cnn_model: 'SingleViewCNN' = None, 
                 mlo_cnn_model: 'SingleViewCNN' = None,
                 mlp_hidden_dim: int = 64, # New: MLP hidden layer size
                 mlp_learning_rate: float = 0.001, # New: MLP learning rate
                 mlp_num_epochs: int = 50, # New: MLP epochs
                 mlp_batch_size: int = 32, # New: MLP batch size
                 mlp_patience: int = 10): # New: MLP early stopping patience

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cc_cnn = cc_cnn_model
        self.mlo_cnn = mlo_cnn_model

        # MLP Fusion Classifier
        # Input dimension will be 2 (one logit from CC, one from MLO)
        self.mlp_fusion_classifier = MLPFusionClassifier(
            input_dim=2,
            hidden_dim=mlp_hidden_dim,
            num_classes=1 # Binary classification output
        ).to(self.device)

        self.mlp_learning_rate = mlp_learning_rate
        self.mlp_num_epochs = mlp_num_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_patience = mlp_patience

        print(f"Initialized MLP Fusion Classifier:\n{self.mlp_fusion_classifier}")


    def train_fusion_pipeline(self, num_epochs=None, patience=None, subdirectory: str = None,
                              cc_pos_weight=None, mlo_pos_weight=None):
        """
        Trains the entire fusion pipeline: individual CNNs then MLP Fusion.
        Individual CNNs are trained only if they were not provided at initialization.
        """
        # --- Stage 1: Train Individual SingleViewCNN Models (if not provided) ---
        if self.cc_cnn is None:
            print("\n--- Stage 1: Initializing and Training CC SingleViewCNN Model ---")
            if num_epochs is None or patience is None:
                raise ValueError("num_epochs and patience must be provided to train individual CNNs.")
            # Assuming self.train_dataset, etc. are the *wrapped single-view* datasets for CC/MLO.
            # If not, you'd need to explicitly pass the wrapped datasets to SingleViewCNN.
            self.cc_cnn = SingleViewCNN(
                train_dataset=self.train_dataset, # Assuming this is a CC-specific wrapped dataset
                validation_dataset=self.val_dataset, # Assuming this is a CC-specific wrapped dataset
                test_dataset=self.test_dataset, # Assuming this is a CC-specific wrapped dataset
                view_type="CC"
            )
            _, _, cc_test_performance = self.cc_cnn.Train(num_epochs, patience, subdirectory, pos_weight=cc_pos_weight)
            print(f"\nCC Test Performance after training: {cc_test_performance}")
        else:
            print("\n--- Stage 1: Using Pre-trained CC SingleViewCNN Model ---")

        if self.mlo_cnn is None:
            print("\n--- Stage 1: Initializing and Training MLO SingleViewCNN Model ---")
            if num_epochs is None or patience is None:
                raise ValueError("num_epochs and patience must be provided to train individual CNNs.")
            self.mlo_cnn = SingleViewCNN(
                train_dataset=self.train_dataset, # Assuming this is an MLO-specific wrapped dataset
                validation_dataset=self.val_dataset, # Assuming this is an MLO-specific wrapped dataset
                test_dataset=self.test_dataset, # Assuming this is an MLO-specific wrapped dataset
                view_type="MLO"
            )
            _, _, mlo_test_performance = self.mlo_cnn.Train(num_epochs, patience, subdirectory, pos_weight=mlo_pos_weight)
            print(f"\nMLO Test Performance after training: {mlo_test_performance}")
        else:
            print("\n--- Stage 1: Using Pre-trained MLO SingleViewCNN Model ---")


        # --- Stage 2: Extract Logits and Train MLP Fusion Classifier ---
        print("\n--- Stage 2: Extracting Logits and Training MLP Fusion Classifier ---")

        # Create DataLoaders for logit extraction for the specific single-view datasets
        # that were used for training the individual CNNs.
        cc_train_ds_for_logits = self.cc_cnn.torch_dataset_train_view
        mlo_train_ds_for_logits = self.mlo_cnn.torch_dataset_train_view

        cc_train_loader_for_logits = DataLoader(cc_train_ds_for_logits, batch_size=self.cc_cnn.BATCH_SIZE, shuffle=False)
        mlo_train_loader_for_logits = DataLoader(mlo_train_ds_for_logits, batch_size=self.mlo_cnn.BATCH_SIZE, shuffle=False)

        cc_train_logits, training_labels_for_mlp = self.cc_cnn.get_trained_model_logits(
            cc_train_loader_for_logits, "CC Train Logit Extraction"
        )
        mlo_train_logits, _ = self.mlo_cnn.get_trained_model_logits(
            mlo_train_loader_for_logits, "MLO Train Logit Extraction"
        )

        if not np.array_equal(training_labels_for_mlp, _):
            raise ValueError("Labels extracted from CC and MLO training sets do not match!")

        # Prepare data for MLP training
        X_train_mlp = np.hstack((cc_train_logits.reshape(-1, 1), mlo_train_logits.reshape(-1, 1)))
        y_train_mlp = training_labels_for_mlp

        # Convert to PyTorch tensors and create DataLoader for MLP
        X_train_tensor = torch.tensor(X_train_mlp, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_mlp, dtype=torch.float32).unsqueeze(1) # For BCEWithLogitsLoss

        mlp_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        mlp_train_loader = DataLoader(mlp_train_dataset, batch_size=self.mlp_batch_size, shuffle=True)

        # Set up MLP training
        optimizer_mlp = optim.Adam(self.mlp_fusion_classifier.parameters(), lr=self.mlp_learning_rate)
        criterion_mlp = nn.BCEWithLogitsLoss()

        best_mlp_val_loss = float('inf')
        epochs_no_improvement_mlp = 0
        best_mlp_wts = copy.deepcopy(self.mlp_fusion_classifier.state_dict())

        self.mlp_fusion_classifier.train()
        for epoch in range(self.mlp_num_epochs):
            total_loss = 0
            for i, (inputs, labels) in enumerate(mlp_train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer_mlp.zero_grad()
                outputs = self.mlp_fusion_classifier(inputs)
                loss = criterion_mlp(outputs, labels)
                loss.backward()
                optimizer_mlp.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(mlp_train_loader)

            # Evaluate MLP on validation logits (from the CNNs on validation set)
            val_loss, val_acc, _, _, _ = self._evaluate_mlp_on_logits(
                self.mlp_fusion_classifier, "validation", criterion_mlp
            )

            print(f"MLP Epoch [{epoch+1}/{self.mlp_num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_mlp_val_loss:
                best_mlp_val_loss = val_loss
                best_mlp_wts = copy.deepcopy(self.mlp_fusion_classifier.state_dict())
                epochs_no_improvement_mlp = 0
                print(f"MLP validation loss improved to {best_mlp_val_loss:.4f}. Saving MLP weights.")
            else:
                epochs_no_improvement_mlp += 1
                print(f"MLP validation loss did not improve. Patience: {epochs_no_improvement_mlp}/{self.mlp_patience}")

            if epochs_no_improvement_mlp >= self.mlp_patience:
                print(f"MLP early stopping triggered after {epoch + 1} epochs.")
                break

        self.mlp_fusion_classifier.load_state_dict(best_mlp_wts)
        print("MLP Fusion Classifier Training Complete.")

    def _evaluate_mlp_on_logits(self, mlp_model, dataset_type: Literal["training", "validation", "test"], criterion):
        """
        Helper to get logits from CNNs for a specified dataset type and evaluate the MLP on them.
        Returns loss, accuracy, and optionally logits/labels for plotting.
        """
        if dataset_type == "training":
            cc_ds = self.cc_cnn.torch_dataset_train_view
            mlo_ds = self.mlo_cnn.torch_dataset_train_view
            name = "Train"
        elif dataset_type == "validation":
            cc_ds = self.cc_cnn.torch_dataset_val_view
            mlo_ds = self.mlo_cnn.torch_dataset_val_view
            name = "Validation"
        elif dataset_type == "test":
            cc_ds = self.cc_cnn.torch_dataset_test_view
            mlo_ds = self.mlo_cnn.torch_dataset_test_view
            name = "Test"
        else:
            raise ValueError("dataset_type must be 'training', 'validation', or 'test'.")

        cc_loader = DataLoader(cc_ds, batch_size=self.cc_cnn.BATCH_SIZE, shuffle=False)
        mlo_loader = DataLoader(mlo_ds, batch_size=self.mlo_cnn.BATCH_SIZE, shuffle=False)

        cc_logits, labels = self.cc_cnn.get_trained_model_logits(cc_loader, f"CC {name} Logit Eval")
        mlo_logits, _ = self.mlo_cnn.get_trained_model_logits(mlo_loader, f"MLO {name} Logit Eval")

        if not np.array_equal(labels, _):
            raise ValueError(f"Labels extracted for MLP evaluation on {name} do not match!")

        X_mlp_eval = np.hstack((cc_logits.reshape(-1, 1), mlo_logits.reshape(-1, 1)))
        y_mlp_eval = labels

        X_eval_tensor = torch.tensor(X_mlp_eval, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_mlp_eval, dtype=torch.float32).unsqueeze(1).to(self.device)

        mlp_model.eval()
        with torch.no_grad():
            outputs = mlp_model(X_eval_tensor)
            loss = criterion(outputs, y_eval_tensor).item()
            preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy().flatten()
            acc = accuracy_score(y_mlp_eval, preds)

        return loss, acc, outputs.cpu().numpy().flatten(), y_mlp_eval, preds

    def evaluate_fusion_model(self, dataset_type: Literal["validation", "test"]):
        """
        Evaluates the combined MLP Fusion model on the specified dataset.
        """
        if self.mlp_fusion_classifier is None:
            raise RuntimeError("MLP Fusion classifier has not been trained yet. Call train_fusion_pipeline first.")

        print(f"\n--- Evaluating Full MLP Fusion Model on {dataset_type.capitalize()} Set ---")

        # Get evaluation metrics directly from the _evaluate_mlp_on_logits helper
        total_loss, total_acc, final_logits, final_labels, final_preds = self._evaluate_mlp_on_logits(
            self.mlp_fusion_classifier, dataset_type, nn.BCEWithLogitsLoss()
        )

        lr_probabilities = torch.sigmoid(torch.tensor(final_logits)).cpu().numpy() # Convert logits to probabilities

        roc_auc = roc_auc_score(final_labels, lr_probabilities)
        cm = confusion_matrix(final_labels, final_preds)

        print(f"Fusion Model {dataset_type.capitalize()} Accuracy: {total_acc:.4f}")
        print(f"Fusion Model {dataset_type.capitalize()} ROC AUC: {roc_auc:.4f}")
        print(f"Fusion Model {dataset_type.capitalize()} Confusion Matrix:\n{cm}")

        show_confusion_matrix(cm, f"MLP Fusion Model ({dataset_type.capitalize()} Set)")
        tn, fp, fn, tp = cm.ravel()
        performance_results = get_performance_metrics(tp, fp, tn, fn, f"MLP Fusion Model {dataset_type.capitalize()}")

        fpr, tpr, _ = roc_curve(final_labels, lr_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - MLP Fusion Model ({dataset_type.capitalize()} Set)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        return total_acc, roc_auc, cm, performance_results
