
# Import all necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import os
import pandas as pd
import pydicom
import re
import numpy as np
import matplotlib.pyplot as plt 
import time
import copy
from typing import Literal
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score

from single_view_cnn import SingleViewCNN
from utils import show_confusion_matrix, get_performance_metrics

# In the report this is renamed Multi-View Stacking using Logistic Regression (MVS LR)
# From Van Loon et al 2024 https://arxiv.org/pdf/2010.16271 as this 
#  LateFusionLR Model which uses the logits from the SingleView models as inputs into a Logistic Regression model
class LateFusionLR:
    """
    Two-stage late fusion model:
    1. (Optional and not recommended) Trains individual SingleViewCNN branches (CC and MLO) if not provided. Will not normally be used.
    2. Extracts logits from these trained branches on the training data.
    3. Trains a Logistic Regression model on the concatenated logits.
    4. Evaluates the full pipeline on validation/test data.
    """
    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 cc_cnn_model: SingleViewCNN,  
                 mlo_cnn_model: SingleViewCNN): 

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Use provided models if available, otherwise prepare for training them internally
        self.cc_cnn = cc_cnn_model
        self.mlo_cnn = mlo_cnn_model

        # Ensure that if models are provided, their datasets match the fusion model's intention
        '''if self.cc_cnn and (self.cc_cnn.torch_dataset_train_view is not self.train_dataset or
                            self.cc_cnn.torch_dataset_val_view is not self.val_dataset or
                            self.cc_cnn.torch_dataset_test_view is not self.test_dataset):
             # This check assumes `torch_dataset_X_view` refers to the *single-view wrapped* datasets.
             print("Warning: Provided CC CNN datasets might not align with fusion model's datasets.")
        if self.mlo_cnn and (self.mlo_cnn.torch_dataset_train_view is not self.train_dataset or
                             self.mlo_cnn.torch_dataset_val_view is not self.val_dataset or
                             self.mlo_cnn.torch_dataset_test_view is not self.test_dataset):
             print("Warning: Provided MLO CNN datasets might not align with fusion model's datasets.")'''
        self.logistic_regression_model = None

    def train_fusion_pipeline(self, num_epochs=None, patience=None, subdirectory: str = None,
                              cc_pos_weight=None, mlo_pos_weight=None):
        """
        Trains the entire fusion pipeline: individual CNNs then Logistic Regression.
        Individual CNNs are trained only if they were not provided at initialization.
        """
        '''
        Removed as decided to make it compulsory to pass trained Single View models which means there's
        more transparency with the run as it 
        if self.cc_cnn is None:
            print("\n---   Initializing and Training CC SingleViewCNN Model ---")
            if num_epochs is None or patience is None:
                raise ValueError("num_epochs and patience must be provided to train individual CNNs.")
            # Assume self.train_dataset etc. here are the *single-view wrapped* datasets for CC
            self.cc_cnn = SingleViewCNN(
                train_dataset=self.train_dataset, # These must be the CC-specific datasets
                validation_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                view_type="CC"
            )
            _, _, cc_test_performance = self.cc_cnn.Train(num_epochs, patience, subdirectory, pos_weight=cc_pos_weight)
            print(f"\nCC Test Performance after training: {cc_test_performance}")
        else:
            print("\n---  Using Pre-trained CC SingleViewCNN Model ---")

        if self.mlo_cnn is None:
            print("\n---  Initializing and Training MLO SingleViewCNN Model ---")
            if num_epochs is None or patience is None:
                raise ValueError("num_epochs and patience must be provided to train individual CNNs.")
            # Assume self.train_dataset etc. here are the *single-view wrapped* datasets for MLO
            self.mlo_cnn = SingleViewCNN(
                train_dataset=self.train_dataset, # These must be the MLO-specific datasets
                validation_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                view_type="MLO"
            )
            _, _, mlo_test_performance = self.mlo_cnn.Train(num_epochs, patience, subdirectory, pos_weight=mlo_pos_weight)
            print(f"\nMLO Test Performance after training: {mlo_test_performance}")
        else:
            print("\n--- Using Pre-trained MLO SingleViewCNN Model ---")'''


        print("\n--- Extracting Logits and Training Logistic Regression ---")

        # For logit extraction, we need DataLoaders for the *single-view wrapped* training datasets
        # These data loaders need to be created from the specific datasets that the SingleViewCNN instances expect.
        # This requires the `train_dataset` etc. attributes of `LogisticRegressionFusionModel` to be the
        # *multi-view* datasets, and then we extract the specific single-view datasets from the `cc_cnn` and `mlo_cnn` objects.
        
        # Retrieve the specific single-view training datasets from the (provided) SingleViewCNN instances
        cc_train_ds_for_logits = self.cc_cnn.torch_dataset_train_view
        mlo_train_ds_for_logits = self.mlo_cnn.torch_dataset_train_view

        BATCH_SIZE = 32
        cc_train_loader_for_logits = DataLoader(cc_train_ds_for_logits, batch_size=BATCH_SIZE, shuffle=False)
        mlo_train_loader_for_logits = DataLoader(mlo_train_ds_for_logits, batch_size=BATCH_SIZE, shuffle=False)

        # Get logits from the trained CC & MLO model on the training data
        cc_train_logits, training_labels_for_lr = self.cc_cnn.get_trained_model_logits(
            cc_train_loader_for_logits, "CC Train Logit Extraction"
        )
        mlo_train_logits, _ = self.mlo_cnn.get_trained_model_logits(
            mlo_train_loader_for_logits, "MLO Train Logit Extraction"
        )

        if not np.array_equal(training_labels_for_lr, _): # Check labels match
            raise ValueError("Labels extracted from CC and MLO training sets do not match!")

        # Stack the logits from the CC and MLO views
        X_train_lr = np.hstack((cc_train_logits.reshape(-1, 1), mlo_train_logits.reshape(-1, 1)))
        y_train_lr = training_labels_for_lr

        # C is the Inverse of regularization strength, has been set to the default
        # liblinear is used and works for binary classification
        self.logistic_regression_model = LogisticRegression(solver='liblinear', C=1.0)
        self.logistic_regression_model.fit(X_train_lr, y_train_lr)

        print("\nLogistic Regression Model Training Complete.")
        print(f"Logistic Regression Coefficients: {self.logistic_regression_model.coef_}")
        print(f"Logistic Regression Intercept: {self.logistic_regression_model.intercept_}")

    def evaluate_fusion_model(self, dataset_type: Literal["validation", "test"]):
        """
        Evaluates the combined Logistic Regression model on the specified dataset.
        """
        if self.logistic_regression_model is None:
            raise RuntimeError("Logistic Regression model has not been trained yet. Call train_fusion_pipeline first.")

        # Determine datasets and dataloaders to use for logit extraction
        if dataset_type == "validation":
            cc_eval_ds = self.cc_cnn.torch_dataset_val_view
            mlo_eval_ds = self.mlo_cnn.torch_dataset_val_view
            dataset_name = "Validation"
        elif dataset_type == "test":
            cc_eval_ds = self.cc_cnn.torch_dataset_test_view
            mlo_eval_ds = self.mlo_cnn.torch_dataset_test_view
            dataset_name = "Test"
        else:
            raise ValueError("dataset_type must be 'validation' or 'test'.")
        BATCH_SIZE = 32
        cc_eval_loader_for_logits = DataLoader(cc_eval_ds, batch_size=BATCH_SIZE, shuffle=False)
        mlo_eval_loader_for_logits = DataLoader(mlo_eval_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\n--- Evaluating Full Fusion Model on {dataset_name} Set ---")
        # Get the logits for each view and the labels (from CC view which is the same as MLO view)
        cc_eval_logits, evaluation_labels = self.cc_cnn.get_trained_model_logits(
            cc_eval_loader_for_logits, f"CC {dataset_name} Logit Extraction"
        )
        mlo_eval_logits, _ = self.mlo_cnn.get_trained_model_logits(
            mlo_eval_loader_for_logits, f"MLO {dataset_name} Logit Extraction"
        )

        if not np.array_equal(evaluation_labels, _):
            raise ValueError(f"Labels extracted from CC and MLO {dataset_name} sets do not match!")

        X_eval_lr = np.hstack((cc_eval_logits.reshape(-1, 1), mlo_eval_logits.reshape(-1, 1)))
        y_eval = evaluation_labels

        lr_predictions = self.logistic_regression_model.predict(X_eval_lr)
        lr_probabilities = self.logistic_regression_model.predict_proba(X_eval_lr)[:, 1]

        accuracy = accuracy_score(y_eval, lr_predictions)
        roc_auc = roc_auc_score(y_eval, lr_probabilities)
        cm = confusion_matrix(y_eval, lr_predictions)

        print(f"Fusion Model {dataset_name} Accuracy: {accuracy:.4f}")
        print(f"Fusion Model {dataset_name} ROC AUC: {roc_auc:.4f}")
        print(f"Fusion Model {dataset_name} Confusion Matrix:\n{cm}")

        show_confusion_matrix(cm, f"Fusion Model ({dataset_name} Set)")
        tn, fp, fn, tp = cm.ravel()
        performance_results = get_performance_metrics(tp, fp, tn, fn, f"Fusion Model {dataset_name}")

        fpr, tpr, _ = roc_curve(y_eval, lr_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - Fusion Model ({dataset_name} Set)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        return accuracy, roc_auc, cm, performance_results