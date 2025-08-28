import os
import keras
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from transformers import AutoTokenizer, TFAutoModel
import tensorflow_datasets as tfds
from keras.datasets import mnist 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, accuracy_score


# Helper function to show Performance Metrics
# displays: confusion matrix
# confusion matrix
# sensitivity and specificity
# classification report
# AUC scores
# ROC curve
# Precision-Recall Curve 
# Predicted Probabilities by Class
def show_performance_metrics(all_labels, all_probs):

    # Convert probabilities to binary predictions
    binary_preds = np.array(all_probs) > 0.5
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    labels = ['No Dyslexia', 'Dyslexia']

    # --- Confusion Matrix Plot ---
    print("\nConfusion Matrix:")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ---  Extract CM components and calculate metrics ---
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\nAdditional Metrics for Test Results:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # --- Classification Report ---
    f1 = f1_score(all_labels, binary_preds)
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=labels))

    # ---  AUC Scores ---
    roc_auc = roc_auc_score(all_labels, all_probs)  # Use raw probabilities
    print(f"ROC AUC Score (Prob-based): {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    print(f"PR AUC Score: {pr_auc:.4f}")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # ---  Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})", color='green')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # ---  Predicted Probabilities by Class ---
    probs_class_0 = np.sort(np.array(all_probs)[all_labels == 0])
    probs_class_1 = np.sort(np.array(all_probs)[all_labels == 1])

    plt.figure(figsize=(10, 6))
    plt.plot(probs_class_0, label='Class 0 (No Dyslexia)', color='blue')
    plt.plot(probs_class_1, label='Class 1 (Dyslexia)', color='orange')
    plt.axhline(0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel("Sample Index (sorted)")
    plt.ylabel("Predicted Probability")
    plt.title("Predicted Probabilities by True Class")
    plt.legend()
    plt.grid(True)
    plt.show()

IMG_HEIGHT, IMG_WIDTH = 224, 224



# Function to preprocess images
def preprocess_image(image_path, label):
    # Load image file
    img = tf.io.read_file(image_path)
    # Decode image 
    img = tf.image.decode_jpeg(img, channels=3) # Use decode_png for PNGs
    # Resize image
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Apply ResNet50 specific preprocessing
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

# Function for data augmentation (only for training data)
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # More augmentation could be added here, e.g., random zoom, rotation but not using augmentation 
    # currently but may in future work
    return image, label


def apply_augmentation(inputs, label):
    img = inputs['image_input']
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    inputs['image_input'] = img
    return inputs, label
