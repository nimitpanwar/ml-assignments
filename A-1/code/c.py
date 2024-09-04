import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Handle missing values (simple imputation with mean for this example)
data.fillna(data.mean(), inplace=True)

# Feature scaling methods
def min_max_scaling(features):
    return (features - features.min()) / (features.max() - features.min())

# Split the dataset into train, validation, and test sets (70:15:15)
def train_test_val_split(features, labels, train_size=0.7, val_size=0.15):
    total_size = len(features)
    train_end = int(train_size * total_size)
    val_end = int((train_size + val_size) * total_size)
    
    X_train = features[:train_end]
    y_train = labels[:train_end]
    X_val = features[train_end:val_end]
    y_val = labels[train_end:val_end]
    X_test = features[val_end:]
    y_test = labels[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Prepare data
features = data.drop(columns=['HeartDisease'])
labels = data['HeartDisease'].values

# Min-max scaling
features_min_max = min_max_scaling(features)
X_train_mm, y_train_mm, X_val_mm, y_val_mm, X_test_mm, y_test_mm = train_test_val_split(features_min_max.values, labels)

# Initialize weights and bias
def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function
def compute_loss(y_true, y_pred):
    m = len(y_true)
    loss = -1/m * (np.dot(y_true, np.log(y_pred)) + np.dot((1 - y_true), np.log(1 - y_pred)))
    return loss

# Gradient descent algorithm
def gradient_descent(X, y, X_val, y_val, weights, bias, learning_rate, iterations):
    m = len(y)
    loss_history = []
    val_loss_history = []
    
    for i in range(iterations):
        # Forward propagation
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute loss
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        
        # Compute gradients
        dw = 1/m * np.dot(X.T, (y_pred - y))
        db = 1/m * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Validation loss
        z_val = np.dot(X_val, weights) + bias
        y_val_pred = sigmoid(z_val)
        val_loss = compute_loss(y_val, y_val_pred)
        val_loss_history.append(val_loss)
    
    return weights, bias, loss_history, val_loss_history

# Training parameters
learning_rate = 0.01
iterations = 1000

# Train model with Min-max scaling
weights_mm, bias_mm = initialize_weights(X_train_mm.shape[1])
weights_mm, bias_mm, loss_history_mm, val_loss_history_mm = gradient_descent(
    X_train_mm, y_train_mm, X_val_mm, y_val_mm, weights_mm, bias_mm, learning_rate, iterations)

# Predict on the validation set
z_val = np.dot(X_val_mm, weights_mm) + bias_mm
y_val_pred_prob = sigmoid(z_val)
y_val_pred = [1 if i > 0.5 else 0 for i in y_val_pred_prob]

# Debugging: Print predictions and true values
print("Predicted probabilities:", y_val_pred_prob)
print("Predicted classes:", y_val_pred)
print("True classes:", y_val_mm)

# Confusion matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

conf_matrix = confusion_matrix(y_val_mm, y_val_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, Recall, F1 Score, ROC-AUC Score
def precision_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def roc_auc_score(y_true, y_pred_prob):
    thresholds = np.sort(y_pred_prob)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = [1 if i >= threshold else 0 for i in y_pred_prob]
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        tpr.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) != 0 else 0)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    return np.trapz(tpr, fpr)

precision = precision_score(y_val_mm, y_val_pred)
recall = recall_score(y_val_mm, y_val_pred)
f1 = f1_score(y_val_mm, y_val_pred)
roc_auc = roc_auc_score(y_val_mm, y_val_pred_prob)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_history_mm, label='Min-max Scaling - Training Loss')
plt.plot(val_loss_history_mm, label='Min-max Scaling - Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration (Min-max Scaling)')
plt.legend()

plt.tight_layout()
plt.show()

# Analysis
print("\nAnalysis:")
print("The confusion matrix provides a summary of the prediction results on the validation set.")
print("Precision indicates the proportion of positive identifications that were actually correct.")
print("Recall indicates the proportion of actual positives that were correctly identified.")
print("The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both.")
print("The ROC-AUC score measures the model's ability to distinguish between classes, with a higher score indicating better performance.")