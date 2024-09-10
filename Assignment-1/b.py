# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split

# # # Load the dataset
# # df = pd.read_csv('heart_disease.csv')

# # # Handle missing values
# # df.fillna(df.mean(), inplace=True)

# # # Features and target
# # X = df.drop(columns=['HeartDisease'])
# # y = df['HeartDisease']

# # # Split the dataset into train, test, validation (70:15:15)
# # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # # Min-max scaling function
# # def min_max_scaling(X):
# #     return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# # # Sigmoid function with clipping
# # def sigmoid(z):
# #     z = np.clip(z, -500, 500)  # Clip to avoid overflow
# #     return 1 / (1 + np.exp(-z))

# # # Cross-entropy loss function with clipping
# # def compute_loss(y_true, y_pred):
# #     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Clip to avoid log(0)
# #     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # # Accuracy calculation
# # def calculate_accuracy(y_true, y_pred):
# #     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
# #     return np.mean(y_true == y_pred_classes)

# # # Gradient Descent for Logistic Regression
# # def gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=1000):
# #     m, n = X.shape
# #     weights = np.zeros(n)
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     for i in range(num_iters):
# #         # Forward pass
# #         linear_model = np.dot(X, weights) + bias
# #         y_pred = sigmoid(linear_model)
        
# #         # Compute train loss and accuracy
# #         train_loss = compute_loss(y, y_pred)
# #         train_accuracy = calculate_accuracy(y, y_pred)
        
# #         # Validation forward pass
# #         val_linear_model = np.dot(X_val, weights) + bias
# #         val_pred = sigmoid(val_linear_model)
        
# #         # Compute validation loss and accuracy
# #         val_loss = compute_loss(y_val, val_pred)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)
        
# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)
        
# #         # Gradients
# #         dw = (1 / m) * np.dot(X.T, (y_pred - y))
# #         db = (1 / m) * np.sum(y_pred - y)
        
# #         # Update weights and bias
# #         weights -= lr * dw
# #         bias -= lr * db

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# # # Run gradient descent
# # weights, bias, train_losses, val_losses, train_accuracies, val_accuracies = gradient_descent(
# #     X_train, y_train, X_val, y_val, lr=0.01, num_iters=1000
# # )

# # # Print final metrics
# # print("\nFinal Metrics:")
# # print(f"Final Training Loss: {train_losses[-1]:.4f}")
# # print(f"Final Validation Loss: {val_losses[-1]:.4f}")
# # print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
# # print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")

# # # Plotting
# # plt.figure(figsize=(12, 10))

# # plt.subplot(2, 2, 1)
# # plt.plot(train_losses, label='Training Loss')
# # plt.title("Training Loss vs Iteration")
# # plt.xlabel("Iterations")
# # plt.ylabel("Loss")

# # plt.subplot(2, 2, 2)
# # plt.plot(val_losses, label='Validation Loss', color='orange')
# # plt.title("Validation Loss vs Iteration")
# # plt.xlabel("Iterations")
# # plt.ylabel("Loss")

# # plt.subplot(2, 2, 3)
# # plt.plot(train_accuracies, label='Training Accuracy', color='green')
# # plt.title("Training Accuracy vs Iteration")
# # plt.xlabel("Iterations")
# # plt.ylabel("Accuracy")

# # plt.subplot(2, 2, 4)
# # plt.plot(val_accuracies, label='Validation Accuracy', color='red')
# # plt.title("Validation Accuracy vs Iteration")
# # plt.xlabel("Iterations")
# # plt.ylabel("Accuracy")

# # plt.tight_layout()
# # plt.legend()
# # plt.show()


# # B





# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split

# # # Load the dataset
# # df = pd.read_csv('heart_disease.csv')

# # # Handle missing values
# # df.fillna(df.mean(), inplace=True)

# # # Features and target
# # X = df.drop(columns=['HeartDisease'])
# # y = df['HeartDisease']

# # # Split the dataset into train, test, validation (70:15:15)
# # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # # Min-max scaling function
# # def min_max_scaling(X):
# #     return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# # # Sigmoid function with clipping
# # def sigmoid(z):
# #     z = np.clip(z, -500, 500)  # Clip to avoid overflow
# #     return 1 / (1 + np.exp(-z))

# # # Cross-entropy loss function with clipping
# # def compute_loss(y_true, y_pred):
# #     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Clip to avoid log(0)
# #     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # # Accuracy calculation
# # def calculate_accuracy(y_true, y_pred):
# #     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
# #     return np.mean(y_true == y_pred_classes)

# # # Gradient Descent for Logistic Regression
# # def gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=1000):
# #     m, n = X.shape
# #     weights = np.zeros(n)
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     for i in range(num_iters):
# #         # Forward pass
# #         linear_model = np.dot(X, weights) + bias
# #         y_pred = sigmoid(linear_model)
        
# #         # Compute train loss and accuracy
# #         train_loss = compute_loss(y, y_pred)
# #         train_accuracy = calculate_accuracy(y, y_pred)
        
# #         # Validation forward pass
# #         val_linear_model = np.dot(X_val, weights) + bias
# #         val_pred = sigmoid(val_linear_model)
        
# #         # Compute validation loss and accuracy
# #         val_loss = compute_loss(y_val, val_pred)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)
        
# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)
        
# #         # Gradients
# #         dw = (1 / m) * np.dot(X.T, (y_pred - y))
# #         db = (1 / m) * np.sum(y_pred - y)
        
# #         # Update weights and bias
# #         weights -= lr * dw
# #         bias -= lr * db

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# # # Run gradient descent with no scaling
# # weights, bias, train_losses_no_scaling, val_losses_no_scaling, train_accuracies_no_scaling, val_accuracies_no_scaling = gradient_descent(
# #     X_train, y_train, X_val, y_val, lr=0.01, num_iters=1000
# # )

# # # Run gradient descent with min-max scaling
# # X_train_scaled = min_max_scaling(X_train)
# # X_val_scaled = min_max_scaling(X_val)

# # weights, bias, train_losses_min_max_scaling, val_losses_min_max_scaling, train_accuracies_min_max_scaling, val_accuracies_min_max_scaling = gradient_descent(
# #     X_train_scaled, y_train, X_val_scaled, y_val, lr=0.01, num_iters=1000
# # )

# # # Print final metrics for no scaling
# # print("\nFinal Metrics (No Scaling):")
# # print(f"Final Training Loss: {train_losses_no_scaling[-1]:.4f}")
# # print(f"Final Validation Loss: {val_losses_no_scaling[-1]:.4f}")
# # print(f"Final Training Accuracy: {train_accuracies_no_scaling[-1]:.4f}")
# # print(f"Final Validation Accuracy: {val_accuracies_no_scaling[-1]:.4f}")

# # # Print final metrics for min-max scaling
# # print("\nFinal Metrics (Min-Max Scaling):")
# # print(f"Final Training Loss: {train_losses_min_max_scaling[-1]:.4f}")
# # print(f"Final Validation Loss: {val_losses_min_max_scaling[-1]:.4f}")
# # print(f"Final Training Accuracy: {train_accuracies_min_max_scaling[-1]:.4f}")
# # print(f"Final Validation Accuracy: {val_accuracies_min_max_scaling[-1]:.4f}")

# # # Plotting Loss vs Iteration for each scaling method
# # plt.figure(figsize=(12, 6))

# # plt.plot(train_losses_no_scaling, label='Training Loss (No Scaling)')
# # plt.plot(val_losses_no_scaling, label='Validation Loss (No Scaling)', color='orange')
# # plt.plot(train_losses_min_max_scaling, label='Training Loss (Min-Max Scaling)', linestyle='--')
# # plt.plot(val_losses_min_max_scaling, label='Validation Loss (Min-Max Scaling)', color='red', linestyle='--')

# # plt.title("Loss vs Iteration for Different Scaling Methods")
# # plt.xlabel("Iterations")
# # plt.ylabel("Loss")
# # plt.legend()
# # plt.show()

# # # Plotting Accuracy vs Iteration for each scaling method
# # plt.figure(figsize=(12, 6))

# # plt.plot(train_accuracies_no_scaling, label='Training Accuracy (No Scaling)')
# # plt.plot(val_accuracies_no_scaling, label='Validation Accuracy (No Scaling)', color='orange')
# # plt.plot(train_accuracies_min_max_scaling, label='Training Accuracy (Min-Max Scaling)', linestyle='--')
# # plt.plot(val_accuracies_min_max_scaling, label='Validation Accuracy (Min-Max Scaling)', color='red', linestyle='--')

# # plt.title("Accuracy vs Iteration for Different Scaling Methods")
# # plt.xlabel("Iterations")
# # plt.ylabel("Accuracy")
# # plt.legend()
# # plt.show()



# # C




# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

# # # Load the dataset
# # df = pd.read_csv('heart_disease.csv')

# # # Handle missing values
# # df.fillna(df.mean(), inplace=True)

# # # Features and target
# # X = df.drop(columns=['HeartDisease'])
# # y = df['HeartDisease']

# # # Split the dataset into train, test, validation (70:15:15)
# # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # # Min-max scaling function
# # def min_max_scaling(X):
# #     return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# # # Sigmoid function with clipping
# # def sigmoid(z):
# #     z = np.clip(z, -500, 500)  # Clip to avoid overflow
# #     return 1 / (1 + np.exp(-z))

# # # Cross-entropy loss function with clipping
# # def compute_loss(y_true, y_pred):
# #     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Clip to avoid log(0)
# #     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # # Accuracy calculation
# # def calculate_accuracy(y_true, y_pred):
# #     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
# #     return np.mean(y_true == y_pred_classes)

# # # Gradient Descent for Logistic Regression with Random Initialization
# # def gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=1000):
# #     m, n = X.shape
# #     weights = np.random.randn(n) * 0.01  # Small random initialization
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     for i in range(num_iters):
# #         # Forward pass
# #         linear_model = np.dot(X, weights) + bias
# #         y_pred = sigmoid(linear_model)
        
# #         # Compute train loss and accuracy
# #         train_loss = compute_loss(y, y_pred)
# #         train_accuracy = calculate_accuracy(y, y_pred)
        
# #         # Validation forward pass
# #         val_linear_model = np.dot(X_val, weights) + bias
# #         val_pred = sigmoid(val_linear_model)
        
# #         # Compute validation loss and accuracy
# #         val_loss = compute_loss(y_val, val_pred)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)
        
# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)
        
# #         # Gradients
# #         dw = (1 / m) * np.dot(X.T, (y_pred - y))
# #         db = (1 / m) * np.sum(y_pred - y)
        
# #         # Update weights and bias
# #         weights -= lr * dw
# #         bias -= lr * db

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies, val_pred

# # # Run gradient descent
# # weights, bias, train_losses, val_losses, train_accuracies, val_accuracies, val_preds = gradient_descent(
# #     X_train, y_train, X_val, y_val, lr=0.25, num_iters=1000
# # )


# # # Run gradient descent with min-max scaling
# # X_train_scaled = min_max_scaling(X_train)
# # X_val_scaled = min_max_scaling(X_val)


# # # Convert continuous predictions to binary
# # y_val_preds_binary = [1 if i > 0.5 else 0 for i in val_preds]

# # # Print some of the predictions and actual values for inspection
# # print("Sample of predictions and actual values:")
# # for i in range(10):
# #     print(f"Prediction: {y_val_preds_binary[i]}, Actual: {y_val.iloc[i]}")

# # # Check if the model is predicting only one class
# # unique_preds = np.unique(y_val_preds_binary)
# # print(f"Unique predictions: {unique_preds}")

# # # Calculate metrics
# # conf_matrix = confusion_matrix(y_val, y_val_preds_binary)
# # precision = precision_score(y_val, y_val_preds_binary, zero_division=1)
# # recall = recall_score(y_val, y_val_preds_binary, zero_division=1)
# # f1 = f1_score(y_val, y_val_preds_binary, zero_division=1)
# # roc_auc = roc_auc_score(y_val, val_preds)

# # # Print metrics
# # print("\nConfusion Matrix:")
# # print(conf_matrix)

# # print(f"\nPrecision: {precision:.4f}")
# # print(f"Recall: {recall:.4f}")
# # print(f"F1 Score: {f1:.4f}")
# # print(f"ROC-AUC Score: {roc_auc:.4f}")

# # # Plot Confusion Matrix
# # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['No Disease', 'Disease'])
# # disp.plot(cmap='Blues')
# # plt.title('Confusion Matrix')
# # plt.show()


# # D




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# # Load the dataset
# df = pd.read_csv('heart_disease.csv')

# # Handle missing values
# df.fillna(df.mean(), inplace=True)

# # Features and target
# X = df.drop(columns=['HeartDisease']).values  # Convert to numpy array
# y = df['HeartDisease'].values  # Convert to numpy array

# # Split the dataset into train, test, validation (70:15:15)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# def min_max_scaling(X):
#     return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# # Sigmoid function with clipping
# def sigmoid(z):
#     z = np.clip(z, -500, 500)
#     return 1 / (1 + np.exp(-z))

# # Cross-entropy loss function with clipping
# def compute_loss(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # Accuracy calculation
# def calculate_accuracy(y_true, y_pred):
#     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
#     return np.mean(y_true == y_pred_classes)

# # Mini-Batch Gradient Descent function
# def mini_batch_gradient_descent(X, y, X_val, y_val, batch_size, lr=0.01, num_iters=200):
#     m, n = X.shape
#     weights = np.zeros(n)
#     bias = 0
#     train_losses = []
#     val_losses = []
#     train_accuracies = []
#     val_accuracies = []

#     for i in range(num_iters):
#         # Shuffle data
#         indices = np.arange(m)
#         np.random.shuffle(indices)
#         X_shuffled = X[indices]
#         y_shuffled = y[indices]

#         # Mini-batch gradient descent
#         for start in range(0, m, batch_size):
#             end = min(start + batch_size, m)
#             X_batch = X_shuffled[start:end]
#             y_batch = y_shuffled[start:end]

#             # Forward pass
#             linear_model = np.dot(X_batch, weights) + bias
#             y_pred = sigmoid(linear_model)

#             # Compute loss
#             loss = compute_loss(y_batch, y_pred)

#             # Compute gradients
#             dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
#             db = (1 / len(y_batch)) * np.sum(y_pred - y_batch)

#             # Update weights and bias
#             weights -= lr * dw
#             bias -= lr * db

#         # Forward pass on the entire training set
#         train_pred = sigmoid(np.dot(X, weights) + bias)
#         val_pred = sigmoid(np.dot(X_val, weights) + bias)

#         # Compute metrics
#         train_loss = compute_loss(y, train_pred)
#         val_loss = compute_loss(y_val, val_pred)
#         train_accuracy = calculate_accuracy(y, train_pred)
#         val_accuracy = calculate_accuracy(y_val, val_pred)

#         # Record metrics
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_accuracies.append(train_accuracy)
#         val_accuracies.append(val_accuracy)

#     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# def stochastic_gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=200):
#     m, n = X.shape
#     weights = np.zeros(n)
#     bias = 0
#     train_losses = []
#     val_losses = []
#     train_accuracies = []
#     val_accuracies = []

#     for i in range(num_iters):
#         # Shuffle data
#         indices = np.arange(m)
#         np.random.shuffle(indices)
#         X_shuffled = X[indices]
#         y_shuffled = y[indices]

#         # SGD
#         for j in range(m):
#             X_i = X_shuffled[j:j+1]  # X_i has shape (1, n)
#             y_i = y_shuffled[j:j+1]  # y_i has shape (1,)

#             # Forward pass
#             linear_model = np.dot(X_i, weights) + bias
#             y_pred = sigmoid(linear_model)

#             # Compute loss
#             loss = compute_loss(y_i, y_pred)

#             # Compute gradients
#             dw = (y_pred - y_i) * X_i.flatten()  # Flatten X_i to match weights shape
#             db = (y_pred - y_i)

#             # Update weights and bias
#             weights -= lr * dw
#             bias -= lr * db

#         # Forward pass on the entire training set
#         train_pred = sigmoid(np.dot(X, weights) + bias)
#         val_pred = sigmoid(np.dot(X_val, weights) + bias)

#         # Compute metrics
#         train_loss = compute_loss(y, train_pred)
#         val_loss = compute_loss(y_val, val_pred)
#         train_accuracy = calculate_accuracy(y, train_pred)
#         val_accuracy = calculate_accuracy(y_val, val_pred)

#         # Record metrics
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_accuracies.append(train_accuracy)
#         val_accuracies.append(val_accuracy)

#     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies


# # Run mini-batch gradient descent with different batch sizes
# mini_batch_sizes = [4, 32, 64]
# mini_batch_results = {}

# for batch_size in mini_batch_sizes:
#     print(f"Running Mini-Batch Gradient Descent with batch size {batch_size}...")
#     weights_mb, bias_mb, train_losses_mb, val_losses_mb, train_accuracies_mb, val_accuracies_mb = mini_batch_gradient_descent(
#         X_train, y_train, X_val, y_val, batch_size=batch_size, lr=0.01, num_iters=200
#     )
#     mini_batch_results[batch_size] = (train_losses_mb, val_losses_mb, train_accuracies_mb, val_accuracies_mb)

# # Run stochastic gradient descent with different batch sizes (essentially just SGD here)
# sgd_batch_sizes = [4, 16, 32]
# sgd_results = {}

# for batch_size in sgd_batch_sizes:
#     print(f"Running Stochastic Gradient Descent with batch size {batch_size}...")
#     weights_sgd, bias_sgd, train_losses_sgd, val_losses_sgd, train_accuracies_sgd, val_accuracies_sgd = stochastic_gradient_descent(
#         X_train, y_train, X_val, y_val, lr=0.01, num_iters=200
#     )
#     sgd_results[batch_size] = (train_losses_sgd, val_losses_sgd, train_accuracies_sgd, val_accuracies_sgd)

# # Plotting Loss vs Iteration for each batch size
# plt.figure(figsize=(12, 12))

# # Mini-Batch Gradient Descent Plots
# plt.subplot(2, 2, 1)
# for batch_size in mini_batch_sizes:
#     plt.plot(mini_batch_results[batch_size][0], label=f'Mini-Batch {batch_size} Train Loss')
# plt.title('Mini-Batch Loss vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(2, 2, 2)
# for batch_size in mini_batch_sizes:
#     plt.plot(mini_batch_results[batch_size][1], label=f'Mini-Batch {batch_size} Validation Loss')
# plt.title('Mini-Batch Validation Loss vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(2, 2, 3)
# for batch_size in mini_batch_sizes:
#     plt.plot(mini_batch_results[batch_size][2], label=f'Mini-Batch {batch_size} Train Accuracy')
# plt.title('Mini-Batch Training Accuracy vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(2, 2, 4)
# for batch_size in mini_batch_sizes:
#     plt.plot(mini_batch_results[batch_size][3], label=f'Mini-Batch {batch_size} Validation Accuracy')
# plt.title('Mini-Batch Validation Accuracy vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Plotting Loss vs Iteration for Stochastic Gradient Descent
# plt.figure(figsize=(12, 12))

# plt.subplot(2, 2, 1)
# for batch_size in sgd_batch_sizes:
#     plt.plot(sgd_results[batch_size][0], label=f'SGD (Batch Size {batch_size}) Train Loss')
# plt.title('SGD Loss vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(2, 2, 2)
# for batch_size in sgd_batch_sizes:
#     plt.plot(sgd_results[batch_size][1], label=f'SGD (Batch Size {batch_size}) Validation Loss')
# plt.title('SGD Validation Loss vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(2, 2, 3)
# for batch_size in sgd_batch_sizes:
#     plt.plot(sgd_results[batch_size][2], label=f'SGD (Batch Size {batch_size}) Train Accuracy')
# plt.title('SGD Training Accuracy vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(2, 2, 4)
# for batch_size in sgd_batch_sizes:
#     plt.plot(sgd_results[batch_size][3], label=f'SGD (Batch Size {batch_size}) Validation Accuracy')
# plt.title('SGD Validation Accuracy vs Iteration')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()



# # E



# # import numpy as np
# # import pandas as pd
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# # from sklearn.utils import shuffle

# # # Sigmoid function with clipping
# # def sigmoid(z):
# #     z = np.clip(z, -500, 500)
# #     return 1 / (1 + np.exp(-z))

# # # Cross-entropy loss function with clipping
# # def compute_loss(y_true, y_pred):
# #     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
# #     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # # Accuracy calculation
# # def calculate_accuracy(y_true, y_pred):
# #     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
# #     return np.mean(y_true == y_pred_classes)

# # # Gradient Descent function
# # def gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=200):
# #     m, n = X.shape
# #     weights = np.zeros(n)
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     for i in range(num_iters):
# #         # Forward pass
# #         linear_model = np.dot(X, weights) + bias
# #         y_pred = sigmoid(linear_model)

# #         # Compute loss
# #         train_loss = compute_loss(y, y_pred)

# #         # Compute gradients
# #         dw = (1 / m) * np.dot(X.T, (y_pred - y))
# #         db = (1 / m) * np.sum(y_pred - y)

# #         # Update weights and bias
# #         weights -= lr * dw
# #         bias -= lr * db

# #         # Forward pass on validation set
# #         val_pred = sigmoid(np.dot(X_val, weights) + bias)

# #         # Compute metrics
# #         train_loss = compute_loss(y, y_pred)
# #         val_loss = compute_loss(y_val, val_pred)
# #         train_accuracy = calculate_accuracy(y, y_pred)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)

# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# # # Mini-batch Gradient Descent function
# # def mini_batch_gradient_descent(X, y, X_val, y_val, batch_size, lr=0.01, num_iters=200):
# #     m, n = X.shape
# #     weights = np.zeros(n)
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     for i in range(num_iters):
# #         # Shuffle data
# #         indices = np.arange(m)
# #         np.random.shuffle(indices)
# #         X_shuffled = X[indices]
# #         y_shuffled = y[indices]

# #         # Mini-batch gradient descent
# #         for start in range(0, m, batch_size):
# #             end = min(start + batch_size, m)
# #             X_batch = X_shuffled[start:end]
# #             y_batch = y_shuffled[start:end]

# #             # Forward pass
# #             linear_model = np.dot(X_batch, weights) + bias
# #             y_pred = sigmoid(linear_model)

# #             # Compute loss
# #             loss = compute_loss(y_batch, y_pred)
# #             train_loss = loss

# #             # Compute gradients
# #             dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
# #             db = (1 / len(y_batch)) * np.sum(y_pred - y_batch)

# #             # Update weights and bias
# #             weights -= lr * dw
# #             bias -= lr * db

# #         # Forward pass on the entire training set
# #         train_pred = sigmoid(np.dot(X, weights) + bias)
# #         val_pred = sigmoid(np.dot(X_val, weights) + bias)

# #         # Compute metrics
# #         train_loss = compute_loss(y, train_pred)
# #         val_loss = compute_loss(y_val, val_pred)
# #         train_accuracy = calculate_accuracy(y, train_pred)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)

# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# # # K-Fold Cross Validation function
# # def k_fold_cross_validation(X, y, k, lr=0.01, num_iters=200, batch_size=None):
# #     # Shuffle the dataset
# #     X, y = shuffle(X, y, random_state=42)
    
# #     # Split the data into k folds
# #     fold_size = len(X) // k
# #     accuracies = []
# #     precisions = []
# #     recalls = []
# #     f1_scores = []

# #     for fold in range(k):
# #         # Create training and validation sets
# #         val_start = fold * fold_size
# #         val_end = val_start + fold_size
        
# #         X_val = X[val_start:val_end]
# #         y_val = y[val_start:val_end]
# #         X_train = np.concatenate([X[:val_start], X[val_end:]])
# #         y_train = np.concatenate([y[:val_start], y[val_end:]])
        
# #         # Run gradient descent or mini-batch gradient descent
# #         if batch_size is None:
# #             weights, bias, _, _, train_accuracies, val_accuracies = gradient_descent(
# #                 X_train, y_train, X_val, y_val, lr=lr, num_iters=num_iters
# #             )
# #         else:
# #             weights, bias, _, _, train_accuracies, val_accuracies = mini_batch_gradient_descent(
# #                 X_train, y_train, X_val, y_val, batch_size=batch_size, lr=lr, num_iters=num_iters
# #             )

# #         # Make predictions
# #         y_pred = sigmoid(np.dot(X_val, weights) + bias)
# #         y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
        
# #         # Calculate metrics
# #         accuracy = accuracy_score(y_val, y_pred_classes)
# #         precision = precision_score(y_val, y_pred_classes, zero_division=0)
# #         recall = recall_score(y_val, y_pred_classes, zero_division=0)
# #         f1 = f1_score(y_val, y_pred_classes, zero_division=0)
        
# #         accuracies.append(accuracy)
# #         precisions.append(precision)
# #         recalls.append(recall)
# #         f1_scores.append(f1)
    
# #     # Calculate average and standard deviation
# #     avg_accuracy = np.mean(accuracies)
# #     std_accuracy = np.std(accuracies)
# #     avg_precision = np.mean(precisions)
# #     std_precision = np.std(precisions)
# #     avg_recall = np.mean(recalls)
# #     std_recall = np.std(recalls)
# #     avg_f1 = np.mean(f1_scores)
# #     std_f1 = np.std(f1_scores)
    
# #     return {
# #         'avg_accuracy': avg_accuracy,
# #         'std_accuracy': std_accuracy,
# #         'avg_precision': avg_precision,
# #         'std_precision': std_precision,
# #         'avg_recall': avg_recall,
# #         'std_recall': std_recall,
# #         'avg_f1': avg_f1,
# #         'std_f1': std_f1
# #     }

# # # Load and prepare the dataset
# # df = pd.read_csv('heart_disease.csv')
# # df.fillna(df.mean(), inplace=True)
# # X = df.drop(columns=['HeartDisease']).values
# # y = df['HeartDisease'].values

# # # Perform k-fold cross-validation
# # results = k_fold_cross_validation(X, y, k=5, lr=0.01, num_iters=200, batch_size=None)

# # # Print the results
# # print("K-Fold Cross-Validation Results:")
# # print(f"Average Accuracy: {results['avg_accuracy']:.4f} (Std: {results['std_accuracy']:.4f})")
# # print(f"Average Precision: {results['avg_precision']:.4f} (Std: {results['std_precision']:.4f})")
# # print(f"Average Recall: {results['avg_recall']:.4f} (Std: {results['std_recall']:.4f})")
# # print(f"Average F1 Score: {results['avg_f1']:.4f} (Std: {results['std_f1']:.4f})")



# # F




# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

# # # Load the dataset
# # df = pd.read_csv('heart_disease.csv')

# # # Handle missing values
# # df.fillna(df.mean(), inplace=True)

# # # Features and target
# # X = df.drop(columns=['HeartDisease'])
# # y = df['HeartDisease']

# # # Split the dataset into train, test, validation (70:15:15)
# # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # # Min-max scaling function
# # def min_max_scaling(X):
# #     return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# # # Sigmoid function with clipping
# # def sigmoid(z):
# #     z = np.clip(z, -500, 500)  # Clip to avoid overflow
# #     return 1 / (1 + np.exp(-z))

# # # Cross-entropy loss function with clipping and regularization
# # def compute_loss(y_true, y_pred, weights, l1_lambda=0, l2_lambda=0):
# #     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Clip to avoid log(0)
# #     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
# #     if l1_lambda > 0:
# #         loss += l1_lambda * np.sum(np.abs(weights))
# #     if l2_lambda > 0:
# #         loss += 0.5 * l2_lambda * np.sum(weights ** 2)
# #     return loss

# # # Accuracy calculation
# # def calculate_accuracy(y_true, y_pred):
# #     y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
# #     return np.mean(y_true == y_pred_classes)

# # # Gradient Descent with Early Stopping
# # def gradient_descent(X, y, X_val, y_val, lr=0.01, num_iters=1000, early_stopping_patience=10, l1_lambda=0, l2_lambda=0):
# #     m, n = X.shape
# #     weights = np.random.randn(n) * 0.01  # Small random initialization
# #     bias = 0
# #     train_losses = []
# #     val_losses = []
# #     train_accuracies = []
# #     val_accuracies = []

# #     best_val_loss = float('inf')
# #     patience_counter = 0

# #     for i in range(num_iters):
# #         # Forward pass
# #         linear_model = np.dot(X, weights) + bias
# #         y_pred = sigmoid(linear_model)
        
# #         # Compute train loss and accuracy
# #         train_loss = compute_loss(y, y_pred, weights, l1_lambda, l2_lambda)
# #         train_accuracy = calculate_accuracy(y, y_pred)
        
# #         # Validation forward pass
# #         val_linear_model = np.dot(X_val, weights) + bias
# #         val_pred = sigmoid(val_linear_model)
        
# #         # Compute validation loss and accuracy
# #         val_loss = compute_loss(y_val, val_pred, weights, l1_lambda, l2_lambda)
# #         val_accuracy = calculate_accuracy(y_val, val_pred)
        
# #         # Record metrics
# #         train_losses.append(train_loss)
# #         val_losses.append(val_loss)
# #         train_accuracies.append(train_accuracy)
# #         val_accuracies.append(val_accuracy)
        
# #         # Gradients
# #         dw = (1 / m) * np.dot(X.T, (y_pred - y))
# #         db = (1 / m) * np.sum(y_pred - y)
        
# #         # Update weights and bias
# #         weights -= lr * (dw + l2_lambda * weights)
# #         bias -= lr * db

# #         # Check for early stopping
# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             patience_counter = 0
# #         else:
# #             patience_counter += 1
# #             if patience_counter >= early_stopping_patience:
# #                 print(f"Early stopping at iteration {i}")
# #                 break

# #     return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies, val_pred

# # # Run gradient descent with early stopping and regularization
# # X_train_scaled = min_max_scaling(X_train)
# # X_val_scaled = min_max_scaling(X_val)

# # weights, bias, train_losses, val_losses, train_accuracies, val_accuracies, val_preds = gradient_descent(
# #     X_train_scaled, y_train, X_val_scaled, y_val, lr=0.01, num_iters=1000, early_stopping_patience=10, l1_lambda=0.01, l2_lambda=0.01
# # )

# # # Convert continuous predictions to binary
# # y_val_preds_binary = [1 if i > 0.5 else 0 for i in val_preds]

# # # Print some of the predictions and actual values for inspection
# # print("Sample of predictions and actual values:")
# # for i in range(10):
# #     print(f"Prediction: {y_val_preds_binary[i]}, Actual: {y_val.iloc[i]}")

# # # Check if the model is predicting only one class
# # unique_preds = np.unique(y_val_preds_binary)
# # print(f"Unique predictions: {unique_preds}")

# # # Calculate metrics
# # conf_matrix = confusion_matrix(y_val, y_val_preds_binary)
# # precision = precision_score(y_val, y_val_preds_binary, zero_division=1)
# # recall = recall_score(y_val, y_val_preds_binary, zero_division=1)
# # f1 = f1_score(y_val, y_val_preds_binary, zero_division=1)
# # roc_auc = roc_auc_score(y_val, val_preds)

# # # Print metrics
# # print("\nConfusion Matrix:")
# # print(conf_matrix)

# # print(f"\nPrecision: {precision:.4f}")
# # print(f"Recall: {recall:.4f}")
# # print(f"F1 Score: {f1:.4f}")
# # print(f"ROC-AUC Score: {roc_auc:.4f}")

# # # Plot Confusion Matrix
# # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['No Disease', 'Disease'])
# # disp.plot(cmap='Blues')
# # plt.title('Confusion Matrix')
# # plt.show()

# # # Plot Training and Validation Loss
# # plt.figure(figsize=(12, 5))
# # plt.plot(train_losses, label='Training Loss')
# # plt.plot(val_losses, label='Validation Loss')
# # plt.xlabel('Iterations')
# # plt.ylabel('Loss')
# # plt.title('Training and Validation Loss vs Iteration')
# # plt.legend()
# # plt.show()

# # # Plot Training and Validation Accuracy
# # plt.figure(figsize=(12, 5))
# # plt.plot(train_accuracies, label='Training Accuracy')
# # plt.plot(val_accuracies, label='Validation Accuracy')
# # plt.xlabel('Iterations')
# # plt.ylabel('Accuracy')
# # plt.title('Training and Validation Accuracy vs Iteration')
# # plt.legend()
# # plt.show()
