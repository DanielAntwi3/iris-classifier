# Hands-On with a Simple Machine Learning Project
# Step 1: Preparing the Data

from sklearn.datasets import load_iris          # Import the iris dataset
iris = load_iris()                              # Load the dataset
X = iris.data                                   # shape (150, 4)
y = iris.target                                 # shape (150,)
print(iris.feature_names, iris.target_names)    # Print feature and target names

# Split into training and test sets: 
from sklearn.model_selection import train_test_split    # Import train_test_split for splitting the dataset
                                                        # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # 80% training and 20% testing

# Step 2: Choosing and Training a Model
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier for building the model
model = DecisionTreeClassifier(random_state=42) # Create a Decision Tree Classifier model

# Train (fit) the model: 
model.fit(X_train, y_train) # Train the model on the training data

# Step 3: Making Predictions
y_pred = model.predict(X_test)  # Predict the labels for the test set
print("Predictions:", y_pred[:5])   # Print the first 5 predictions
print("True labels:", y_test[:5])   # Print the first 5 true labels

# Step 4: Evaluating the Model
from sklearn.metrics import accuracy_score  # Import accuracy_score to evaluate the model
accuracy = accuracy_score(y_test, y_pred)   # Calculate the accuracy of the model
print("Accuracy:", accuracy)

# To illustrate iteration: 
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier for k-NN model
                                                    # Create a k-NN Classifier model
model2 = KNeighborsClassifier(n_neighbors=5)        # Train the k-NN model
model2.fit(X_train, y_train)                        # Train the k-NN model on the training data
y_pred2 = model2.predict(X_test)                    # Predict the labels for the test set using k-NN
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))    # Calculate and print the accuracy of the k-NN model

model = DecisionTreeClassifier(max_depth=3, random_state=42)    # Create a Decision Tree Classifier model with a maximum depth of 3 (Hyperparameter)
model.fit(X_train, y_train)                                     # Train the model on the training data


## Step 5: Generate and Save Confusion Matrix
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create predictions using the final model
y_pred = model.predict(X_test)

# Step 5: Generate and Save Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)  # Compute confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)  # Create display object

# Create output directory if it doesn't exist
output_dir = r"C:\Users\Administrator\OneDrive\Documents\GitHub\iris-classifier\iris-classifier\outputs"
os.makedirs(output_dir, exist_ok=True)

# Plot and save confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Decision Tree")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()


# Step 6: Save the trained model to outputs/trained_model_x.pkl

import joblib
import os


# Create output directory if it doesn't exist
output_dir = r"C:\Users\Administrator\OneDrive\Documents\GitHub\iris-classifier\iris-classifier\outputs"
os.makedirs(output_dir, exist_ok=True)


# Define the correct model path
model_path = os.path.join(output_dir, "trained_model_x.pkl")

# Save the trained Decision Tree model
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

