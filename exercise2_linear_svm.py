"""
Exercise 2: Linear SVM on EuroSAT Features

Task: Train a Linear SVM classifier and evaluate its performance
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from Exercise 1
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print("="*70)
print("EXERCISE 2: Linear SVM")
print("="*70)

# TODO: Create Linear SVM classifier
# Hint: Use SVC with kernel='linear'
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)

# TODO: Train the model
print("\nTraining Linear SVM...")
svm_linear.fit(X_train_scaled, y_train)
print("✓ Training complete")

# TODO: Make predictions on test set
y_pred = svm_linear.predict(X_test_scaled)

# TODO: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Linear SVM Accuracy: {accuracy*100:.2f}%")

# TODO: Display confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# TODO: Display classification report
print(f"\n📊 Classification Report:")
classes = ['AnnualCrop', 'Forest', 'HerbaceousVeg', 'Highway', 
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
           'River', 'SeaLake']
print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

# Save results
np.save('linear_svm_predictions.npy', y_pred)
print(f"\n✓ Results saved")

# Questions:
print("\n" + "="*70)
print("QUESTIONS:")
print("="*70)
print("Q1: What is the accuracy of your Linear SVM?")
print(f"A1: {accuracy*100:.2f}%")
print("\nQ2: Which classes are most often confused?")
print("A2: Check the confusion matrix above")
print("\nQ3: Why is feature scaling important for SVM?")
print("A3: SVM uses distances between points. Without scaling, features")
print("    with larger ranges dominate the distance calculation.")
