"""
Exercise 3: RBF Kernel SVM

Task: Train RBF kernel SVM and compare with Linear SVM
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load linear SVM results for comparison
y_pred_linear = np.load('linear_svm_predictions.npy')
acc_linear = accuracy_score(y_test, y_pred_linear)

print("="*70)
print("EXERCISE 3: RBF Kernel SVM")
print("="*70)

# TODO: Create RBF SVM classifier
# Hint: Use kernel='rbf', C=10, gamma=0.01
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)

# TODO: Train the model
print("\nTraining RBF SVM...")
print("Parameters: C=10, gamma=0.01")
svm_rbf.fit(X_train_scaled, y_train)
print("✓ Training complete")

# TODO: Make predictions
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# TODO: Calculate accuracy
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"\n📈 RBF SVM Accuracy: {acc_rbf*100:.2f}%")

# Display confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rbf)
print(cm)

# Display classification report
print(f"\n📊 Classification Report:")
classes = ['AnnualCrop', 'Forest', 'HerbaceousVeg', 'Highway', 
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
           'River', 'SeaLake']
print(classification_report(y_test, y_pred_rbf, target_names=classes, zero_division=0))

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"\n{'Model':<20} {'Accuracy':<15}")
print("-"*35)
print(f"{'Linear SVM':<20} {acc_linear*100:>6.2f}%")
print(f"{'RBF Kernel SVM':<20} {acc_rbf*100:>6.2f}%")
print(f"\n{'Improvement:':<20} {(acc_rbf-acc_linear)*100:>+6.2f}%")

better = "RBF Kernel SVM" if acc_rbf > acc_linear else "Linear SVM"
print(f"{'Winner:':<20} {better}")

# Save results
np.save('rbf_svm_predictions.npy', y_pred_rbf)
print(f"\n✓ Results saved")

# Questions
print("\n" + "="*70)
print("QUESTIONS:")
print("="*70)
print("\nQ1: Which model performs better?")
print(f"A1: {better}")
print("\nQ2: Why does RBF kernel often outperform linear SVM?")
print("A2: RBF kernel can capture non-linear patterns by mapping data to")
print("    higher dimensions. Land cover RGB features often have non-linear")
print("    relationships (e.g., forests and pastures both green but different shades).")
print("\nQ3: What happens if you increase gamma to 1.0?")
print("A3: Higher gamma makes the decision boundary more complex and wiggly,")
print("    fitting training data more closely but risking overfitting.")
