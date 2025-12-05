"""
Lab 13: Support Vector Machines (SVM)
EuroSAT Land Cover Classification

Author: Machine Learning Lab
Date: December 5, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import load_img, img_to_array
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("Lab 13: Support Vector Machines (SVM)")
print("EuroSAT Land Cover Classification")
print("="*70)

# ============================================================================
# Exercise 1: Load EuroSAT & Extract Features
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 1: Load EuroSAT & Extract Features")
print("="*70)

# NOTE: Update this path to your EuroSAT dataset location
# Download from: https://github.com/phelber/EuroSAT
DATASET_PATH = "EuroSAT_RGB"  # Update this path

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n⚠️  WARNING: Dataset not found at {DATASET_PATH}")
    print("Creating synthetic data for demonstration purposes...")
    
    # Create synthetic data mimicking EuroSAT features
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']
    
    X = []
    y = []
    
    # Generate synthetic RGB mean values for each class
    class_profiles = {
        'AnnualCrop': (120, 140, 80),      # Greenish-yellow
        'Forest': (30, 140, 80),            # Dark green
        'HerbaceousVegetation': (90, 150, 70),  # Light green
        'Highway': (120, 120, 120),         # Gray
        'Industrial': (115, 100, 80),       # Brown-gray
        'Pasture': (80, 160, 90),           # Bright green
        'PermanentCrop': (110, 130, 75),    # Medium green
        'Residential': (130, 110, 90),      # Tan
        'River': (60, 80, 140),             # Blue
        'SeaLake': (12, 50, 160)            # Dark blue
    }
    
    # Generate 100 samples per class with variation
    for label, cls in enumerate(classes):
        base_r, base_g, base_b = class_profiles[cls]
        for _ in range(100):
            # Add random variation
            mean_r = base_r + np.random.randn() * 15
            mean_g = base_g + np.random.randn() * 15
            mean_b = base_b + np.random.randn() * 15
            
            X.append([mean_r, mean_g, mean_b])
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Generated synthetic dataset")
    print(f"  Total samples: {len(X)}")
    print(f"  Classes: {len(classes)}")
    
else:
    # Load actual EuroSAT dataset
    classes = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    print(f"\n✓ Found EuroSAT dataset")
    print(f"  Classes: {classes}")
    
    X = []
    y = []
    
    print("\nExtracting features from images...")
    for label, cls in enumerate(classes):
        folder = os.path.join(DATASET_PATH, cls)
        images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        
        # Limit to 100 images per class for speed
        for img_name in images[:100]:
            try:
                img = load_img(os.path.join(folder, img_name), target_size=(64, 64))
                arr = img_to_array(img)
                
                # Extract mean RGB values
                mean_r = arr[:,:,0].mean()
                mean_g = arr[:,:,1].mean()
                mean_b = arr[:,:,2].mean()
                
                X.append([mean_r, mean_g, mean_b])
                y.append(label)
            except Exception as e:
                print(f"  Error loading {img_name}: {e}")
                continue
        
        print(f"  {cls}: {len([i for i in y if i == label])} samples")
    
    X = np.array(X)
    y = np.array(y)

# Create class name mapping
class_names = classes if 'classes' in locals() else [f"Class_{i}" for i in range(len(np.unique(y)))]

print(f"\n📊 Dataset Summary:")
print(f"  Feature shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Number of classes: {len(np.unique(y))}")
print(f"  Features: Mean R, Mean G, Mean B")

# Display sample data
print(f"\n📝 Sample data (first 5 samples):")
df_sample = pd.DataFrame(X[:5], columns=['Mean R', 'Mean G', 'Mean B'])
df_sample['Class'] = [class_names[y[i]] for i in range(5)]
print(df_sample.to_string(index=False))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n🔀 Train-Test Split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features standardized using StandardScaler")

# ============================================================================
# Exercise 2: Linear SVM on EuroSAT Features
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 2: Linear SVM on EuroSAT Features")
print("="*70)

# Train Linear SVM
print("\n🔧 Training Linear SVM...")
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Predict
y_pred_linear = svm_linear.predict(X_test_scaled)

# Evaluate
acc_linear = accuracy_score(y_test, y_pred_linear)
print(f"\n✓ Linear SVM trained successfully")
print(f"\n📈 Linear SVM Performance:")
print(f"  Accuracy: {acc_linear*100:.2f}%")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_linear, 
                          target_names=class_names, 
                          zero_division=0))

print(f"\n🔢 Confusion Matrix:")
cm_linear = confusion_matrix(y_test, y_pred_linear)
print(cm_linear)

# ============================================================================
# Exercise 3: Kernel SVM (RBF) for Non-linear Patterns
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 3: Kernel SVM (RBF) for Non-linear Patterns")
print("="*70)

# Train RBF Kernel SVM
print("\n🔧 Training RBF Kernel SVM...")
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Predict
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# Evaluate
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"\n✓ RBF SVM trained successfully")
print(f"\n📈 RBF SVM Performance:")
print(f"  Accuracy: {acc_rbf*100:.2f}%")
print(f"  Parameters: C=10, gamma=0.01")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_rbf, 
                          target_names=class_names,
                          zero_division=0))

print(f"\n🔢 Confusion Matrix:")
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
print(cm_rbf)

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"\n{'Model':<20} {'Accuracy':<15}")
print(f"{'-'*35}")
print(f"{'Linear SVM':<20} {acc_linear*100:>6.2f}%")
print(f"{'RBF Kernel SVM':<20} {acc_rbf*100:>6.2f}%")
print(f"\n{'Better Model:':<20} {'RBF Kernel SVM' if acc_rbf > acc_linear else 'Linear SVM'}")

# ============================================================================
# Exercise 4: Visualizing Feature Space
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 4: Visualizing Feature Space")
print("="*70)

# Plot RGB means for two land classes
class1 = 0  # First class
class2 = 1  # Second class

# Get indices for these classes
idx1 = np.where(y == class1)[0]
idx2 = np.where(y == class2)[0]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('SVM Feature Space Visualization', fontsize=16, fontweight='bold')

# Plot 1: Mean R vs Mean G for two classes
ax1 = axes[0, 0]
ax1.scatter(X[idx1, 0], X[idx1, 1], c='red', alpha=0.6, s=50, 
           label=class_names[class1], edgecolors='k')
ax1.scatter(X[idx2, 0], X[idx2, 1], c='blue', alpha=0.6, s=50, 
           label=class_names[class2], edgecolors='k')
ax1.set_xlabel('Mean R', fontsize=11)
ax1.set_ylabel('Mean G', fontsize=11)
ax1.set_title('Red vs Green Channel', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Mean R vs Mean B for two classes
ax2 = axes[0, 1]
ax2.scatter(X[idx1, 0], X[idx1, 2], c='red', alpha=0.6, s=50, 
           label=class_names[class1], edgecolors='k')
ax2.scatter(X[idx2, 0], X[idx2, 2], c='blue', alpha=0.6, s=50, 
           label=class_names[class2], edgecolors='k')
ax2.set_xlabel('Mean R', fontsize=11)
ax2.set_ylabel('Mean B', fontsize=11)
ax2.set_title('Red vs Blue Channel', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: All classes in 2D (R vs G)
ax3 = axes[1, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
for i, cls_name in enumerate(class_names):
    idx = np.where(y == i)[0]
    ax3.scatter(X[idx, 0], X[idx, 1], c=[colors[i]], alpha=0.6, s=30, 
               label=cls_name, edgecolors='k', linewidth=0.5)
ax3.set_xlabel('Mean R', fontsize=11)
ax3.set_ylabel('Mean G', fontsize=11)
ax3.set_title('All Classes: Red vs Green', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='best')
ax3.grid(True, alpha=0.3)

# Plot 4: Accuracy comparison
ax4 = axes[1, 1]
models = ['Linear SVM', 'RBF SVM']
accuracies = [acc_linear * 100, acc_rbf * 100]
bars = ax4.bar(models, accuracies, color=['steelblue', 'coral'], 
               edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('svm_visualization.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'svm_visualization.png'")
plt.show()

# ============================================================================
# Exercise 5: GIS Application Exercise — Land Cover Classification
# ============================================================================

print("\n" + "="*70)
print("EXERCISE 5: GIS Application - Land Cover Classification")
print("="*70)

# Create simplified GIS dataset
print("\n📊 Creating simplified land cover dataset...")

gis_data = pd.DataFrame({
    'Mean R': [110, 45, 30, 12, 115, 40, 35, 50, 120, 15],
    'Mean G': [95, 120, 140, 50, 100, 130, 145, 125, 105, 55],
    'Mean B': [70, 60, 80, 160, 80, 75, 85, 65, 75, 165],
    'Class': ['Urban', 'Agriculture', 'Forest', 'Water', 'Urban', 
              'Agriculture', 'Forest', 'Agriculture', 'Urban', 'Water']
})

print("\n📝 GIS Land Cover Dataset:")
print(gis_data.to_string(index=False))

# Encode classes
gis_classes = ['Agriculture', 'Forest', 'Urban', 'Water']
class_mapping = {cls: idx for idx, cls in enumerate(gis_classes)}
gis_data['Class_ID'] = gis_data['Class'].map(class_mapping)

X_gis = gis_data[['Mean R', 'Mean G', 'Mean B']].values
y_gis = gis_data['Class_ID'].values

# Scale features
X_gis_scaled = scaler.fit_transform(X_gis)

# Train SVM
print("\n🔧 Training RBF SVM on GIS data...")
svm_gis = SVC(kernel='rbf', C=5, gamma=0.02, random_state=42)
svm_gis.fit(X_gis_scaled, y_gis)

print(f"✓ GIS SVM model trained successfully")
print(f"  Parameters: kernel='rbf', C=5, gamma=0.02")

# Predict new pixel
new_pixel = np.array([[90, 110, 60]])
new_pixel_scaled = scaler.transform(new_pixel)
prediction = svm_gis.predict(new_pixel_scaled)
predicted_class = gis_classes[prediction[0]]

print(f"\n🎯 Prediction for new pixel [R=90, G=110, B=60]:")
print(f"  Predicted Class: {predicted_class}")

# Visualize GIS data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('GIS Land Cover Classification', fontsize=14, fontweight='bold')

# Plot 1: Training data
ax1 = axes[0]
gis_colors = {'Agriculture': 'green', 'Forest': 'darkgreen', 
              'Urban': 'red', 'Water': 'blue'}
for cls in gis_classes:
    mask = gis_data['Class'] == cls
    ax1.scatter(gis_data[mask]['Mean R'], gis_data[mask]['Mean G'], 
               c=gis_colors[cls], label=cls, s=100, alpha=0.7, 
               edgecolors='k', linewidth=1.5)

ax1.set_xlabel('Mean R', fontsize=11)
ax1.set_ylabel('Mean G', fontsize=11)
ax1.set_title('Training Data', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: With prediction
ax2 = axes[1]
for cls in gis_classes:
    mask = gis_data['Class'] == cls
    ax2.scatter(gis_data[mask]['Mean R'], gis_data[mask]['Mean G'], 
               c=gis_colors[cls], label=cls, s=100, alpha=0.7, 
               edgecolors='k', linewidth=1.5)

# Plot new prediction
ax2.scatter(new_pixel[0, 0], new_pixel[0, 1], 
           c='yellow', s=300, marker='*', 
           edgecolors='black', linewidth=2, 
           label=f'New Pixel ({predicted_class})', zorder=5)

ax2.set_xlabel('Mean R', fontsize=11)
ax2.set_ylabel('Mean G', fontsize=11)
ax2.set_title('With New Prediction', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gis_land_cover.png', dpi=300, bbox_inches='tight')
print(f"\n✓ GIS visualization saved as 'gis_land_cover.png'")
plt.show()

# ============================================================================
# LAB QUESTIONS
# ============================================================================

print("\n" + "="*70)
print("LAB QUESTIONS - ANSWERS")
print("="*70)

print("\n" + "-"*70)
print("Q1: Model Comparison")
print("-"*70)

print(f"\na) Linear SVM Accuracy: {acc_linear*100:.2f}%")
print(f"b) RBF Kernel SVM Accuracy: {acc_rbf*100:.2f}%")

print(f"\n💡 Analysis:")
if acc_rbf > acc_linear:
    print(f"   RBF Kernel SVM performs better by {(acc_rbf-acc_linear)*100:.2f}%")
    print(f"   This suggests the feature patterns (RGB means) have non-linear")
    print(f"   relationships that the RBF kernel can capture better than a")
    print(f"   linear boundary. Different land cover types may have complex")
    print(f"   spectral signatures that overlap in RGB space.")
else:
    print(f"   Linear SVM performs better by {(acc_linear-acc_rbf)*100:.2f}%")
    print(f"   This suggests the feature patterns (RGB means) are relatively")
    print(f"   linearly separable. The classes have distinct spectral signatures")
    print(f"   that can be separated with a linear hyperplane.")

print("\n" + "-"*70)
print("Q2: Using the Better-Performing Model")
print("-"*70)

better_model = svm_rbf if acc_rbf > acc_linear else svm_linear
better_name = "RBF Kernel SVM" if acc_rbf > acc_linear else "Linear SVM"

print(f"\nBetter Model: {better_name}")

print(f"\na) Conceptual Explanation of SVM Boundary:")
print(f"   The SVM creates decision boundaries in the 3D RGB feature space.")
print(f"   ")
print(f"   - Urban areas typically have high R, medium G, low B (tan/gray)")
print(f"   - Agriculture has medium R, high G, medium B (green)")
print(f"   - Water has low R, low G, high B (blue)")
print(f"   - Pasture has medium R, very high G, medium B (bright green)")
print(f"   ")
if better_name == "RBF Kernel SVM":
    print(f"   The RBF kernel creates non-linear, curved boundaries that can")
    print(f"   separate overlapping regions. It maps the 3D RGB space into a")
    print(f"   higher-dimensional space where classes become more separable.")
else:
    print(f"   The linear SVM creates flat hyperplanes that divide the 3D RGB")
    print(f"   space. Each boundary is a plane that separates one class from")
    print(f"   others using the one-vs-rest strategy.")

# Predict the specific pixel from Q2b
q2_pixel = np.array([[100, 95, 65]])
q2_pixel_scaled = scaler.transform(q2_pixel)
q2_prediction = better_model.predict(q2_pixel_scaled)
q2_class = class_names[q2_prediction[0]]

print(f"\nb) Prediction for pixel [R=100, G=95, B=65]:")
print(f"   Predicted Class: {q2_class}")
print(f"   ")
print(f"   Explanation:")
print(f"   This pixel has relatively high Red (100), medium-high Green (95),")
print(f"   and medium Blue (65) values. These values suggest a tan/brown color")
print(f"   typical of urban/residential areas or bare soil.")
print(f"   ")
print(f"   The SVM calculated the distance from this point to the decision")
print(f"   boundaries and determined it falls on the '{q2_class}' side.")
print(f"   The support vectors near this region defined the boundary that")
print(f"   classified this pixel.")

print(f"\nc) Effect of Gamma on Boundary Curvature:")
print(f"   Gamma (γ) controls the influence radius of support vectors.")
print(f"   ")
print(f"   🔹 High Gamma (e.g., γ = 1.0):")
print(f"      - Each support vector has small influence radius")
print(f"      - Creates tight, complex, wiggly boundaries")
print(f"      - Can fit training data very closely (risk of overfitting)")
print(f"      - In RGB space: creates small 'islands' around each class")
print(f"   ")
print(f"   🔹 Low Gamma (e.g., γ = 0.001):")
print(f"      - Support vectors have large influence radius")
print(f"      - Creates smooth, gentle boundaries")
print(f"      - Better generalization (smoother decision regions)")
print(f"      - In RGB space: creates broader separation zones")
print(f"   ")
print(f"   🔹 Current γ = 0.01:")
print(f"      - Balanced setting for EuroSAT RGB features")
print(f"      - Moderate complexity to capture spectral variations")
print(f"      - Allows some curvature without excessive overfitting")

print("\n" + "="*70)
print("LAB 13 COMPLETE!")
print("="*70)
print("\n✓ All exercises completed successfully")
print("✓ All questions answered")
print("✓ Visualizations saved")
print("\n📁 Output files:")
print("   - svm_visualization.png")
print("   - gis_land_cover.png")
print("\n" + "="*70)
