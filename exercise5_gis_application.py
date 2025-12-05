"""
Exercise 5: GIS Application - Land Cover Classification

Task: Apply SVM to simplified GIS dataset for land cover prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

print("="*70)
print("EXERCISE 5: GIS Land Cover Classification")
print("="*70)

# TODO: Create GIS dataset
gis_data = pd.DataFrame({
    'Mean R': [110, 45, 30, 12, 115, 40, 35, 50, 120, 15],
    'Mean G': [95, 120, 140, 50, 100, 130, 145, 125, 105, 55],
    'Mean B': [70, 60, 80, 160, 80, 75, 85, 65, 75, 165],
    'Class': ['Urban', 'Agriculture', 'Forest', 'Water', 'Urban', 
              'Agriculture', 'Forest', 'Agriculture', 'Urban', 'Water']
})

print("\n📊 GIS Land Cover Dataset:")
print(gis_data.to_string(index=False))

# Encode classes
gis_classes = ['Agriculture', 'Forest', 'Urban', 'Water']
class_mapping = {cls: idx for idx, cls in enumerate(gis_classes)}
gis_data['Class_ID'] = gis_data['Class'].map(class_mapping)

# Prepare features and labels
X_gis = gis_data[['Mean R', 'Mean G', 'Mean B']].values
y_gis = gis_data['Class_ID'].values

# TODO: Scale features
scaler = StandardScaler()
X_gis_scaled = scaler.fit_transform(X_gis)

# TODO: Train RBF SVM
print("\n🔧 Training SVM on GIS data...")
svm_gis = SVC(kernel='rbf', C=5, gamma=0.02, random_state=42)
svm_gis.fit(X_gis_scaled, y_gis)
print("✓ Model trained successfully")
print("  Parameters: kernel='rbf', C=5, gamma=0.02")

# TODO: Predict for new pixel
new_pixel = np.array([[90, 110, 60]])
new_pixel_scaled = scaler.transform(new_pixel)
prediction = svm_gis.predict(new_pixel_scaled)
predicted_class = gis_classes[prediction[0]]

print(f"\n🎯 Prediction for new pixel [R=90, G=110, B=60]:")
print(f"   Predicted Class: {predicted_class}")

# Additional predictions
test_pixels = np.array([
    [100, 95, 65],   # Should be Urban (tan/brown)
    [40, 130, 70],   # Should be Agriculture (green)
    [25, 145, 85],   # Should be Forest (dark green)
    [10, 45, 170]    # Should be Water (blue)
])

print(f"\n🔍 Additional Test Predictions:")
print(f"\n{'Mean R':<10} {'Mean G':<10} {'Mean B':<10} {'Predicted Class':<20}")
print("-"*50)

for pixel in test_pixels:
    pixel_scaled = scaler.transform(pixel.reshape(1, -1))
    pred = svm_gis.predict(pixel_scaled)
    pred_class = gis_classes[pred[0]]
    print(f"{pixel[0]:<10.0f} {pixel[1]:<10.0f} {pixel[2]:<10.0f} {pred_class:<20}")

# TODO: Visualize
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

# Plot 2: With predictions
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
           label=f'New ({predicted_class})', zorder=5)

ax2.set_xlabel('Mean R', fontsize=11)
ax2.set_ylabel('Mean G', fontsize=11)
ax2.set_title('With New Prediction', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise5_gis_classification.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'exercise5_gis_classification.png'")
plt.show()

# Questions
print("\n" + "="*70)
print("QUESTIONS:")
print("="*70)
print("\nQ1: How does the SVM separate Urban vs Agriculture vs Water?")
print("A1: The SVM creates decision boundaries in 3D RGB space:")
print("    - Urban: High R, Medium G, Low B (tan/gray colors)")
print("    - Agriculture: Medium R, High G, Medium B (green colors)")
print("    - Water: Low R, Low G, High B (blue colors)")
print("    The RBF kernel creates curved boundaries to separate these regions.")

print("\nQ2: Why might the prediction for [90, 110, 60] be classified as it is?")
print(f"A2: Predicted as '{predicted_class}'.")
print("    R=90 (medium), G=110 (medium-high), B=60 (medium-low)")
print("    This RGB combination suggests a greenish-brown or olive color,")
print("    which could indicate vegetation (agriculture) or mixed urban/vegetation.")

print("\nQ3: What are limitations of using only RGB means?")
print("A3: - Spatial information is lost (texture, patterns)")
print("    - Spectral resolution is limited (only 3 bands)")
print("    - Similar colors can represent different land covers")
print("    - No consideration of context or neighboring pixels")
print("    - Better features: texture, spectral indices (NDVI), shape features")

print("\n" + "="*70)
print("✓ EXERCISE 5 COMPLETE")
print("="*70)
