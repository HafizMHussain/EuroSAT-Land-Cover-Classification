"""
Exercise 1: Load EuroSAT & Extract Features

Task: Extract RGB mean values from EuroSAT dataset images
"""

import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TODO: Update this path to your EuroSAT dataset location
DATASET_PATH = "EuroSAT_RGB"

# Initialize empty lists
X = []
y = []

# Check if dataset exists
if os.path.exists(DATASET_PATH):
    classes = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # TODO: Complete the feature extraction loop
    for label, cls in enumerate(classes):
        folder = os.path.join(DATASET_PATH, cls)
        
        # Limit to 100 images per class for speed
        for img_name in os.listdir(folder)[:100]:
            # TODO: Load image
            img = load_img(os.path.join(folder, img_name), target_size=(64, 64))
            arr = img_to_array(img)
            
            # TODO: Extract mean RGB values
            mean_r = arr[:,:,0].mean()
            mean_g = arr[:,:,1].mean()
            mean_b = arr[:,:,2].mean()
            
            # TODO: Append to lists
            X.append([mean_r, mean_g, mean_b])
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset loaded:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
else:
    print(f"Dataset not found at {DATASET_PATH}")
    print("Creating synthetic data for practice...")
    
    # Synthetic data for practice
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']
    
    class_profiles = {
        'AnnualCrop': (120, 140, 80),
        'Forest': (30, 140, 80),
        'HerbaceousVegetation': (90, 150, 70),
        'Highway': (120, 120, 120),
        'Industrial': (115, 100, 80),
        'Pasture': (80, 160, 90),
        'PermanentCrop': (110, 130, 75),
        'Residential': (130, 110, 90),
        'River': (60, 80, 140),
        'SeaLake': (12, 50, 160)
    }
    
    for label, cls in enumerate(classes):
        base_r, base_g, base_b = class_profiles[cls]
        for _ in range(100):
            mean_r = base_r + np.random.randn() * 15
            mean_g = base_g + np.random.randn() * 15
            mean_b = base_b + np.random.randn() * 15
            
            X.append([mean_r, mean_g, mean_b])
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nSynthetic dataset created:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

# TODO: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain-Test Split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")

# TODO: Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled successfully")

# Save for next exercises
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"\n✓ Data saved for next exercises")
