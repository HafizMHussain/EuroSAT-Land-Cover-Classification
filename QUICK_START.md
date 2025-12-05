# 🚀 Quick Start Guide - Lab 13: SVM

## Option 1: Run Complete Lab (Recommended for Quick Results)

### Single Command:
```powershell
python lab13_svm_complete.py
```

This runs all 5 exercises + answers all lab questions in one go!

**Output**:
- ✅ All exercises completed
- ✅ Model comparison (Linear vs RBF)
- ✅ Visualizations saved (2 PNG files)
- ✅ All questions answered
- ✅ Complete console output with analysis

**Time**: ~1-2 minutes

---

## Option 2: Step-by-Step Exercises (For Learning)

Run each exercise individually to understand the flow:

### Step 1: Load Data
```powershell
python exercise1_load_data.py
```
**Creates**: `X_train_scaled.npy`, `X_test_scaled.npy`, `y_train.npy`, `y_test.npy`

### Step 2: Linear SVM
```powershell
python exercise2_linear_svm.py
```
**Creates**: `linear_svm_predictions.npy`

### Step 3: RBF SVM
```powershell
python exercise3_rbf_svm.py
```
**Creates**: `rbf_svm_predictions.npy`  
**Shows**: Model comparison

### Step 4: Visualize
```powershell
python exercise4_visualization.py
```
**Creates**: `exercise4_visualization.png`

### Step 5: GIS Application
```powershell
python exercise5_gis_application.py
```
**Creates**: `exercise5_gis_classification.png`

---

## 📦 Installation

### Quick Install:
```powershell
pip install -r requirements.txt
```

### Or Manual Install:
```powershell
pip install numpy pandas scikit-learn matplotlib tensorflow keras pillow
```

---

## 📁 File Structure After Running

```
SVM LAB/
│
├── requirements.txt                    # Dependencies
├── README.md                          # Full documentation
├── QUICK_START.md                     # This file
├── LAB_QUESTIONS_ANSWERS.md          # Detailed Q&A
│
├── lab13_svm_complete.py             # ⭐ Complete lab (run this!)
│
├── exercise1_load_data.py            # Individual exercises
├── exercise2_linear_svm.py
├── exercise3_rbf_svm.py
├── exercise4_visualization.py
├── exercise5_gis_application.py
│
└── [Generated Files]
    ├── svm_visualization.png         # 4-panel visualization
    ├── gis_land_cover.png           # GIS classification
    ├── exercise4_visualization.png   # From exercise 4
    ├── exercise5_gis_classification.png
    ├── X_train_scaled.npy           # Training data
    ├── X_test_scaled.npy            # Test data
    ├── y_train.npy                  # Training labels
    ├── y_test.npy                   # Test labels
    ├── linear_svm_predictions.npy   # Linear SVM results
    └── rbf_svm_predictions.npy      # RBF SVM results
```

---

## ⚡ What Each File Does

| File | Purpose | Run Time |
|------|---------|----------|
| `lab13_svm_complete.py` | Everything in one script | 1-2 min |
| `exercise1_load_data.py` | Load & extract features | 10 sec |
| `exercise2_linear_svm.py` | Train Linear SVM | 5 sec |
| `exercise3_rbf_svm.py` | Train RBF SVM + compare | 10 sec |
| `exercise4_visualization.py` | Create visualizations | 5 sec |
| `exercise5_gis_application.py` | GIS land cover demo | 5 sec |

---

## 🎯 Expected Results

### Model Accuracies (Synthetic Data):
- **Linear SVM**: ~60-75%
- **RBF SVM**: ~70-85%

### Visual Outputs:
1. **Feature space plots** showing class distributions
2. **Model comparison** bar chart
3. **GIS predictions** with new pixel classification

### Console Output Includes:
- ✅ Dataset summary
- ✅ Training progress
- ✅ Accuracy scores
- ✅ Confusion matrices
- ✅ Classification reports
- ✅ Answers to all lab questions

---

## 🔍 Troubleshooting

### Problem: Import errors
```powershell
# Solution:
pip install --upgrade numpy pandas scikit-learn matplotlib tensorflow keras pillow
```

### Problem: "Dataset not found"
**Solution**: The script automatically generates synthetic data. No action needed!

### Problem: Plots don't show
**Solution**: Check if PNG files are saved in the folder. If Windows, use:
```powershell
explorer .
```

### Problem: Low accuracy
**Solution**: Normal with RGB features only! Real applications use more features.

---

## 💡 Quick Tips

1. **Use Complete Lab First**: Run `lab13_svm_complete.py` to see everything
2. **Read Console Output**: Contains detailed explanations
3. **Check Visualizations**: Open the PNG files for insights
4. **Review Questions**: Check `LAB_QUESTIONS_ANSWERS.md`
5. **Experiment**: Try changing C and gamma values

---

## 🎓 Learning Path

### Beginner:
1. Run `lab13_svm_complete.py`
2. Read console output carefully
3. View visualizations
4. Read `LAB_QUESTIONS_ANSWERS.md`

### Intermediate:
1. Run exercises step-by-step (1→2→3→4→5)
2. Modify parameters (C, gamma)
3. Try different kernels (poly, sigmoid)
4. Experiment with feature extraction

### Advanced:
1. Download real EuroSAT dataset
2. Extract more features (texture, spectral indices)
3. Implement grid search for parameter tuning
4. Add cross-validation
5. Try ensemble methods

---

## 📊 Key Parameters to Experiment With

### In `exercise3_rbf_svm.py` or `lab13_svm_complete.py`:

```python
# Original:
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01)

# Try:
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1)    # More complex
svm_rbf = SVC(kernel='rbf', C=100, gamma=0.001) # Smoother
svm_rbf = SVC(kernel='poly', degree=3)          # Polynomial
```

---

## ✅ Checklist

Before submitting your lab:

- [ ] Installed all dependencies
- [ ] Ran complete lab successfully
- [ ] Generated both PNG visualizations
- [ ] Reviewed accuracy scores
- [ ] Read model comparison
- [ ] Answered Q1 (model comparison)
- [ ] Answered Q2a (boundary explanation)
- [ ] Answered Q2b (pixel prediction)
- [ ] Answered Q2c (gamma effect)
- [ ] Understood key concepts

---

## 🎯 Main Learning Outcomes

After completing this lab, you should understand:

✅ **SVM Fundamentals**:
- Hyperplane and margin concepts
- Support vectors importance
- Kernel trick for non-linear separation

✅ **Practical Skills**:
- Feature extraction from images
- Training Linear and RBF SVM
- Model evaluation and comparison
- Parameter tuning (C and gamma)

✅ **GIS Application**:
- Land cover classification
- RGB feature analysis
- Prediction for new pixels

---

## 🆘 Need Help?

1. **Check README.md** for full documentation
2. **Read LAB_QUESTIONS_ANSWERS.md** for detailed explanations
3. **Review console output** for hints and explanations
4. **Examine visualizations** for insights

---

## 🚀 Ready to Start?

```powershell
# Install dependencies
pip install -r requirements.txt

# Run complete lab
python lab13_svm_complete.py

# Done! Check output and visualizations
```

**Good luck! 🎓**
