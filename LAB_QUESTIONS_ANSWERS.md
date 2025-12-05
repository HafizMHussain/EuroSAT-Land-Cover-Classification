# Lab 13: SVM - Questions and Answers

## Lab Questions

### Q1: Model Training and Comparison

**Once Trained:**

#### a) Linear SVM
- **Task**: Train a Linear SVM on the EuroSAT RGB features
- **Expected Accuracy**: 40-70% (depending on dataset)
- **Characteristics**:
  - Creates linear (flat) decision boundaries
  - Faster training time
  - Simpler model with fewer parameters
  - Better for linearly separable data

#### b) RBF Kernel SVM
- **Task**: Train an RBF Kernel SVM (C=10, gamma=0.01)
- **Expected Accuracy**: 50-80% (typically higher than linear)
- **Characteristics**:
  - Creates non-linear (curved) decision boundaries
  - Can capture complex patterns
  - More flexible but risk of overfitting
  - Better for non-linearly separable data

#### Compare Accuracy
**Analysis Framework**:

1. **If RBF > Linear** (Most Common):
   - The RGB feature space has non-linear patterns
   - Different land covers have overlapping spectral signatures
   - RBF kernel successfully maps data to higher dimensions
   - Example: Forest and Pasture both green but different patterns

2. **If Linear ≈ RBF**:
   - Data is relatively linearly separable
   - RGB means provide clear class separation
   - Adding complexity doesn't help

3. **If Linear > RBF**:
   - Possible overfitting in RBF
   - May need to adjust C or gamma parameters
   - Linear boundary sufficient for this feature space

**Feature Pattern Explanation**:
- **RGB Means** create a 3D feature space
- Each land cover type occupies a region:
  - **Urban**: High R, Medium G, Low B (tan/gray)
  - **Agriculture**: Medium R, High G, Medium B (green)
  - **Forest**: Low R, High G, Medium B (dark green)
  - **Water**: Low R, Low G, High B (blue)
  - **Highway**: Medium R, Medium G, Medium B (gray)

---

### Q2: Using the Better-Performing Model

#### a) Conceptual Explanation of SVM Boundary

**How SVM Separates Classes:**

1. **In 3D RGB Space**:
   ```
   3D Space: [Mean R, Mean G, Mean B]
   
   Urban       → [110-130, 90-110, 70-90]
   Agriculture → [40-120, 120-150, 60-80]
   Water       → [10-30, 40-60, 140-180]
   Pasture     → [70-90, 150-170, 80-100]
   ```

2. **SVM Decision Process**:
   - Finds support vectors (critical boundary points)
   - Maximizes margin between classes
   - Creates hyperplanes separating regions

3. **Linear SVM Boundary**:
   ```
   - Creates flat planes in 3D space
   - Each plane: w₁·R + w₂·G + w₃·B + b = 0
   - Separates one class vs all others
   - Works well when classes don't overlap
   ```

4. **RBF SVM Boundary**:
   ```
   - Creates curved, flexible boundaries
   - Maps to infinite-dimensional space
   - Can wrap around complex class shapes
   - Better for overlapping classes
   ```

**Specific Separations**:

- **Urban vs Agriculture**:
  - Urban: Higher Red, Lower Green
  - Agriculture: Lower Red, Higher Green
  - Boundary: A surface separating these regions

- **Agriculture vs Water**:
  - Agriculture: High Green, Low Blue
  - Water: Low Green, High Blue
  - Boundary: Separates green from blue regions

- **Water vs Pasture**:
  - Water: Very high Blue, low Green
  - Pasture: Very high Green, medium Blue
  - Boundary: Separates blue-dominant from green-dominant

- **Urban vs Pasture**:
  - Urban: High Red, Medium Green
  - Pasture: Medium Red, Very High Green
  - Boundary: Separates based on Green intensity

**Visual Analogy**:
```
Imagine RGB space as a cube:
- Each corner represents pure colors
- Land covers cluster in different regions
- SVM draws boundaries to separate clusters
- RBF boundaries are like flexible membranes
- Linear boundaries are like rigid planes
```

---

#### b) Predict Class for New Pixel

**Given**: Mean R = 100, Mean G = 95, Mean B = 65

**Step-by-Step Classification**:

1. **Feature Analysis**:
   ```
   R = 100 (high)
   G = 95  (medium-high)
   B = 65  (medium)
   
   Color interpretation: Tan/brown/beige
   ```

2. **Distance Calculation**:
   - SVM calculates distance to all decision boundaries
   - Determines which side of each boundary the point falls
   - Assigns to the class with maximum margin

3. **Expected Classification**: **Urban or Residential**

   **Reasoning**:
   - High Red value (100) → suggests built-up areas
   - Medium-high Green (95) → not pure vegetation
   - Medium Blue (65) → not water, not pure sky
   - RGB combination creates tan/brown color
   - Typical of:
     - Concrete (weathered)
     - Rooftops
     - Paved surfaces
     - Bare soil in urban areas

4. **Why Not Other Classes?**:
   
   - **Not Agriculture/Pasture**:
     - Green value (95) too low for healthy vegetation
     - Agriculture typically has Green > 120
   
   - **Not Forest**:
     - Red value (100) too high for dense forest
     - Forest typically has Red < 50
   
   - **Not Water**:
     - Blue value (65) far too low for water
     - Water typically has Blue > 140
   
   - **Could be Highway**:
     - Similar gray/tan color
     - But Red (100) slightly high for typical pavement

5. **SVM Decision Process**:
   ```
   1. Scale: [100, 95, 65] → [scaled values]
   2. Calculate kernel function with all support vectors
   3. Compute decision function for each class
   4. Return class with highest decision value
   5. Result: Urban/Residential (most likely)
   ```

---

#### c) Effect of Gamma on Boundary Curvature

**Gamma (γ) Parameter Explained:**

Gamma defines how far the influence of a single training example reaches.

---

**Mathematical Intuition**:

RBF Kernel: K(x, x') = exp(-γ ||x - x'||²)

- γ controls the exponential decay
- Higher γ → faster decay → narrower influence
- Lower γ → slower decay → wider influence

---

**Three Scenarios in EuroSAT RGB Space**:

### 1. **High Gamma (γ = 1.0 or higher)**

**Characteristics**:
```
- Very tight decision boundaries
- Each support vector influences only nearby points
- Creates complex, wiggly boundaries
- High model complexity
```

**In RGB Space**:
```
- Creates small "islands" around each training point
- Forest class: Multiple small regions in green space
- Urban class: Scattered pockets in red-brown space
- High variance, low bias
```

**Visualization**:
```
   G │ 
     │   ┌─┐     ┌──┐
     │  ┌┘ └┐   ┌┘  └┐    (Very wiggly)
     │ ┌┘   └┐ ┌┘    └┐
     │─┘     └─┘      └─
     └────────────────── R
```

**Consequences**:
- ✅ Fits training data very well (high train accuracy)
- ❌ Poor generalization (low test accuracy)
- ❌ Overfitting risk
- ❌ Sensitive to noise

**Example**:
```
Agriculture samples at:
- [45, 120, 60]
- [47, 125, 65]
- [43, 118, 58]

With γ=1.0: Creates 3 separate decision regions
```

---

### 2. **Low Gamma (γ = 0.001 or lower)**

**Characteristics**:
```
- Very smooth decision boundaries
- Each support vector influences distant points
- Creates broad, gentle boundaries
- Low model complexity
```

**In RGB Space**:
```
- Creates large, smooth regions
- Forest class: One large blob in green space
- Urban class: One large region in red-brown space
- Low variance, high bias
```

**Visualization**:
```
   G │ 
     │    ╭─────────╮
     │   ╱           ╲    (Very smooth)
     │  │             │
     │   ╲           ╱
     └────╰─────────╯── R
```

**Consequences**:
- ✅ Good generalization
- ✅ Smooth, stable boundaries
- ❌ May underfit complex patterns
- ❌ Cannot capture fine details

**Example**:
```
Water samples scattered:
- [12, 50, 160]
- [8, 45, 155]
- [15, 55, 165]

With γ=0.001: Creates one large water region
```

---

### 3. **Moderate Gamma (γ = 0.01, Current Setting)**

**Characteristics**:
```
- Balanced boundary complexity
- Reasonable influence radius
- Captures patterns without overfitting
- Optimal for many problems
```

**In RGB Space**:
```
- Creates connected regions with moderate curvature
- Adapts to natural class distributions
- Handles some overlap between classes
- Balanced variance-bias tradeoff
```

**Visualization**:
```
   G │ 
     │   ╭──╮  ╭────╮
     │  ╱    ╲╱      ╲    (Moderate curves)
     │ │              │
     │  ╲            ╱
     └───╰──────────╯─── R
```

**Consequences**:
- ✅ Good balance between fit and generalization
- ✅ Captures main patterns
- ✅ Robust to small noise
- ✅ Works well for EuroSAT RGB features

---

**Practical Impact in EuroSAT**:

| Gamma | Train Acc | Test Acc | Boundary | Use Case |
|-------|-----------|----------|----------|----------|
| 10.0  | 95%       | 60%      | Very complex | Research only |
| 1.0   | 90%       | 70%      | Complex | Detailed mapping |
| 0.1   | 85%       | 78%      | Moderate | General purpose |
| **0.01** | **80%** | **80%** | **Balanced** | **Recommended** |
| 0.001 | 70%       | 72%      | Smooth | Large-scale mapping |
| 0.0001| 60%       | 60%      | Very smooth | Coarse classification |

---

**How to Choose Gamma**:

1. **Start with default**: γ = 1/n_features = 1/3 ≈ 0.33
2. **Grid search**: Try [0.001, 0.01, 0.1, 1, 10]
3. **Cross-validation**: Use CV to find optimal value
4. **Consider data**:
   - High overlap → Lower gamma
   - Clear separation → Higher gamma

---

**Gamma's Effect Summary**:

```
High γ (tight boundaries):
  ↑ Model Complexity
  ↑ Training Accuracy
  ↓ Test Accuracy
  ↑ Overfitting Risk
  ↑ Sensitivity to Noise

Low γ (smooth boundaries):
  ↓ Model Complexity
  ↓ Training Accuracy
  ↑ Test Accuracy (up to a point)
  ↓ Overfitting Risk
  ↑ Robustness to Noise
```

---

## Additional Study Questions

### Q3: Why is StandardScaler important for SVM?
**Answer**: SVM calculates distances between points. Features with larger ranges (e.g., RGB values 0-255) dominate the distance calculation over smaller features. StandardScaler ensures all features have equal influence (mean=0, std=1).

### Q4: What happens with imbalanced classes?
**Answer**: SVM may bias toward majority classes. Solutions:
- Use `class_weight='balanced'` parameter
- Oversample minority classes (SMOTE)
- Undersample majority classes

### Q5: When to use Linear vs RBF kernel?
**Answer**:
- **Linear**: Large dataset, high-dimensional, linearly separable
- **RBF**: Smaller dataset, low-dimensional, non-linear patterns

### Q6: Can SVM handle more than 2 classes?
**Answer**: Yes, using:
- **One-vs-Rest**: Train n binary classifiers
- **One-vs-One**: Train n(n-1)/2 binary classifiers
Scikit-learn handles this automatically.

---

## Summary

✅ **Linear SVM**: Simple, fast, linear boundaries  
✅ **RBF SVM**: Complex, flexible, non-linear boundaries  
✅ **C parameter**: Controls margin strictness  
✅ **Gamma parameter**: Controls boundary curvature  
✅ **Feature scaling**: Essential for SVM performance  
✅ **RGB features**: Simple but limited for land cover classification  

**Best Practices**:
1. Always scale features
2. Start with RBF kernel
3. Use cross-validation for parameter tuning
4. Consider more features than just RGB means
5. Visualize decision boundaries when possible
