# Classical vs. Quantum Machine Learning for Doodle Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.43+-purple.svg)](https://qiskit.org/)

## 🎯 Project Overview

This project implements a comprehensive comparison between **classical and
quantum machine learning approaches** for **hand-drawn sketch (doodle)
classification**. The goal is to evaluate the performance, efficiency, and
generalization capabilities of both paradigms on the same feature set extracted
from stroke-based drawings.

### 🔬 Research Question

_Can quantum machine learning algorithms provide advantages over classical
approaches for sketch recognition tasks when using identical handcrafted feature
representations?_

## 📊 Dataset Description

### Raw Data Format

Each data sample is stored as a JSON file with the following structure:

```json
{
  "session": 1663053145814,
  "student": "Radu", 
  "drawings": {
    "car": [
      [ [x1, y1], [x2, y2], ... ],  // stroke 1
      [ [x1, y1], [x2, y2], ... ],  // stroke 2
      // ... more strokes
    ]
  }
}
```

### Processed Data Formats

#### 1. **vectorized_data.json** - Flattened Coordinates

```json
[
  {
    "id": 1663053145814,
    "label": "car",
    "vector": [x1, y1, x2, y2, ..., xn, yn]
  }
]
```

#### 2. **stroke_vectors.json** - Stroke-Aware Representation

```json
[
  {
    "id": 1663053145814,
    "label": "car", 
    "sa_vector": [ [x, y, stroke_id], ... ],
    "no_strokes": 3,
    "no_points": 45
  }
]
```

#### 3. **feature_vectors.json** - Handcrafted Features

```json
[
  {
    "id": ...,
    "label": ...,
    "features": {
      "no_strokes": 3,
      "avg_stroke_length": 45.2,
      "bbox_width": 120.5,
      // ... more features
    }
  }
]
```

## 🧮 Feature Engineering

### Extracted Features (19 total)

The following handcrafted features are computed for each doodle:

| Feature Category         | Features                                                           | Description                       |
| ------------------------ | ------------------------------------------------------------------ | --------------------------------- |
| **Stroke Properties**    | `no_strokes`, `no_points`, `avg_stroke_length`                     | Basic stroke statistics           |
| **Geometric Properties** | `bbox_width`, `bbox_height`, `aspect_ratio`                        | Bounding box characteristics      |
| **Spatial Features**     | `centroid_x`, `centroid_y`, `start_x`, `start_y`, `end_x`, `end_y` | Position information              |
| **Shape Complexity**     | `compactness`, `convex_hull_area`, `area`, `perimeter`             | Shape complexity metrics          |
| **Symmetry**             | `horizontal_symmetry`, `vertical_symmetry`                         | Symmetry measures                 |
| **Stroke Quality**       | `straightness`                                                     | Ratio of Euclidean to path length |

### Feature Computation Pipeline

1. **Stroke Analysis**: Extract individual strokes from coordinate sequences
2. **Geometric Calculations**: Compute bounding boxes, centroids, areas
3. **Symmetry Detection**: Analyze horizontal and vertical symmetry
4. **Quality Metrics**: Calculate compactness, straightness ratios
5. **Normalization**: StandardScaler for feature scaling

## 🏗️ Project Structure

```
QC_Project/
├── README.md                          # This file
├── LICENSE                           # MIT License
├── practise.py                       # Experimentation notebook
├── clustering_outlier_removal.py     # Data cleaning pipeline
│
├── data/                             # Raw JSON doodle files
│   ├── 1663053145814.json
│   ├── 1663307917621.json
│   └── ... (100+ doodle files)
│
├── Classical/                        # Classical ML Pipeline
│   ├── data_formating.py            # Data preprocessing & vectorization
│   ├── Feature_Extraction.py        # Handcrafted feature computation
│   ├── ML.ipynb                     # Classical ML experiments
│   ├── model_results.csv            # Performance metrics
│   ├── Image_data/                  # Stroke visualizations
│   │   ├── car/
│   │   ├── house/
│   │   └── ... (class folders)
│   └── processed_data/              # Intermediate data files
│       ├── vectorized_data.json
│       ├── stroke_vectors.json
│       ├── feature_vectors.json
│       └── clean_features.csv
│
├── Quantum/                         # Quantum ML Pipeline
│   └── encodings.py                # Quantum encoding methods
│
├── Dataset_Collection/              # Data collection utilities
├── Lab_Class/                      # Lab experiments
└── Papers/                         # Research references
```

## 🔧 Implementation Details

### Classical ML Pipeline

#### 1. Data Preprocessing (`Classical/data_formating.py`)

- **JSON Loading**: Batch processing of doodle files
- **Vectorization**: Convert strokes to coordinate vectors
- **Stroke-Aware Processing**: Maintain stroke identity information
- **Image Generation**: Create PNG visualizations of doodles

#### 2. Feature Extraction (`Classical/Feature_Extraction.py`)

- **MongoDB Integration**: Store/retrieve processed features
- **Comprehensive Logging**: Track feature extraction progress
- **Error Handling**: Robust processing of malformed data
- **Geometric Computations**: Advanced shape analysis algorithms

#### 3. Machine Learning (`Classical/ML.ipynb`)

- **Data Preprocessing**: StandardScaler normalization, label encoding
- **Model Training**: KNN, Random Forest, SVM, Neural Networks
- **Evaluation**: Cross-validation, confusion matrices, classification reports
- **Visualization**: PCA, t-SNE dimensionality reduction

#### 4. Data Cleaning (`clustering_outlier_removal.py`)

- **Outlier Detection**: Isolation Forest, Local Outlier Factor, Statistical
  methods
- **Class-wise Clustering**: Understand intra-class data structure
- **Quality Assessment**: Silhouette scores, cluster analysis
- **Automated Filtering**: Remove problematic samples

### Quantum ML Pipeline

#### 1. Quantum Encoding (`Quantum/encodings.py`)

- **Amplitude Encoding**: Normalize feature vectors as quantum amplitudes
- **Data Preprocessing**: StandardScaler + LabelEncoder compatibility
- **MongoDB Integration**: Load filtered features for quantum processing
- **Qubit Calculation**: Automatic qubit requirement computation

#### 2. Planned Quantum Algorithms

- **Quantum KNN (QKNN)**: Distance-based classification in Hilbert space
- **Variational Quantum Classifier (VQC)**: Parameterized quantum circuits
- **Quantum Support Vector Machine (QSVM)**: Kernel methods in quantum space

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
pip install pymongo pillow scipy

# Quantum computing
pip install qiskit qiskit-aer qiskit-machine-learning

# Optional: For advanced visualizations
pip install plotly dash
```

### Database Setup

```bash
# Install and start MongoDB
brew install mongodb-community
brew services start mongodb-community

# Create database and collections
mongosh
> use Doodle_Classifier
> db.createCollection("Extracted_Features")
> db.createCollection("Filtered_Features")
```

### Running the Pipeline

#### 1. Data Preprocessing

```python
# Process raw JSON files into structured formats
python Classical/data_formating.py
```

#### 2. Feature Extraction

```python
# Extract handcrafted features and store in MongoDB
python Classical/Feature_Extraction.py
```

#### 3. Data Cleaning

```python
# Remove outliers and prepare clean dataset
python clustering_outlier_removal.py
```

#### 4. Classical ML Training

```python
# Open and run the Jupyter notebook
jupyter notebook Classical/ML.ipynb
```

#### 5. Quantum ML Preparation

```python
# Prepare quantum encodings
python Quantum/encodings.py
```

## 📈 Results and Analysis

### Performance Metrics

- **Accuracy**: Classification accuracy on test set
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Cross-Validation**: Robust performance estimation

### Comparison Framework

- **Same Feature Set**: Both classical and quantum models use identical features
- **Consistent Preprocessing**: Same normalization and encoding pipeline
- **Fair Evaluation**: Identical train/test splits and evaluation metrics
- **Computational Complexity**: Training time and resource usage comparison

### Expected Outcomes

- **Classical Baseline**: Establish strong classical performance benchmarks
- **Quantum Advantage**: Identify scenarios where quantum algorithms excel
- **Feature Importance**: Understand which features benefit most from quantum
  processing
- **Scalability Analysis**: Performance trends with dataset size and feature
  dimensionality

## 🔮 Future Work

### Immediate Next Steps

1. **Complete Quantum Implementation**: Finish QKNN and VQC algorithms
2. **Hyperparameter Optimization**: Grid search for both classical and quantum
   models
3. **Feature Selection**: Identify optimal feature subsets for quantum advantage
4. **Noise Analysis**: Study robustness to quantum hardware noise

### Advanced Directions

1. **Hybrid Classical-Quantum Models**: Combine both paradigms
2. **Real Quantum Hardware**: Experiments on IBM Quantum, IonQ platforms
3. **Scalability Studies**: Performance with larger datasets and feature spaces
4. **Novel Quantum Features**: Quantum-native feature extraction methods

## 📚 References

### Key Papers

- _Quantum Machine Learning_ - Biamonte et al. (2017)
- _Quantum algorithms for supervised and unsupervised machine learning_ - Lloyd
  et al. (2014)
- _The quest for a Quantum Neural Network_ - Kwek et al. (2021)

### Datasets and Benchmarks

- Quick, Draw! Dataset (Google)
- Custom Doodle Collection (This Project)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## 🙏 Acknowledgments

- **Quantum Computing Community**: For open-source quantum frameworks
- **Classical ML Libraries**: Scikit-learn, NumPy, Pandas ecosystems
- **Data Contributors**: Students who provided doodle samples
- **Research Inspiration**: Academic papers on quantum machine learning

## 📧 Contact

**Author**: Rithvik Rajesh\
**Repository**:
[classical-quantum-sketch-ml](https://github.com/Rithvik-Rajesh/classical-quantum-sketch-ml)\
**Issues**: Please use GitHub Issues for bug reports and feature requests

---

_This project represents an exploration into the frontier of quantum machine
learning, comparing traditional and quantum approaches on a practical computer
vision task. The goal is to contribute to our understanding of when and where
quantum algorithms may provide advantages in machine learning applications._
