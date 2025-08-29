import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize, ZZFeatureMap
from qiskit.circuit import ParameterVector
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

# -----------------------------
#   Data Loading & Preprocessing
# -----------------------------

def load_data_from_mongo(uri="mongodb://localhost:27017/", 
                        db_name="Doodle_Classifier", 
                        collection_name="Filtered_Features",
                        sample_limit=None):
    """Load preprocessed data from MongoDB with optional sampling"""
    try:
        client = MongoClient(uri)
        collection = client[db_name][collection_name]

        data = []
        cursor = collection.find()
        if sample_limit:
            cursor = cursor.limit(sample_limit)
            
        for doc in cursor:
            row = {
                'label': doc["label"],
                **doc["features"]
            }
            data.append(row)
        
        client.close()
        
        if not data:
            raise ValueError("No data found in the collection")
            
        df = pd.DataFrame(data)
        X = df.drop(columns=['label']).values.astype(float)
        y = df['label'].values
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y, df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def preprocess_data(X, y, pca_dim=None, encoding_type='amplitude'):
    """
    Enhanced preprocessing with encoding-specific optimizations
    
    Args:
        X: Feature matrix
        y: Labels
        pca_dim: PCA dimension reduction (None for no PCA)
        encoding_type: Type of encoding for preprocessing optimization
    """
    # Standardize features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if requested
    pca = None
    if pca_dim is not None and pca_dim < X_scaled.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_processed = pca.fit_transform(X_scaled)
        print(f"Applied PCA: {X_scaled.shape[1]} -> {pca_dim} dimensions")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_processed = X_scaled
    
    # Encoding-specific preprocessing
    if encoding_type == 'basis':
        # For basis encoding, we need features in [0,1] range
        basis_scaler = MinMaxScaler()
        X_processed = basis_scaler.fit_transform(X_processed)
        print("Applied MinMax scaling for basis encoding")
    else:
        basis_scaler = None
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Final feature shape: {X_processed.shape}")
    
    return X_processed, y_encoded, scaler, label_encoder, pca, basis_scaler

# -----------------------------
#   Optimized Quantum Encoding Functions
# -----------------------------

def create_amplitude_feature_map(num_features):
    """
    Create amplitude encoding feature map
    Conceptually better: encode features as quantum state amplitudes
    """
    # Calculate qubits needed for amplitude encoding
    num_qubits = int(np.ceil(np.log2(num_features)))
    target_length = 2 ** num_qubits
    
    def amplitude_map(features):
        features = np.array(features, dtype=float)
        
        # Normalize to unit vector for quantum amplitudes
        norm = np.linalg.norm(features)
        if norm == 0:
            raise ValueError("Feature vector has zero norm")
        state = features / norm
        
        # Pad to power of 2 length
        if len(state) < target_length:
            state = np.pad(state, (0, target_length - len(state)), 'constant')
        
        # Create circuit
        qc = QuantumCircuit(num_qubits)
        init_gate = Initialize(state)
        qc.append(init_gate, range(num_qubits))
        return qc
    
    return amplitude_map, num_qubits


def create_rotation_feature_map(num_features, reps=1, entanglement='full'):
    """
    Create rotation encoding feature map using ZZFeatureMap
    Conceptually better: hardware-efficient with entanglement
    """
    feature_map = ZZFeatureMap(
        feature_dimension=num_features, 
        reps=reps, 
        entanglement=entanglement,
        parameter_prefix='x'
    )
    
    return feature_map, feature_map.num_qubits


def create_basis_feature_map(num_features):
    """
    Create basis-like encoding using parameterized rotations
    Conceptually better: maps [0,1] features to |0⟩/|1⟩ basis states smoothly
    """
    params = ParameterVector("x", num_features)
    qc = QuantumCircuit(num_features)
    
    # Use RY rotations: RY(π*x) maps x=0 -> |0⟩, x=1 -> |1⟩
    for i in range(num_features):
        qc.ry(params[i] * np.pi, i)
    
    return qc, num_features


def create_hybrid_feature_map(num_features, amp_ratio=0.5):
    """
    Create hybrid encoding combining amplitude and rotation
    Conceptually better: leverages advantages of both methods
    """
    amp_features = int(num_features * amp_ratio)
    rot_features = num_features - amp_features
    
    # Amplitude part
    amp_qubits = int(np.ceil(np.log2(amp_features))) if amp_features > 0 else 0
    
    # Rotation part  
    rot_qubits = rot_features
    
    total_qubits = amp_qubits + rot_qubits
    
    def hybrid_map(features):
        features = np.array(features)
        qc = QuantumCircuit(total_qubits)
        
        if amp_features > 0:
            # Amplitude encoding for first part
            amp_part = features[:amp_features]
            norm = np.linalg.norm(amp_part)
            if norm > 0:
                amp_state = amp_part / norm
                target_length = 2 ** amp_qubits
                if len(amp_state) < target_length:
                    amp_state = np.pad(amp_state, (0, target_length - len(amp_state)), 'constant')
                
                init_gate = Initialize(amp_state)
                qc.append(init_gate, range(amp_qubits))
        
        # Rotation encoding for second part
        if rot_features > 0:
            rot_part = features[amp_features:]
            for i, angle in enumerate(rot_part):
                qc.ry(angle, amp_qubits + i)
        
        return qc
    
    return hybrid_map, total_qubits


# -----------------------------
#   Classical Kernel Computations
# -----------------------------

def compute_amplitude_kernel_optimized(X1, X2=None, eps=1e-12):
    """
    Optimized amplitude kernel computation
    Kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|² where |ψ(x)⟩ = x/||x||
    """
    def normalize_rows_safe(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms
    
    X1_norm = normalize_rows_safe(X1)
    if X2 is None:
        X2_norm = X1_norm
    else:
        X2_norm = normalize_rows_safe(X2)
    
    # Compute fidelity kernel: |⟨x|y⟩|²
    inner_products = np.dot(X1_norm, X2_norm.T)
    kernel_matrix = np.abs(inner_products) ** 2
    
    return kernel_matrix


# -----------------------------
#   Utility Functions
# -----------------------------

def get_encoding_info(encoding_type, num_features, **kwargs):
    """Get theoretical information about encoding requirements"""
    if encoding_type == 'amplitude':
        qubits = int(np.ceil(np.log2(num_features)))
        depth = "Variable (depends on Initialize gate decomposition)"
        
    elif encoding_type == 'rotation':
        reps = kwargs.get('reps', 1)
        feature_map = ZZFeatureMap(num_features, reps=reps)
        qubits = feature_map.num_qubits
        depth = feature_map.decompose().depth()
        
    elif encoding_type == 'basis':
        qubits = num_features
        depth = 1  # Single layer of RY gates
        
    elif encoding_type == 'hybrid':
        amp_ratio = kwargs.get('amp_ratio', 0.5)
        amp_features = int(num_features * amp_ratio)
        rot_features = num_features - amp_features
        amp_qubits = int(np.ceil(np.log2(amp_features))) if amp_features > 0 else 0
        qubits = amp_qubits + rot_features
        depth = "Variable (hybrid)"
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    return {
        'encoding_type': encoding_type,
        'num_features': num_features,
        'num_qubits': qubits,
        'circuit_depth': depth
    }


def compare_encodings(num_features):
    """Compare all encoding methods for given feature dimension"""
    encodings = ['amplitude', 'rotation', 'basis', 'hybrid']
    comparison = []
    
    for enc in encodings:
        info = get_encoding_info(enc, num_features)
        comparison.append(info)
    
    return pd.DataFrame(comparison)


# -----------------------------
#   Demo and Testing
# -----------------------------

def demo_all_encodings(sample_features, sample_label):
    """Demonstrate all encoding methods on a sample"""
    print("=" * 60)
    print("OPTIMIZED QUANTUM ENCODING DEMONSTRATIONS")
    print("=" * 60)
    
    num_features = len(sample_features)
    print(f"Sample features: {num_features} dimensions")
    print(f"Sample label: {sample_label}")
    print(f"Feature range: [{np.min(sample_features):.3f}, {np.max(sample_features):.3f}]")
    
    # Show encoding comparisons
    print("\nEncoding Comparison:")
    comparison_df = compare_encodings(num_features)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


# -----------------------------
#   Main Pipeline Function
# -----------------------------

def prepare_encoded_data(uri="mongodb://localhost:27017/",
                        db_name="Doodle_Classifier", 
                        collection_name="Filtered_Features",
                        encoding_type='amplitude',
                        pca_dim=6,
                        sample_limit=None):
    """
    Complete optimized pipeline for quantum encoding preparation
    
    Returns:
        dict: Contains all necessary data and encoders for quantum ML
    """
    # Load data
    X, y, df = load_data_from_mongo(uri, db_name, collection_name, sample_limit)
    if X is None:
        return None
    
    # Preprocess with encoding-specific optimizations
    X_processed, y_encoded, scaler, label_encoder, pca, basis_scaler = preprocess_data(
        X, y, pca_dim=pca_dim, encoding_type=encoding_type
    )
    
    # Get encoding information
    encoding_info = get_encoding_info(encoding_type, X_processed.shape[1])
    
    # Create feature map based on encoding type
    if encoding_type == 'amplitude':
        feature_map_func, num_qubits = create_amplitude_feature_map(X_processed.shape[1])
        feature_map = feature_map_func
    elif encoding_type == 'rotation':
        feature_map, num_qubits = create_rotation_feature_map(X_processed.shape[1])
    elif encoding_type == 'basis':
        feature_map, num_qubits = create_basis_feature_map(X_processed.shape[1])
    elif encoding_type == 'hybrid':
        feature_map, num_qubits = create_hybrid_feature_map(X_processed.shape[1])
    
    return {
        'X_processed': X_processed,
        'y_encoded': y_encoded,
        'feature_map': feature_map,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'pca': pca,
        'basis_scaler': basis_scaler,
        'encoding_info': encoding_info,
        'raw_data': (X, y, df)
    }


# -----------------------------
#   Testing
# -----------------------------

if __name__ == "__main__":
    # Test with small sample
    data_package = prepare_encoded_data(
        encoding_type='amplitude',
        pca_dim=6,
        sample_limit=100
    )
    
    if data_package:
        X_processed = data_package['X_processed']
        y_encoded = data_package['y_encoded']
        encoding_info = data_package['encoding_info']
        
        print("\nData Package Ready:")
        print(f"Processed shape: {X_processed.shape}")
        print(f"Encoding info: {encoding_info}")
        
        # Demo on first sample
        sample_features = X_processed[0]
        sample_label = y_encoded[0]
        
        demo_all_encodings(sample_features, sample_label)
        print("\nReady for quantum machine learning!")
    else:
        print("Failed to prepare data package")