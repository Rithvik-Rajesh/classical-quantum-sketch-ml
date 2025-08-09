import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# -----------------------------
#   Data Loading & Preprocessing
# -----------------------------

def load_data_from_mongo(uri="mongodb://localhost:27017/", db_name="Doodle_Classifier", collection_name="Filtered_Features"):
    """Load preprocessed data from MongoDB"""
    try:
        client = MongoClient(uri)
        collection = client[db_name][collection_name]

        data = []
        for doc in collection.find():
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
        
        return X, y, df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def preprocess_data(X, y):
    """Simple preprocessing: standardization and label encoding"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_scaled, y_encoded, scaler, label_encoder


# -----------------------------
#  Quantum Encoding Functions
# -----------------------------

def amplitude_encode(features, normalize=True):
    """
    Amplitude encoding: encode classical data as quantum amplitudes
    
    Args:
        features: Feature vector (1D array)
        normalize: Whether to normalize the feature vector
    
    Returns:
        QuantumCircuit: Quantum circuit with amplitude encoding
    """
    features = np.array(features, dtype=float)
    
    if normalize:
        norm = np.linalg.norm(features)
        if norm == 0:
            raise ValueError("Feature vector has zero norm, cannot normalize.")
        state = features / norm
    else:
        state = features
    
    # Ensure state is normalized for quantum amplitudes
    if not np.isclose(np.sum(np.abs(state)**2), 1.0, atol=1e-6):
        state = state / np.linalg.norm(state)
    
    # Number of qubits needed (must be power of 2)
    num_qubits = int(np.ceil(np.log2(len(state))))
    target_length = 2 ** num_qubits
    
    # Pad with zeros if needed
    if len(state) < target_length:
        state = np.pad(state, (0, target_length - len(state)), 'constant')
    
    qc = QuantumCircuit(num_qubits)
    
    try:
        init_gate = Initialize(state)
        qc.append(init_gate, range(num_qubits))
    except Exception as e:
        print(f"Error in amplitude encoding: {e}")
        return None
        
    return qc


def basis_encode(integer_value, num_qubits=None):
    """
    Basis encoding: encode integer as computational basis state
    
    Args:
        integer_value: Integer to encode
        num_qubits: Number of qubits 
    
    Returns:
        QuantumCircuit: Quantum circuit with basis encoding
    """
    integer_value = int(integer_value)
    
    if num_qubits is None:
        num_qubits = max(1, integer_value.bit_length())
    
    if integer_value >= 2**num_qubits:
        raise ValueError(f"Integer {integer_value} requires more than {num_qubits} qubits")
    
    qc = QuantumCircuit(num_qubits)
    
    # Apply X gates for each bit that is 1
    for i in range(num_qubits):
        if (integer_value >> i) & 1:
            qc.x(i)
    
    return qc


def rotation_encode(features, rotation_gate='ry', scale_factor=np.pi):
    """
    Rotation encoding: encode features as rotation angles
    
    Args:
        features: Feature vector
        rotation_gate: 'rx', 'ry', or 'rz'
        scale_factor: Scaling factor for rotations
    
    Returns:
        QuantumCircuit: Quantum circuit with rotation encoding
    """
    features = np.array(features, dtype=float)
    
    # Scale features to reasonable range for rotations
    if np.max(np.abs(features)) > 1:
        features = features / np.max(np.abs(features))
    
    angles = features * scale_factor
    
    qc = QuantumCircuit(len(features))
    
    for i, angle in enumerate(angles):
        if rotation_gate == 'rx':
            qc.rx(angle, i)
        elif rotation_gate == 'ry':
            qc.ry(angle, i)
        elif rotation_gate == 'rz':
            qc.rz(angle, i)
        else:
            raise ValueError("rotation_gate must be 'rx', 'ry', or 'rz'")
    
    return qc


def encode_dataset(X, y, encoding_type='amplitude', **kwargs):
    """
    Encode entire dataset using specified encoding method
    
    Args:
        X: Feature matrix
        y: Labels
        encoding_type: 'amplitude', 'basis', 'rotation'
        **kwargs: Additional arguments for encoding functions
    
    Returns:
        list: List of quantum circuits
    """
    circuits = []
    
    for i in range(len(X)):
        if encoding_type == 'amplitude':
            circuit = amplitude_encode(X[i], **kwargs)
        elif encoding_type == 'rotation':
            circuit = rotation_encode(X[i], **kwargs)
        elif encoding_type == 'basis':
            # For basis encoding, we'll encode the label
            circuit = basis_encode(y[i], **kwargs)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        circuits.append(circuit)
    
    return circuits

# -----------------------------
#      Utility Functions
# -----------------------------

def get_encoding_info(circuit):
    """Get information about the encoded circuit"""
    return {
        'num_qubits': circuit.num_qubits,
        'depth': circuit.depth(),
        'num_gates': len(circuit.data),
        'gate_types': list(set([gate[0].name for gate in circuit.data]))
    }


def prepare_quantum_data(uri="mongodb://localhost:27017/", 
                        db_name="Doodle_Classifier", 
                        collection_name="Filtered_Features",
                        encoding_type='amplitude'):
    """
    Complete pipeline: load data and prepare quantum circuits
    
    Args:
        uri: MongoDB URI
        db_name: Database name
        collection_name: Collection name
        encoding_type: Type of quantum encoding
    
    Returns:
        tuple: (circuits, labels, scaler, label_encoder, raw_data)
    """
    # Load data
    X, y, df = load_data_from_mongo(uri, db_name, collection_name)
    
    if X is None:
        return None, None, None, None, None
    
    # Preprocess
    X_scaled, y_encoded, scaler, label_encoder = preprocess_data(X, y)
    
    # Encode to quantum circuits
    circuits = encode_dataset(X_scaled, y_encoded, encoding_type)
    
    return circuits, y_encoded, scaler, label_encoder, (X_scaled, df)


# -----------------------------
#      Example Usage Function
# -----------------------------

def demo_encodings(sample_features, sample_label):
    """Demonstrate all encoding methods on a sample"""
    print("="*50)
    print("QUANTUM ENCODING DEMONSTRATIONS")
    print("="*50)
    
    print(f"Sample shape: {sample_features.shape}")
    print(f"Sample label: {sample_label}")
    
    # Amplitude Encoding
    print("\n1. Amplitude Encoding:")
    amp_circuit = amplitude_encode(sample_features)
    if amp_circuit:
        info = get_encoding_info(amp_circuit)
        print(f"   Qubits: {info['num_qubits']}, Depth: {info['depth']}, Gates: {info['num_gates']}")
    
    # Basis Encoding
    print("\n2. Basis Encoding:")
    basis_circuit = basis_encode(sample_label, num_qubits=5)
    info = get_encoding_info(basis_circuit)
    print(f"   Qubits: {info['num_qubits']}, Depth: {info['depth']}, Gates: {info['num_gates']}")
    
    # Rotation Encoding
    print("\n3. Rotation Encoding:")
    rot_circuit = rotation_encode(sample_features)
    info = get_encoding_info(rot_circuit)
    print(f"   Qubits: {info['num_qubits']}, Depth: {info['depth']}, Gates: {info['num_gates']}")
    
    return {
        'amplitude': amp_circuit,
        'basis': basis_circuit,
        'rotation': rot_circuit,
    }


# -----------------------------
#      Main (for testing)
# -----------------------------

if __name__ == "__main__":
    # Test the encoding functions
    circuits, y_encoded, scaler, label_encoder, (X_scaled, df) = prepare_quantum_data()
    
    if circuits is not None:
        print(f"Loaded {len(circuits)} samples")
        print(f"Classes: {label_encoder.classes_}")
        print(f"Feature dimensions: {X_scaled.shape[1]}")
        
        # Demo on first sample
        sample_features = X_scaled[0]
        sample_label = y_encoded[0]
        
        demo_circuits = demo_encodings(sample_features, sample_label)
        
        print(f"\nReady for quantum machine learning!")
        print(f"Use: from quantum_encoding import amplitude_encode, rotation_encode, basis_encode")
    else:
        print("Failed to load data")