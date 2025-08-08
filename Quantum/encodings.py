from pymongo import MongoClient
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
#   Load and preprocess data
# -----------------------------

def load_data_from_mongo(uri="mongodb://localhost:27017/", db_name="Doodle_Classifier", collection_name="Filtered_Features"):
    client = MongoClient(uri)
    collection = client[db_name][collection_name]

    X = []
    y = []

    for doc in collection.find():
        features = list(doc["features"].values())
        label = doc["label"]
        X.append(features)
        y.append(label)

    return np.array(X, dtype=float), np.array(y)


def preprocess_features(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled, y_encoded, scaler, label_encoder


# -----------------------------
#      Encoding Functions
# -----------------------------

def amplitude_encode(features):
    # Normalize the vector
    norm = np.linalg.norm(features)
    if norm == 0:
        raise ValueError("Feature vector has zero norm, cannot normalize.")
    state = features / norm

    # Number of qubits needed
    num_qubits = int(np.ceil(np.log2(len(state))))
    # Pad with zeros if not a power of 2
    if len(state) < 2 ** num_qubits:
        state = np.pad(state, (0, 2 ** num_qubits - len(state)), 'constant')

    qc = QuantumCircuit(num_qubits)
    init_gate = Initialize(state)
    qc.append(init_gate, range(num_qubits))
    qc.barrier()
    
    return qc


def basis_encode(integer, num_qubits):
    qc = QuantumCircuit(num_qubits)
    
    for i in range(num_qubits):
        if (integer >> i) & 1:
            qc.x(i)
            
    qc.barrier()
    return qc


def rotation_encode(features):
    qc = QuantumCircuit(len(features))
    
    for i, f in enumerate(features):
        qc.ry(f, i)  
        
    qc.barrier()
    return qc


# -----------------------------
# 3. Example usage
# -----------------------------
if __name__ == "__main__":
    # Load data
    X, y = load_data_from_mongo(
        uri="mongodb://localhost:27017/",
        db_name="Doodle_Classifier",
        collection_name="Filtered_Features"
    )

    # Preprocess
    X_scaled, y_encoded, scaler, label_encoder = preprocess_features(X, y)

    # Pick one sample for demo
    sample_features = X_scaled[0]
    sample_label = y_encoded[0]

    print("Sample label:", label_encoder.inverse_transform([sample_label])[0])

    # Amplitude Encoding
    amp_circuit = amplitude_encode(sample_features)
    print("Amplitude Encoding Circuit:\n", amp_circuit)

    # Basis Encoding (e.g., using label as integer)
    basis_circuit = basis_encode(sample_label, num_qubits=5)
    print("Basis Encoding Circuit:\n", basis_circuit)

    # Rotation Encoding
    rot_circuit = rotation_encode(sample_features)
    print("Rotation Encoding Circuit:\n", rot_circuit)
