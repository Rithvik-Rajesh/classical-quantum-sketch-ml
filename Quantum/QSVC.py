import numpy as np
import pandas as pd
import time
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# -----------------------------------
# 1. Load and preprocess data
# -----------------------------------
def prepare_quantum_data(uri, db_name, collection_name):
    """
    Loads data from MongoDB, scales features, encodes labels.
    Returns scaled X, encoded y, scaler, and label encoder.
    """
    client = MongoClient(uri)
    collection = client[db_name][collection_name]
    data = list(collection.find({}, {"_id": 0}))  # exclude _id
    
    df = pd.DataFrame(data)
    y = df['label'].values
    X = df.drop(columns=['label']).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_scaled, y_encoded, scaler, label_encoder


# -----------------------------------
# 2. Encoding methods
# -----------------------------------
def encode_dataset(X, encoding_type):
    """
    Encodes a dataset into a list of quantum circuits.
    encoding_type: 'amplitude', 'rotation', 'basis'
    """
    circuits = []
    num_qubits = X.shape[1]
    
    for sample in X:
        qc = QuantumCircuit(num_qubits)
        
        if encoding_type == 'amplitude':
            norm = np.linalg.norm(sample)
            if norm > 0:
                normalized = sample / norm
            else:
                normalized = sample
            qc.initialize(normalized, range(num_qubits))
        
        elif encoding_type == 'rotation':
            for i, feature in enumerate(sample):
                qc.ry(feature, i)
        
        elif encoding_type == 'basis':
            binary = [int(x > 0) for x in sample]
            for i, bit in enumerate(binary):
                if bit == 1:
                    qc.x(i)
        
        circuits.append(qc)
    
    return circuits


# -----------------------------------
# 3. QSVC comparison pipeline
# -----------------------------------
def run_qsvc_comparison(uri="mongodb://localhost:27017/",
                        db_name="Doodle_Classifier",
                        collection_name="Filtered_Features"):
    X_scaled, y_encoded, scaler, label_encoder = prepare_quantum_data(uri, db_name, collection_name)
    
    encoding_types = ['amplitude', 'rotation', 'basis']
    results = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    backend = Aer.get_backend('statevector_simulator')
    
    for enc in encoding_types:
        print(f"\n=== Running QSVC with {enc.capitalize()} Encoding ===")
        
        train_circuits = encode_dataset(X_train, encoding_type=enc)
        test_circuits = encode_dataset(X_test, encoding_type=enc)
        
        avg_depth = np.mean([qc.depth() for qc in train_circuits])
        avg_qubits = np.mean([qc.num_qubits for qc in train_circuits])
        
        kernel = FidelityQuantumKernel(backend=backend)
        
        start_time = time.time()
        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(train_circuits, y_train)
        train_time = time.time() - start_time
        
        y_pred = qsvc.predict(test_circuits)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Encoding': enc,
            'Accuracy': acc,
            'Train Time (s)': train_time,
            'Avg Depth': avg_depth,
            'Avg Qubits': avg_qubits
        })
        
        print(f"Accuracy: {acc:.4f} | Time: {train_time:.2f}s | Depth: {avg_depth:.1f} | Qubits: {avg_qubits:.1f}")
    
    results_df = pd.DataFrame(results)
    print("\n=== Final Encoding Performance Comparison ===")
    print(results_df)
    return results_df


if __name__ == "__main__":
    run_qsvc_comparison()
