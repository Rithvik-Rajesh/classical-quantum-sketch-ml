import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler, StatevectorSampler

# Import our optimized encoding functions
from quantum_encodings import (
    prepare_encoded_data,
    compute_amplitude_kernel_optimized,
    get_encoding_info
)

# -------------------------
# Enhanced SVM Training
# -------------------------
def train_evaluate_svm(K_train, K_test, y_train, y_test, kernel_type="precomputed"):
    """Train and evaluate SVM with detailed metrics"""
    clf = SVC(kernel=kernel_type, C=1.0)
    
    # Training
    t0 = time.time()
    clf.fit(K_train, y_train)
    train_time = time.time() - t0
    
    # Prediction
    y_pred = clf.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional metrics
    unique_labels = np.unique(y_test)
    per_class_acc = {}
    for label in unique_labels:
        mask = (y_test == label)
        if np.sum(mask) > 0:
            per_class_acc[label] = accuracy_score(y_test[mask], y_pred[mask])
    
    return {
        'accuracy': accuracy,
        'train_time': train_time,
        'per_class_accuracy': per_class_acc,
        'num_support_vectors': clf.n_support_.sum() if hasattr(clf, 'n_support_') else None
    }


def setup_quantum_backend(method='statevector'):
    """Setup quantum backend with current Qiskit API"""
    try:
        if method == 'statevector':
            # Use StatevectorSampler for exact computation
            sampler = StatevectorSampler()
            backend = None  # StatevectorSampler doesn't need backend
            print("✓ Using StatevectorSampler for exact quantum computation")
        else:
            # Use AerSimulator with Sampler
            backend = AerSimulator()
            sampler = Sampler()
            print("✓ Using AerSimulator with shot-based computation")
        
        return sampler, backend
        
    except Exception as e:
        print(f"⚠ Warning: Could not setup quantum backend: {e}")
        print("Falling back to classical amplitude kernel only")
        return None, None


def compute_quantum_kernel(X_train, X_test, feature_map, sampler, encoding_name):
    """Compute quantum kernel using current Qiskit API"""
    print(f"Computing {encoding_name} quantum kernel...")
    
    try:
        if encoding_name == 'amplitude_classical':
            # Use our optimized classical computation
            t0 = time.time()
            K_train = compute_amplitude_kernel_optimized(X_train)
            K_test = compute_amplitude_kernel_optimized(X_test, X_train)
            compute_time = time.time() - t0
            
            # Get theoretical info for amplitude encoding
            info = get_encoding_info('amplitude', X_train.shape[1])
            
        else:
            if sampler is None:
                raise ValueError("Quantum backend not available")
                
            # Use Qiskit Machine Learning FidelityQuantumKernel with new API
            fidelity_kernel = FidelityQuantumKernel(
                feature_map=feature_map,
                sampler=sampler  # Use sampler instead of quantum_instance
            )
            
            t0 = time.time()
            K_train = fidelity_kernel.evaluate(X_train, X_train)
            K_test = fidelity_kernel.evaluate(X_test, X_train)
            compute_time = time.time() - t0
            
            info = {
                'num_qubits': feature_map.num_qubits,
                'circuit_depth': feature_map.decompose().depth() if hasattr(feature_map, 'decompose') else 'N/A'
            }
        
        print(f"Kernel computation completed in {compute_time:.2f}s")
        print(f"Kernel matrix shapes: train {K_train.shape}, test {K_test.shape}")
        return K_train, K_test, compute_time, info
        
    except Exception as e:
        print(f"Error computing {encoding_name} kernel: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# -------------------------
# Main Experiment Function
# -------------------------
def run_quantum_encoding_comparison(
    mongo_uri="mongodb://localhost:27017/",
    db_name="Doodle_Classifier",
    collection_name="Filtered_Features",
    pca_dim=6,
    test_size=0.2,
    random_state=42,
    sample_limit=None,
    encodings_to_test=['amplitude', 'rotation', 'basis'],
    quantum_method='statevector'
):
    """
    Complete quantum encoding comparison experiment with updated Qiskit API
    """
    print("=" * 80)
    print("QUANTUM ENCODING COMPARISON FOR DOODLE CLASSIFICATION")
    print("=" * 80)
    print(f"Qiskit version compatibility: Using new Sampler API")
    
    # Setup quantum backend
    sampler, backend = setup_quantum_backend(quantum_method)
    
    results = []
    
    # Test each encoding method
    for encoding_type in encodings_to_test:
        print(f"\n{'='*20} {encoding_type.upper()} ENCODING {'='*20}")
        
        try:
            # Prepare data for this encoding
            data_package = prepare_encoded_data(
                uri=mongo_uri,
                db_name=db_name,
                collection_name=collection_name,
                encoding_type=encoding_type,
                pca_dim=pca_dim,
                sample_limit=sample_limit
            )
            
            if data_package is None:
                print(f"❌ Failed to prepare data for {encoding_type}")
                continue
            
            X_processed = data_package['X_processed']
            y_encoded = data_package['y_encoded']
            feature_map = data_package['feature_map']
            label_encoder = data_package['label_encoder']
            encoding_info = data_package['encoding_info']
            
            print(f"✓ Data prepared: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
            print(f"✓ Classes: {label_encoder.classes_}")
            print(f"✓ Encoding info: {encoding_info['num_qubits']} qubits")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y_encoded
            )
            print(f"✓ Split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            
            # Compute kernel and train SVM
            if encoding_type == 'amplitude':
                # Always use classical computation for amplitude encoding (it's equivalent and faster)
                K_train, K_test, kernel_time, kernel_info = compute_quantum_kernel(
                    X_train, X_test, feature_map, sampler, 'amplitude_classical'
                )
            else:
                # Use quantum computation for other encodings
                if sampler is None:
                    print(f"⚠ Skipping {encoding_type} - quantum backend unavailable")
                    continue
                    
                K_train, K_test, kernel_time, kernel_info = compute_quantum_kernel(
                    X_train, X_test, feature_map, sampler, encoding_type
                )
            
            if K_train is None:
                print(f"❌ Kernel computation failed for {encoding_type}")
                continue
            
            # Validate kernel matrices
            if np.any(np.isnan(K_train)) or np.any(np.isnan(K_test)):
                print(f"❌ Invalid kernel matrices (contains NaN) for {encoding_type}")
                continue
            
            # Train and evaluate SVM
            print("🔄 Training SVM...")
            svm_results = train_evaluate_svm(K_train, K_test, y_train, y_test)
            
            # Compile results
            result = {
                'encoding_type': encoding_type,
                'accuracy': svm_results['accuracy'],
                'kernel_computation_time': kernel_time,
                'svm_training_time': svm_results['train_time'],
                'total_time': kernel_time + svm_results['train_time'],
                'num_qubits': kernel_info.get('num_qubits', encoding_info['num_qubits']),
                'circuit_depth': kernel_info.get('circuit_depth', encoding_info['circuit_depth']),
                'num_support_vectors': svm_results['num_support_vectors']
            }
            
            # Add per-class accuracies
            for class_idx, class_name in enumerate(label_encoder.classes_):
                if class_idx in svm_results['per_class_accuracy']:
                    result[f'acc_{class_name}'] = svm_results['per_class_accuracy'][class_idx]
            
            results.append(result)
            
            # Print summary for this encoding
            print(f"📊 Results Summary:")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Kernel time: {result['kernel_computation_time']:.2f}s")
            print(f"   Training time: {result['svm_training_time']:.2f}s")
            print(f"   Total time: {result['total_time']:.2f}s")
            print(f"   Qubits used: {result['num_qubits']}")
            print(f"   Support vectors: {result['num_support_vectors']}")
            
        except Exception as e:
            print(f"❌ Error processing {encoding_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    if results:
        print(f"\n{'='*80}")
        print("FINAL COMPARISON RESULTS")
        print(f"{'='*80}")
        
        df_results = pd.DataFrame(results)
        
        # Main comparison table
        comparison_cols = ['encoding_type', 'accuracy', 'total_time', 'num_qubits', 'circuit_depth']
        available_cols = [col for col in comparison_cols if col in df_results.columns]
        
        print("\n📋 Main Comparison:")
        print(df_results[available_cols].to_string(index=False, float_format='%.4f'))
        
        # Performance analysis
        print(f"\n🔍 Performance Analysis:")
        print(f"   Mean Accuracy: {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
        print(f"   Mean Total Time: {df_results['total_time'].mean():.2f}s ± {df_results['total_time'].std():.2f}s")
        
        # Best performing encoding
        best_accuracy_idx = df_results['accuracy'].idxmax()
        best_encoding = df_results.loc[best_accuracy_idx]
        print(f"\n🏆 Best Accuracy: {best_encoding['encoding_type']} ({best_encoding['accuracy']:.4f})")
        
        fastest_idx = df_results['total_time'].idxmin()
        fastest_encoding = df_results.loc[fastest_idx]
        print(f"⚡ Fastest: {fastest_encoding['encoding_type']} ({fastest_encoding['total_time']:.2f}s)")
        
        # Efficiency analysis
        df_results['accuracy_per_second'] = df_results['accuracy'] / df_results['total_time']
        best_efficiency_idx = df_results['accuracy_per_second'].idxmax()
        best_efficiency = df_results.loc[best_efficiency_idx]
        print(f"⚖️ Most Efficient: {best_efficiency['encoding_type']} ({best_efficiency['accuracy_per_second']:.4f} acc/s)")
        
        return df_results
    else:
        print("❌ No successful encoding results")
        return None


# -------------------------
# Run Experiment
# -------------------------
if __name__ == "__main__":
    # Check available imports
    try:
        from qiskit import __version__ as qiskit_version
        print(f"🔧 Qiskit version: {qiskit_version}")
    except:
        print("🔧 Qiskit version: Unknown")
    
    try:
        from qiskit_machine_learning import __version__ as qml_version
        print(f"🔧 Qiskit Machine Learning version: {qml_version}")
    except:
        print("🔧 Qiskit Machine Learning version: Unknown")
    
    # Run the complete comparison
    results_df = run_quantum_encoding_comparison(
        sample_limit=100,  # Start small for testing
        pca_dim=6,
        encodings_to_test=['amplitude'],  # Start with just amplitude first
        test_size=0.2,
        random_state=42,
        quantum_method='statevector'
    )
    
    if results_df is not None:
        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'quantum_encoding_results_{timestamp}.csv'
        results_df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to '{filename}'")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Total encodings tested: {len(results_df)}")
        print(f"🎯 Best overall accuracy: {results_df['accuracy'].max():.4f}")
        
    else:
        print("❌ Experiment failed - no results generated")