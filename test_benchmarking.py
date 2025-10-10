#!/usr/bin/env python3
"""
Test script for QML Benchmarking system.

This script demonstrates how to use the comprehensive benchmarking system
with your specific configuration format.
"""

import sys
import os
import json

# Add the Quantum directory to the path so we can import our module
sys.path.append('/Users/rithvikrajesh/Proper_Project/QC_Project/Quantum')


from QML_benchmarking import QMLBenchmarking

def test_config_parsing():
    """Test the configuration parsing functionality."""
    print("Testing configuration parsing...")
    
    config = {
        "experiment_name": "test_run_twolocal_neldermead_pca6",
        "data": { "num_classes": 3, "num_drawings": 50, "pca_dim": 6 },  # Reduced for testing
        "qvc": { "ansatz": "TwoLocal", "optimizer": "NELDER_MEAD", "feature_map": "ZZFeatureMap" },
        "training": { "train_split": 0.85, "maxiter": 10 }  # Reduced iterations for testing
    }
    
    benchmark = QMLBenchmarking()
    parsed_config = benchmark.parse_config(config)
    
    print("✓ Configuration parsed successfully")
    print(f"  Experiment name: {parsed_config['experiment_name']}")
    print(f"  Data settings: {parsed_config['data']}")
    print(f"  QVC settings: {parsed_config['qvc']}")
    print(f"  Training settings: {parsed_config['training']}")
    
    return parsed_config

def test_ansatz_optimizer_mapping():
    """Test the ansatz and optimizer mapping functionality."""
    print("\nTesting model component mapping...")
    
    benchmark = QMLBenchmarking()
    
    # Test ansatz mapping
    ansatz_class = benchmark.get_ansatz_class("TwoLocal")
    print(f"✓ TwoLocal ansatz class: {ansatz_class}")
    
    # Test optimizer mapping
    optimizer_class = benchmark.get_optimizer_class("NELDER_MEAD")
    print(f"✓ NELDER_MEAD optimizer class: {optimizer_class}")
    
    # Test feature map mapping
    feature_map_class = benchmark.get_feature_map_class("ZZFeatureMap")
    print(f"✓ ZZFeatureMap feature map class: {feature_map_class}")

def run_small_benchmark():
    """Run a small benchmark test (requires MongoDB to be running)."""
    print("\nRunning small benchmark test...")
    print("Note: This requires MongoDB to be running with the doodle data")
    
    # Small test configuration
    test_config = {
        "experiment_name": "test_run_small",
        "data": { "num_classes": 2, "num_drawings": 20, "pca_dim": 4 },  # Very small for testing
        "qvc": { "ansatz": "TwoLocal", "optimizer": "COBYLA", "feature_map": "ZZFeatureMap" },
        "training": { "train_split": 0.8, "maxiter": 5 }  # Very few iterations
    }
    
    try:
        benchmark = QMLBenchmarking()
        benchmark.run_benchmark(test_config)
        print("✓ Small benchmark completed successfully")
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        print("This is likely due to MongoDB not running or missing data")

def demonstrate_configuration_examples():
    """Show different configuration examples."""
    print("\nConfiguration Examples:")
    print("="*50)
    
    # Original configuration
    original_config = {
        "experiment_name": "run_05_twolocal_neldermead_pca6",
        "data": { "num_classes": 3, "num_drawings": 300, "pca_dim": 6 },
        "qvc": { "ansatz": "TwoLocal", "optimizer": "NELDER_MEAD", "feature_map": "ZZFeatureMap" },
        "training": { "train_split": 0.85, "maxiter": 80 }
    }
    
    # Alternative configuration for comparison
    comparison_config = {
        "experiment_name": "run_06_realamplitudes_cobyla_pca8",
        "data": { "num_classes": 3, "num_drawings": 300, "pca_dim": 8 },
        "qvc": { "ansatz": "RealAmplitudes", "optimizer": "COBYLA", "feature_map": "ZZFeatureMap" },
        "training": { "train_split": 0.85, "maxiter": 100 }
    }
    
    print("1. Original Configuration:")
    print(json.dumps(original_config, indent=2))
    
    print("\n2. Alternative Configuration (different ansatz/optimizer):")
    print(json.dumps(comparison_config, indent=2))
    
    # Show expected results structure
    print("\n3. Expected Results Structure:")
    expected_results = {
        "experiment_metadata": {
            "experiment_id": "uuid-string",
            "timestamp": "ISO-timestamp",
            "experiment_name": "experiment_name",
            "configuration": "original_config"
        },
        "data_info": {
            "num_classes": "number",
            "num_drawings": "number", 
            "pca_dimensions": "number",
            "train_split": "ratio"
        },
        "quantum_model_results": {
            "model_type": "Quantum_VQC",
            "accuracy": "float",
            "precision": "float",
            "recall": "float", 
            "f1_score": "float",
            "training_time": "seconds",
            "ansatz": "ansatz_type",
            "optimizer": "optimizer_type"
        },
        "classical_models_results": [
            {
                "model_type": "SVM_RBF",
                "accuracy": "float",
                "training_time": "seconds"
            }
        ],
        "comparison_summary": {
            "best_classical_model": "model_name",
            "best_classical_accuracy": "float",
            "quantum_accuracy": "float",
            "quantum_vs_classical": {
                "accuracy_difference": "float",
                "quantum_better": "boolean"
            }
        }
    }
    print(json.dumps(expected_results, indent=2))

if __name__ == "__main__":
    print("QML Benchmarking System Test")
    print("="*50)
    
    # Run tests
    test_config_parsing()
    test_ansatz_optimizer_mapping()
    demonstrate_configuration_examples()
    
    # Optionally run small benchmark if user wants to test with real data
    user_input = input("\nDo you want to run a small benchmark test? (requires MongoDB) [y/N]: ")
    if user_input.lower().startswith('y'):
        run_small_benchmark()
    else:
        print("\nSkipping benchmark test. To run full benchmarks, use:")
        print("from QML_benchmarking import QMLBenchmarking")
        print("benchmark = QMLBenchmarking()")
        print("benchmark.run_benchmark(your_config)")
    
    print("\n✓ All tests completed!")