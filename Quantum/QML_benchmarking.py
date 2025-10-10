import json 
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import COBYLA, NELDER_MEAD, SPSA, ADAM
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from qiskit import QuantumCircuit
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import time


class Benchmarking:
    def __init__(self):        
        # Benchmark Entries
        self.id = uuid.uuid4()
        self.current_time = time.time()
        self.num_features = None
        
        self.objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)
    
class Dataloader:
    def __init__(self,
                 uri:str="mongodb://localhost:27017/",
                 db:str="Doodle_Classifier",
                 collection:str="Filtered_Features",
                 ):
        """
        Args:
            uri (_type_, optional): _description_. Defaults to "mongodb://localhost:27017/".
            db (str, optional): _description_. Defaults to "Doodle_Classifier".
            collection (str, optional): _description_. Defaults to "Filtered_Features".
        """
        
        self.uri = uri
        self.db_name = db
        self.collection = collection
        
    def load_data(self,num_labels:int=2, sample_limit:int=500):
        """_summary_

        Args:
            num_labels (int, optional): _description_. Defaults to 2.
            sample_limit (int, optional): _description_. Defaults to 500.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        try:
            client = MongoClient(self.uri)
            collection = client[self.db_name][self.collection]
            
            data = []
            
            all_unique_labels = collection.distinct("label")
                
            # Determine which labels to use
            if num_labels:
                if num_labels > len(all_unique_labels):
                    print(f"Warning: Requested {num_labels} labels but only {len(all_unique_labels)} available")
                    labels_to_use = all_unique_labels
                else:
                    # Take first num_labels (you could randomize this if needed)
                    labels_to_use = all_unique_labels[:num_labels]
                    print(f"Using {num_labels} labels out of {len(all_unique_labels)} available")
            else:
                labels_to_use = all_unique_labels
                
            print(f"Selected labels: {labels_to_use}")
            
            if sample_limit:
                samples_per_label = sample_limit // len(labels_to_use)
                
                print(f"Found {len(all_unique_labels)} total unique labels")
                print(f"Using {len(labels_to_use)} labels")
                print(f"Sampling {samples_per_label} samples per label")
                
                # Sample equally from each selected label
                for label in labels_to_use:
                    cursor = collection.find({"label": label}).limit(samples_per_label)
                    for doc in cursor:
                        row = {
                            'label': doc["label"],
                            **doc["features"]
                        }
                        data.append(row)
            else:
                # Load all data for selected labels
                cursor = collection.find({"label": {"$in": labels_to_use}})
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
            
            # Print label distribution
            unique_labels_loaded = np.unique(y, return_counts=True)
            print(f"Loaded {len(X)} samples with {X.shape[1]} features")
            print("Label distribution:")
            for label, count in zip(unique_labels_loaded[0], unique_labels_loaded[1]):
                print(f"  {label}: {count} samples")
                
            return X, y, df
        
        except Exception as e:
                print(f"Error loading data: {e}")
                return None, None, None

class DataProcessing:
    def __init__(self, df:pd.DataFrame,X:np.ndarray=None,y:np.ndarray=None):
        self.df = df
        self.X = X
        self.y = y

    def preprocessing(self,X:np.ndarray,y:np.ndarray):
        """Preprocess the input features and labels.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Input labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and labels.
        """
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        return X, y

    def split(self, split:int=0.8):
        """Split the dataset into training and testing sets.

        Args:
            split (int, optional): Proportion of the dataset to include in the train split. Defaults to 0.8.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split datasets.
        """
        
        X,y = self.preprocessing(self.X,self.y)
        
        algorithm_globals.random_seed = 123
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split, random_state=algorithm_globals.random_seed
        )

        return X_train, X_test, y_train, y_test

class ModelTraining:
    def __init__(self, model, train_features, train_labels):
        self.model = model
        self.train_features = train_features
        self.train_labels = train_labels

    def train(self):
        """Train the machine learning model.

        Returns:
            _type_: _description_
        """
        self.model.fit(self.train_features, self.train_labels)
        return self.model

class SVCModel:
    def __init__(self, C:float=1.0, kernel:str='rbf', gamma:str='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)

    def get_model(self):
        return self.model

class MLPModel:
    def __init__(self, activation:str='relu', max_iter:int=100):
        self.activation = activation
        self.max_iter = max_iter
        self.model = MLPClassifier(activation=self.activation,
                                   max_iter=self.max_iter)

    def get_model(self):
        return self.model

class ModelEvaluation:
    """
        Evaluate the performance of a machine learning model.
    """
    def __init__(self, train_features, train_labels, test_features, test_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels

    def model_select(self,model=SVC()):
        """Select and train a machine learning model.

        Args:
            model (_type_, optional): _description_. Defaults to SVC().

        Returns:
            _type_: _description_
        """
        self.model = model
        self.model.fit(self.train_features, self.train_labels)

        return self.model

    def performance(self,id:str=None, confusion_matrix_dir:str=None):
        """
        Accuracy, Precision, Recall, F1-score, Confusion Matrix, Train/Test accuracy, Generalization gap,
        Training & Inference time, Variance across runs.
        """

        # Implement performance metrics and save the graphs as images
        #Training & Inference time
        start_time = time.time()
        self.model.fit(self.train_features, self.train_labels)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred = self.model.predict(self.test_features)
        inference_time = time.time() - start_time
        
        # Accuracy
        accuracy = accuracy_score(self.test_labels, y_pred)
        
        # Precision, Recall, F1-score
        class_report = classification_report(self.test_labels, y_pred)
        precision = precision_score(self.test_labels, y_pred, average='weighted')
        recall = recall_score(self.test_labels, y_pred, average='weighted')
        f1 = f1_score(self.test_labels, y_pred, average='weighted')
        
        # Train/Test accuracy
        train_accuracy = self.model.score(self.train_features, self.train_labels)
        test_accuracy = self.model.score(self.test_features, self.test_labels)
        generalization_gap = train_accuracy - test_accuracy
        
        # Confusion Matrix
        cm = confusion_matrix(self.test_labels, y_pred)
        
        # Save confusion matrix plot in organized folder structure
        plt.clf()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {id}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Use provided directory or default to current directory
        if confusion_matrix_dir:
            image_path = os.path.join(confusion_matrix_dir, f'confusion_matrix_{id}.png')
        else:
            image_path = f'confusion_matrix_{id}.png'
            
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time": training_time,
            "inference_time": inference_time,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "generalization_gap": generalization_gap,
            "classification_report": class_report,
            "confusion_matrix_path": image_path
        }

class PCA_Reduction:
    def __init__(self, n_components:int=2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def reduce(self, X:np.ndarray):
        """Reduce the dimensionality of the dataset using PCA.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Reduced features.
        """
        X_reduced = self.pca.fit_transform(X)
        return X_reduced

class FeatureMap:
    def __init__(self,  map_type=ZZFeatureMap, feature_dimension:int=2, reps:int=2):
        self.feature_dimension = feature_dimension
        self.reps = reps
        
        self.map_type = map_type
        self.set_feature_map(self.map_type)

    def set_feature_map(self,map_type):
        self.feature_map = map_type(feature_dimension=self.feature_dimension, reps=self.reps)

    def get_feature_map(self):
        return self.feature_map

class Ansatz:
    def __init__(self, ansatz_type=RealAmplitudes, num_qubits:int=2, reps:int=2):
        self.num_qubits = num_qubits
        self.reps = reps
        
        self.ansatz_type = ansatz_type
        self.set_ansatz(self.ansatz_type)

    def set_ansatz(self,ansatz_type):
        if ansatz_type == TwoLocal:
            self.ansatz = ansatz_type(num_qubits=self.num_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=self.reps)
        else:
            self.ansatz = ansatz_type(num_qubits=self.num_qubits, reps=self.reps)

    def get_ansatz(self):
        return self.ansatz

class Optimizer:
    def __init__(self, optimizer_type=COBYLA, maxiter:int=40):
        self.optimizer_type = optimizer_type
        self.maxiter = maxiter
        self.optimizer = None
        
        self.set_optimizer(self.optimizer_type)

    def set_optimizer(self,optimizer_type):
        self.optimizer = optimizer_type(maxiter=self.maxiter)

    def get_optimizer(self):
        return self.optimizer
    
class Sampler_Wrapper:
    def __init__(self, sampler_type=StatevectorSampler):
        self.sampler_type = sampler_type
        self.set_sampler(self.sampler_type)

    def set_sampler(self,sampler_type):
        self.sampler = sampler_type()

    def get_sampler(self):
        return self.sampler

class QML:
    def __init__(self,
                 feature_map=ZZFeatureMap,
                 ansatz=RealAmplitudes,
                 optimizer=COBYLA,
                 num_qubits=2,
                 reps=2,
                 maxiter=40,
                 output_shape=None  # Change made here
                 ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.feature_map = FeatureMap(feature_map, feature_dimension=num_qubits, reps=reps).get_feature_map()
        self.ansatz = Ansatz(ansatz, num_qubits=num_qubits, reps=reps).get_ansatz()
        self.optimizer = Optimizer(optimizer, maxiter=maxiter).get_optimizer()
        self.sampler = Sampler_Wrapper().get_sampler()
        self.output_shape = output_shape if output_shape is not None else 2  # Default to 2 if not provided

    def create_vqc(self):
        """Create a Variational Quantum Classifier (VQC).

        Returns:
            VQC: Configured VQC model
        """
        vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            sampler=self.sampler,
            output_shape=self.output_shape
        )
        return vqc


class QMLBenchmarking:
    """
    Comprehensive benchmarking class for comparing Quantum Machine Learning models
    with classical models using configurable parameters.
    """
    
    def __init__(self, results_dir: str = "/Users/rithvikrajesh/Proper_Project/QC_Project/Benchmarking/"):
        """
        Initialize the benchmarking system.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = results_dir
        self.results = []
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        
        # Create main results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Create organized subdirectories for different types of outputs
        self.images_dir = os.path.join(results_dir, "images")
        self.confusion_matrices_dir = os.path.join(self.images_dir, "confusion_matrices")
        self.plots_dir = os.path.join(self.images_dir, "plots")
        self.reports_dir = os.path.join(results_dir, "reports")
        
        # Create all subdirectories
        for directory in [self.images_dir, self.confusion_matrices_dir, self.plots_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse experiment configuration and validate parameters.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            Parsed and validated configuration
        """
        required_keys = ["experiment_name", "data", "qvc", "training"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        # Set defaults
        parsed_config = {
            "experiment_name": config["experiment_name"],
            "data": {
                "num_classes": config["data"].get("num_classes", 3),
                "num_drawings": config["data"].get("num_drawings", 300),
                "pca_dim": config["data"].get("pca_dim", 6)
            },
            "qvc": {
                "ansatz": config["qvc"].get("ansatz", "RealAmplitudes"),
                "optimizer": config["qvc"].get("optimizer", "COBYLA"),
                "feature_map": config["qvc"].get("feature_map", "ZZFeatureMap"),
                "sampler": config["qvc"].get("sampler", "statevector")
            },
            "training": {
                "train_split": config["training"].get("train_split", 0.8),
                "maxiter": config["training"].get("maxiter", 100)
            }
        }
        
        return parsed_config
        
    def load_and_prepare_data(self, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data according to configuration.
        
        Args:
            config: Parsed configuration
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"Loading data with {config['data']['num_classes']} classes and {config['data']['num_drawings']} drawings...")
        
        # Load data
        dataloader = Dataloader()
        X, y, df = dataloader.load_data(
            num_labels=config['data']['num_classes'],
            sample_limit=config['data']['num_drawings']
        )
        
        if X is None:
            raise ValueError("Failed to load data")
            
        # Apply PCA if specified
        if config['data']['pca_dim'] and config['data']['pca_dim'] < X.shape[1]:
            print(f"Applying PCA to reduce dimensions from {X.shape[1]} to {config['data']['pca_dim']}")
            pca = PCA_Reduction(n_components=config['data']['pca_dim'])
            X = pca.reduce(X)
            print(f"Data shape after PCA: {X.shape}")
            
        # Preprocess and split data
        data_processor = DataProcessing(df, X, y)
        X_train, X_test, y_train, y_test = data_processor.split(split=config['training']['train_split'])
        
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    def get_ansatz_class(self, ansatz_name: str):
        """Get ansatz class from name string."""
        ansatz_mapping = {
            "TwoLocal": TwoLocal,
            "RealAmplitudes": RealAmplitudes,
            "EfficientSU2": EfficientSU2
        }
        return ansatz_mapping.get(ansatz_name, RealAmplitudes)
        
    def get_optimizer_class(self, optimizer_name: str):
        """Get optimizer class from name string."""
        optimizer_mapping = {
            "NELDER_MEAD": NELDER_MEAD,
            "COBYLA": COBYLA,
            "SPSA": SPSA,
            "ADAM": ADAM
        }
        return optimizer_mapping.get(optimizer_name, COBYLA)
        
    def get_feature_map_class(self, feature_map_name: str):
        """Get feature map class from name string."""
        feature_map_mapping = {
            "ZZFeatureMap": ZZFeatureMap,
            "PauliFeatureMap": PauliFeatureMap,
            "ZFeatureMap": ZFeatureMap
        }
        return feature_map_mapping.get(feature_map_name, ZZFeatureMap)
    
    def get_sampler_config(self, sampler_name: str):
        """Get sampler configuration from name string."""
        if sampler_name == "statevector":
            return StatevectorSampler()
        elif sampler_name.startswith("shots_"):
            shots = int(sampler_name.split("_")[1])
            # For shots-based sampling, we'll use StatevectorSampler with specific options
            # In a real implementation, you'd use the appropriate shots-based sampler
            return StatevectorSampler()
        else:
            return StatevectorSampler()
    
    def create_experiment_folders(self, experiment_name: str) -> Dict[str, str]:
        """
        Create experiment-specific folders for organized storage.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with paths to experiment-specific folders
        """
        # Create experiment-specific folder
        experiment_folder = os.path.join(self.results_dir, "experiments", experiment_name)
        experiment_images = os.path.join(experiment_folder, "images")
        experiment_confusion_matrices = os.path.join(experiment_images, "confusion_matrices")
        experiment_plots = os.path.join(experiment_images, "plots")
        experiment_reports = os.path.join(experiment_folder, "reports")
        
        # Create all directories
        folders = [experiment_folder, experiment_images, experiment_confusion_matrices, 
                  experiment_plots, experiment_reports]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        return {
            "experiment_folder": experiment_folder,
            "images": experiment_images,
            "confusion_matrices": experiment_confusion_matrices,
            "plots": experiment_plots,
            "reports": experiment_reports
        }
    
    def generate_comparison_plots(self, quantum_results: Dict[str, Any], 
                                 classical_results: List[Dict[str, Any]], 
                                 experiment_name: str, plots_dir: str):
        """
        Generate comparison plots between quantum and classical models.
        
        Args:
            quantum_results: Results from quantum model
            classical_results: Results from classical models
            experiment_name: Name of the experiment
            plots_dir: Directory to save plots
        """
        # Extract accuracy data
        model_names = []
        accuracies = []
        training_times = []
        
        # Add quantum results if available
        if "accuracy" in quantum_results:
            model_names.append("Quantum VQC")
            accuracies.append(quantum_results["accuracy"])
            training_times.append(quantum_results.get("training_time", 0))
        
        # Add classical results
        for result in classical_results:
            if "accuracy" in result:
                model_names.append(result["model_type"].replace("_", " "))
                accuracies.append(result["accuracy"])
                training_times.append(result.get("training_time", 0))
        
        # Create accuracy comparison plot
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Accuracy comparison
        plt.subplot(1, 2, 1)
        colors = ['red' if 'Quantum' in name else 'blue' for name in model_names]
        bars = plt.bar(range(len(model_names)), accuracies, color=colors, alpha=0.7)
        plt.title(f'Model Accuracy Comparison\n{experiment_name}')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Subplot 2: Training time comparison
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(len(model_names)), training_times, color=colors, alpha=0.7)
        plt.title(f'Training Time Comparison\n{experiment_name}')
        plt.xlabel('Models')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times) * 0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(plots_dir, f'model_comparison_{experiment_name}.png')
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create performance metrics radar chart if we have enough data
        self.create_performance_radar_chart(quantum_results, classical_results, experiment_name, plots_dir)
        
        return comparison_plot_path
    
    def create_performance_radar_chart(self, quantum_results: Dict[str, Any], 
                                     classical_results: List[Dict[str, Any]], 
                                     experiment_name: str, plots_dir: str):
        """
        Create a radar chart showing different performance metrics.
        """
        # Find the best classical model for comparison
        best_classical = max([r for r in classical_results if "accuracy" in r], 
                           key=lambda x: x["accuracy"], default=None)
        
        if not best_classical or "accuracy" not in quantum_results:
            return
        
        # Metrics to compare
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        quantum_values = [
            quantum_results.get('accuracy', 0),
            quantum_results.get('precision', 0),
            quantum_results.get('recall', 0),
            quantum_results.get('f1_score', 0)
        ]
        classical_values = [
            best_classical.get('accuracy', 0),
            best_classical.get('precision', 0),
            best_classical.get('recall', 0),
            best_classical.get('f1_score', 0)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        quantum_values += quantum_values[:1]
        classical_values += classical_values[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, quantum_values, 'o-', linewidth=2, label='Quantum VQC', color='red')
        ax.fill(angles, quantum_values, alpha=0.25, color='red')
        
        ax.plot(angles, classical_values, 'o-', linewidth=2, 
               label=f'Best Classical ({best_classical["model_type"]})', color='blue')
        ax.fill(angles, classical_values, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Comparison Radar Chart\n{experiment_name}', 
                    size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        radar_plot_path = os.path.join(plots_dir, f'performance_radar_{experiment_name}.png')
        plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return radar_plot_path
        
    def train_quantum_model(self, config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate quantum model.
        
        Args:
            config: Experiment configuration
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            Dictionary containing model results
        """
        print("\n" + "="*50)
        print("TRAINING QUANTUM MODEL")
        print("="*50)
        
        start_time = time.time()
        
        # Get classes from config
        ansatz_class = self.get_ansatz_class(config['qvc']['ansatz'])
        optimizer_class = self.get_optimizer_class(config['qvc']['optimizer'])
        feature_map_class = self.get_feature_map_class(config['qvc']['feature_map'])
        sampler = self.get_sampler_config(config['qvc']['sampler'])
        
        # Create quantum model
        num_qubits = X_train.shape[1]  # Number of features
        qml_model = QML(
            feature_map=feature_map_class,
            ansatz=ansatz_class,
            optimizer=optimizer_class,
            num_qubits=num_qubits,
            reps=2,
            maxiter=config['training']['maxiter'],
            output_shape=config['data']['num_classes'],  # Set output shape based on number of classes
        )
        
        vqc = qml_model.create_vqc()
        
        # Train model
        print(f"Training quantum model:")
        print(f"  Ansatz: {config['qvc']['ansatz']}")
        print(f"  Optimizer: {config['qvc']['optimizer']}")
        print(f"  Feature Map: {config['qvc']['feature_map']}")
        print(f"  Sampler: {config['qvc']['sampler']}")
        print(f"  Qubits: {num_qubits}, Max iterations: {config['training']['maxiter']}")
        
        try:
            vqc.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"Quantum model training completed in {training_time:.2f} seconds")
            
            # Evaluate model
            evaluator = ModelEvaluation(X_train, y_train, X_test, y_test)
            evaluator.model_select(vqc)
            performance = evaluator.performance(
                id=f"quantum_{config['experiment_name']}_{self.experiment_id[:8]}", 
                confusion_matrix_dir=self.confusion_matrices_dir
            )
            
            # Add model-specific info
            performance.update({
                "model_type": "Quantum_VQC",
                "ansatz": config['qvc']['ansatz'],
                "optimizer": config['qvc']['optimizer'],
                "feature_map": config['qvc']['feature_map'],
                "num_qubits": num_qubits,
                "training_time": training_time,
                "maxiter": config['training']['maxiter']
            })
            
            return performance
            
        except Exception as e:
            print(f"Error training quantum model: {e}")
            return {
                "model_type": "Quantum_VQC",
                "error": str(e),
                "training_time": time.time() - start_time
            }
    
    def train_classical_models(self, config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> List[Dict[str, Any]]:
        """
        Train and evaluate multiple classical models for comparison.
        
        Args:
            config: Experiment configuration
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            List of dictionaries containing results for each classical model
        """
        print("\n" + "="*50)
        print("TRAINING CLASSICAL MODELS")
        print("="*50)
        
        classical_results = []
        
        # Define classical models to compare
        models = {
            "SVM_RBF": SVC(kernel='rbf', C=1.0, gamma='scale'),
            "SVM_Linear": SVC(kernel='linear', C=1.0),
            "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "MLP_Neural_Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=config['training']['maxiter'], random_state=42),
            "Logistic_Regression": LogisticRegression(max_iter=config['training']['maxiter'], random_state=42)
        }
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            try:
                # Train and evaluate model
                evaluator = ModelEvaluation(X_train, y_train, X_test, y_test)
                evaluator.model_select(model)
                performance = evaluator.performance(
                    id=f"{model_name}_{config['experiment_name']}_{self.experiment_id[:8]}", 
                    confusion_matrix_dir=self.confusion_matrices_dir
                )
                
                training_time = time.time() - start_time
                
                # Add model-specific info
                performance.update({
                    "model_type": model_name,
                    "training_time": training_time
                })
                
                classical_results.append(performance)
                print(f"{model_name} - Accuracy: {performance['accuracy']:.4f}, Training time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                classical_results.append({
                    "model_type": model_name,
                    "error": str(e),
                    "training_time": time.time() - start_time
                })
        
        return classical_results
    
    def save_results(self, experiment_config: Dict[str, Any], quantum_results: Dict[str, Any], 
                     classical_results: List[Dict[str, Any]], experiment_folders: Dict[str, str],
                     comparison_plot_path: str = None, radar_plot_path: str = None):
        """
        Save comprehensive benchmark results to JSON file with organized folder structure.
        
        Args:
            experiment_config: Original experiment configuration
            quantum_results: Results from quantum model
            classical_results: Results from classical models
            experiment_folders: Dictionary containing paths to experiment folders
            comparison_plot_path: Path to comparison plot
            radar_plot_path: Path to radar chart plot
        """
        # Compile comprehensive results with image paths
        benchmark_results = {
            "experiment_metadata": {
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "experiment_name": experiment_config["experiment_name"],
                "configuration": experiment_config
            },
            "data_info": {
                "num_classes": experiment_config["data"]["num_classes"],
                "num_drawings": experiment_config["data"]["num_drawings"],
                "pca_dimensions": experiment_config["data"]["pca_dim"],
                "train_split": experiment_config["training"]["train_split"]
            },
            "quantum_model_results": quantum_results,
            "classical_models_results": classical_results,
            "comparison_summary": self._create_comparison_summary(quantum_results, classical_results),
            "generated_files": {
                "experiment_folder": experiment_folders["experiment_folder"],
                "confusion_matrices_folder": experiment_folders["confusion_matrices"],
                "plots_folder": experiment_folders["plots"],
                "comparison_plot": comparison_plot_path,
                "radar_plot": radar_plot_path,
                "confusion_matrices": []
            }
        }
        
        # Collect confusion matrix paths
        all_results = [quantum_results] + classical_results
        for result in all_results:
            if "confusion_matrix_path" in result:
                benchmark_results["generated_files"]["confusion_matrices"].append(result["confusion_matrix_path"])
        
        # Save to experiment-specific folder
        experiment_filename = f"{experiment_config['experiment_name']}_{self.experiment_id[:8]}.json"
        experiment_path = os.path.join(experiment_folders["experiment_folder"], experiment_filename)
        
        with open(experiment_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Append to master results file
        results_path = os.path.join(self.results_dir, "results.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = {"experiments": []}
        else:
            all_results = {"experiments": []}
        
        all_results["experiments"].append(benchmark_results)
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n" + "="*50)
        print("RESULTS SAVED")
        print("="*50)
        print(f"Individual experiment results: {experiment_path}")
        print(f"Master results file: {results_path}")
    
    def _create_comparison_summary(self, quantum_results: Dict[str, Any], classical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comparison summary between quantum and classical models.
        
        Args:
            quantum_results: Quantum model results
            classical_results: Classical models results
            
        Returns:
            Comparison summary dictionary
        """
        summary = {
            "best_classical_model": None,
            "best_classical_accuracy": 0.0,
            "quantum_accuracy": quantum_results.get("accuracy", 0.0),
            "quantum_vs_classical": {},
            "training_time_comparison": {}
        }
        
        # Find best classical model
        for result in classical_results:
            if "accuracy" in result and result["accuracy"] > summary["best_classical_accuracy"]:
                summary["best_classical_accuracy"] = result["accuracy"]
                summary["best_classical_model"] = result["model_type"]
        
        # Comparison metrics
        if summary["best_classical_accuracy"] > 0:
            summary["quantum_vs_classical"]["accuracy_difference"] = summary["quantum_accuracy"] - summary["best_classical_accuracy"]
            summary["quantum_vs_classical"]["quantum_better"] = summary["quantum_accuracy"] > summary["best_classical_accuracy"]
        
        # Training time comparison
        summary["training_time_comparison"]["quantum_time"] = quantum_results.get("training_time", 0.0)
        summary["training_time_comparison"]["classical_times"] = {
            result["model_type"]: result.get("training_time", 0.0) 
            for result in classical_results if "training_time" in result
        }
        
        return summary
    
    def run_benchmark(self, config: Dict[str, Any]):
        """
        Main benchmarking function that runs the complete experiment.
        
        Args:
            config: Experiment configuration dictionary
        """
        print("="*70)
        print("QUANTUM MACHINE LEARNING BENCHMARKING")
        print("="*70)
        
        try:
            # Parse and validate configuration
            parsed_config = self.parse_config(config)
            print(f"Running experiment: {parsed_config['experiment_name']}")
            print(f"Experiment ID: {self.experiment_id}")
            
            # Create experiment-specific folders
            experiment_folders = self.create_experiment_folders(parsed_config['experiment_name'])
            print(f"Experiment folder created: {experiment_folders['experiment_folder']}")
            
            # Update confusion matrices directory to use experiment-specific folder
            self.confusion_matrices_dir = experiment_folders["confusion_matrices"]
            
            # Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data(parsed_config)
            
            # Train quantum model
            quantum_results = self.train_quantum_model(parsed_config, X_train, y_train, X_test, y_test)
            
            # Train classical models
            classical_results = self.train_classical_models(parsed_config, X_train, y_train, X_test, y_test)
            
            # Generate comparison plots
            print("\nGenerating comparison plots...")
            comparison_plot_path = self.generate_comparison_plots(
                quantum_results, classical_results, 
                parsed_config['experiment_name'], 
                experiment_folders["plots"]
            )
            
            # Save results with organized structure
            self.save_results(parsed_config, quantum_results, classical_results, 
                            experiment_folders, comparison_plot_path)
            
            # Print summary
            self._print_summary(quantum_results, classical_results)
            
            print(f"\n" + "="*50)
            print("RESULTS SAVED")
            print("="*50)
            print(f"Experiment folder: {experiment_folders['experiment_folder']}")
            print(f"Confusion matrices: {experiment_folders['confusion_matrices']}")
            print(f"Comparison plots: {experiment_folders['plots']}")
            print(f"Master results file: {os.path.join(self.results_dir, 'results.json')}")
            
        except Exception as e:
            print(f"Error during benchmarking: {e}")
            raise
    
    def _print_summary(self, quantum_results: Dict[str, Any], classical_results: List[Dict[str, Any]]):
        """Print a summary of the benchmark results."""
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        # Quantum results
        if "accuracy" in quantum_results:
            print(f"Quantum VQC Accuracy: {quantum_results['accuracy']:.4f}")
            print(f"Quantum Training Time: {quantum_results.get('training_time', 'N/A'):.2f}s")
        else:
            print(f"Quantum VQC: Failed - {quantum_results.get('error', 'Unknown error')}")
        
        print("\nClassical Models:")
        print("-" * 30)
        
        # Sort classical results by accuracy
        valid_results = [r for r in classical_results if "accuracy" in r]
        valid_results.sort(key=lambda x: x["accuracy"], reverse=True)
        
        for result in valid_results:
            print(f"{result['model_type']:20} - Accuracy: {result['accuracy']:.4f}, Time: {result.get('training_time', 'N/A'):.2f}s")
        
        # Failed models
        failed_results = [r for r in classical_results if "error" in r]
        if failed_results:
            print("\nFailed Models:")
            for result in failed_results:
                print(f"{result['model_type']:20} - Error: {result['error']}")


def main():
    """
    Example usage of the QMLBenchmarking class with the provided configuration.
    """
    # Example configuration as provided by the user
    config = {
        "experiment_name": "run_05_twolocal_neldermead_pca6",
        "data": { "num_classes": 3, "num_drawings": 300, "pca_dim": 6 },
        "qvc": { "ansatz": "TwoLocal", "optimizer": "NELDER_MEAD", "feature_map": "ZZFeatureMap" },
        "training": { "train_split": 0.85, "maxiter": 80 }
    }
    
    # Create and run benchmark
    benchmark = QMLBenchmarking()
    benchmark.run_benchmark(config)



if __name__ == "__main__":
    main()
