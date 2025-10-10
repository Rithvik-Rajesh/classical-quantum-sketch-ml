"""
Enhanced QML Benchmarking System with JSON Config Support and Progress Tracking

This file supports running experiments from JSON configuration files with progress bars.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
import time

# Add path to import the benchmarking module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Quantum'))

from QML_benchmarking import QMLBenchmarking

class ProgressTracker:
    """Simple progress tracker that works without external dependencies."""
    
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._display()
        
    def _display(self):
        """Display current progress."""
        if self.total == 0:
            percentage = 100
        else:
            percentage = (self.current / self.total) * 100
            
        elapsed = time.time() - self.start_time
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description} |{bar}| {self.current}/{self.total} ({percentage:.1f}%) - {elapsed:.1f}s', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

class ConfigRunner:
    """Enhanced runner that handles JSON configs with progress tracking."""
    
    def __init__(self):
        self.benchmark = QMLBenchmarking()
        self.config_dir = Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
    def load_config_file(self, config_path: str):
        """Load configuration from JSON file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Try relative to configs directory
            config_file = self.config_dir / config_path
            if not config_file.suffix:
                config_file = config_file.with_suffix('.json')
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded config from: {config_file}")
        return config
    
    def run_single_config(self, config: Dict[str, Any], experiment_num: int = 1, total_experiments: int = 1) -> str:
        """Run a single experiment configuration with progress tracking."""
        
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {experiment_num}/{total_experiments}: {config['experiment_name']}")
        print(f"{'='*80}")
        
        # Display config details
        print(f"📊 Data Configuration:")
        print(f"   Classes: {config['data']['num_classes']}, Drawings: {config['data']['num_drawings']}, PCA: {config['data']['pca_dim']}")
        
        print(f"⚛️  Quantum Configuration:")
        print(f"   Ansatz: {config['qvc']['ansatz']}, Optimizer: {config['qvc']['optimizer']}")
        print(f"   Feature Map: {config['qvc']['feature_map']}, Sampler: {config['qvc']['sampler']}")
        
        print(f"🎯 Training Configuration:")
        print(f"   Train Split: {config['training']['train_split']}, Max Iterations: {config['training']['maxiter']}")
        
        print("-" * 80)
        
        start_time = time.time()
        try:
            result = self.benchmark.run_benchmark(config)
            elapsed_time = time.time() - start_time
            
            print(f"\n✅ Experiment '{config['experiment_name']}' completed successfully!")
            print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n❌ Experiment '{config['experiment_name']}' failed after {elapsed_time:.2f}s: {str(e)}")
            raise
    
    def run_config_file(self, config_path: str) -> List[str]:
        """Run experiment(s) from a JSON config file with progress tracking."""
        config_data = self.load_config_file(config_path)
        results = []
        
        if isinstance(config_data, list):
            total_experiments = len(config_data)
            print(f"\n🚀 Starting batch run: {total_experiments} experiments")
            print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Overall progress tracker
            overall_start = time.time()
            
            for i, config in enumerate(config_data, 1):
                try:
                    result = self.run_single_config(config, i, total_experiments)
                    results.append(result)
                    
                    # Show overall progress
                    elapsed = time.time() - overall_start
                    avg_time = elapsed / i
                    remaining_time = avg_time * (total_experiments - i)
                    
                    print(f"\n📈 Overall Progress: {i}/{total_experiments} experiments completed")
                    print(f"⏱️  Average time per experiment: {avg_time:.1f}s")
                    print(f"⏳ Estimated remaining time: {remaining_time:.1f}s")
                    
                except Exception as e:
                    print(f"❌ Skipping experiment {i} due to error: {e}")
                    results.append(None)
            
            total_time = time.time() - overall_start
            successful = sum(1 for r in results if r is not None)
            
            print(f"\n{'='*80}")
            print(f"🏁 BATCH RUN COMPLETE")
            print(f"{'='*80}")
            print(f"✅ Successful experiments: {successful}/{total_experiments}")
            print(f"⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"📅 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        else:
            result = self.run_single_config(config_data)
            results.append(result)
        
        return results
    
    def create_sample_configs(self):
        """Create sample configuration files including the comprehensive experiment set."""
        
        # Quick test config
        test_config = {
            "experiment_name": "quick_test_run",
            "data": { "num_classes": 2, "num_drawings": 50, "pca_dim": 2 },
            "qvc": { "ansatz": "RealAmplitudes", "optimizer": "COBYLA", "feature_map": "ZZFeatureMap", "sampler": "statevector" },
            "training": { "train_split": 0.8, "maxiter": 10 }
        }
        
        # Single experiment test
        single_config = {
            "experiment_name": "single_test_experiment",
            "data": { "num_classes": 3, "num_drawings": 100, "pca_dim": 3 },
            "qvc": { "ansatz": "EfficientSU2", "optimizer": "COBYLA", "feature_map": "PauliFeatureMap", "sampler": "statevector" },
            "training": { "train_split": 0.8, "maxiter": 50 }
        }
        
        # Save sample configs
        with open(self.config_dir / "quick_test.json", 'w') as f:
            json.dump(test_config, f, indent=2)
            
        with open(self.config_dir / "single_experiment.json", 'w') as f:
            json.dump(single_config, f, indent=2)
        
        print(f"✅ Sample config files created in: {self.config_dir}")
        print("   - quick_test.json (1 quick test)")
        print("   - single_experiment.json (1 experiment)")
        print(f"\n💡 Full comprehensive config with 15 experiments is available at:")
        print(f"   {self.config_dir / 'comprehensive_experiments.json'}")

def run_comprehensive_experiments():
    """Run the comprehensive experiment set."""
    runner = ConfigRunner()
    print("🚀 Running comprehensive experiment set...")
    runner.run_config_file("comprehensive_experiments.json")

def interactive_menu():
    """Interactive menu for running experiments."""
    runner = ConfigRunner()
    
    while True:
        print("\n" + "="*80)
        print("🧪 QML Benchmarking System - Enhanced JSON Config Runner")
        print("="*80)
        
        # List available config files
        config_files = list(runner.config_dir.glob("*.json"))
        
        print(f"\n📁 Available config files in '{runner.config_dir}':")
        for i, file in enumerate(config_files, 1):
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1
                    print(f"   {i}. {file.name} ({count} experiment{'s' if count > 1 else ''})")
                except:
                    print(f"   {i}. {file.name} (unknown format)")
        
        print(f"\n🎯 Options:")
        print("1. Run from config file (enter filename)")
        print("2. Run comprehensive experiments (all 15)")
        print("3. Quick test (single fast experiment)")
        print("4. Create sample config files")
        print("5. Exit")
        
        choice = input(f"\nEnter choice (1-5): ").strip()
        
        try:
            if choice == "1":
                config_path = input("Enter config file name: ").strip()
                print(f"\n🚀 Starting experiments from: {config_path}")
                runner.run_config_file(config_path)
                
            elif choice == "2":
                print(f"\n⚠️  This will run 15 comprehensive experiments!")
                confirm = input("Are you sure? This may take a long time (y/N): ").strip().lower()
                if confirm.startswith('y'):
                    run_comprehensive_experiments()
                else:
                    print("❌ Cancelled")
                
            elif choice == "3":
                print(f"\n🏃‍♂️ Running quick test...")
                runner.run_config_file("quick_test.json")
                
            elif choice == "4":
                runner.create_sample_configs()
                
            elif choice == "5":
                print("👋 Exiting...")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\n💡 Troubleshooting:")
            print("   - Ensure MongoDB is running")
            print("   - Check database 'Doodle_Classifier' exists")
            print("   - Verify collection 'Filtered_Features' has data")
            print("   - Make sure config file exists and has correct format")

def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description='Enhanced QML Benchmarking System with Progress Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarking.py --config comprehensive_experiments.json
  python run_benchmarking.py --comprehensive
  python run_benchmarking.py --quick-test
  python run_benchmarking.py --interactive
        """
    )
    
    parser.add_argument('--config', '-c', 
                       help='Path to JSON config file')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run all 15 comprehensive experiments')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test experiment')
    parser.add_argument('--create-samples', action='store_true', 
                       help='Create sample config files')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    runner = ConfigRunner()
    
    if args.create_samples:
        runner.create_sample_configs()
        return
    
    if args.config:
        print(f"🚀 Running experiments from: {args.config}")
        runner.run_config_file(args.config)
        return
    
    if args.comprehensive:
        print("⚠️  This will run 15 comprehensive experiments!")
        confirm = input("Continue? This may take several hours (y/N): ").strip().lower()
        if confirm.startswith('y'):
            run_comprehensive_experiments()
        return
    
    if args.quick_test:
        print("🏃‍♂️ Running quick test...")
        runner.run_config_file("quick_test.json")
        return
    
    # Default to interactive mode
    interactive_menu()

if __name__ == "__main__":
    print("🧬 Enhanced QML Benchmarking System")
    print("="*50)
    main()