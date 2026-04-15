#!/usr/bin/env python3
"""
Simple MLP Training Script (No YAML Config Required)

This script provides MLP training without requiring PyYAML configuration files.
All parameters are set directly in the code for easy modification.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Add plotting utilities
sys.path.append(str(Path(__file__).parent))
from utils.plotting import plot_prediction_scatter, plot_training_curves
from utils.mlflow_manager import mlflow_training_session, create_experiment_manager

from src.models.registry import create_mlp
from src.core.data_generator import DataGenerator
from src.api.preprocessing import PreprocessingPipeline
from src.core.dependencies import print_environment_info

def main():
    print("🧠 Simple MLP Training with MLFlow")
    print("=" * 50)
    
    # Configuration (easily modifiable)
    config = {
        'maxorder': 5,
        'n_samples': 1000,
        'hidden_size': 256,
        'n_hidden_layers': 3,
        'epochs': 25,
        'batch_size': 64,
        'learning_rate': 0.001,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'pca_components': 128,
    }
    
    # MLFlow experiment configuration
    experiment_name = "simple_mlp_training"
    run_name = f"quick_training_{int(time.time())}"
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Environment info
    print("\n🔧 Environment:")
    print_environment_info()
    
    # Start MLFlow experiment session
    with mlflow_training_session(experiment_name, run_name, tags={'type': 'simple_training'}) as session:
        
        # Log configuration
        session.tracker.log_params({f"config.{k}": v for k, v in config.items()})
        
        # 1. Generate training data
        print(f"\n1️⃣ Generating {config['n_samples']} samples...")
        start_time = time.time()
        
        generator = DataGenerator.for_ml_training()
        dataset = generator.generate_batch(
            maxorder=config['maxorder'], 
            n_samples=config['n_samples']
        )
        
        data_time = time.time() - start_time
        print(f"   ✅ Data generated in {data_time:.2f}s")
        print(f"   📊 Field shape: {dataset['amplitude'].shape}")
        print(f"   📈 Coefficients E: {dataset['coefficients_e'].shape}")
        print(f"   📈 Coefficients M: {dataset['coefficients_m'].shape}")
        
        # Log data generation metrics
        session.tracker.log_metric("data_generation_time", data_time)
        session.tracker.log_metric("n_samples_generated", config['n_samples'])
        
        # 2. Setup preprocessing
        print("\n2️⃣ Setting up preprocessing...")
        preprocessing = PreprocessingPipeline()
        
        E_theta = dataset['amplitude'][..., 0]
        E_phi = dataset['amplitude'][..., 1]
        a_e = dataset['coefficients_e']
        a_m = dataset['coefficients_m']
        
        from src.core.data_generator import pack_coefficients
        targets = pack_coefficients(a_e, a_m)
        
        preprocessing.fit(E_theta, E_phi, targets=targets)
        X = preprocessing.transform_features(E_theta, E_phi)
        y = preprocessing.process_coefficients(a_e, a_m)
        
        print(f"   ✅ Features: {X.shape}, Targets: {y.shape}")
        
        # Log preprocessing metrics
        session.tracker.log_metric("n_features", X.shape[1])
        session.tracker.log_metric("n_targets", y.shape[1])
        
        # 3. Create model
        print("\n3️⃣ Creating MLP model...")
        model = create_mlp(
            maxorder=config['maxorder'],
            input_dim=X.shape[1],
            hidden_size=config['hidden_size'],
            n_hidden_layers=config['n_hidden_layers'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        print(f"   ✅ Created: {model}")
        
        # 4. Split data
        print("\n4️⃣ Splitting data...")
        n_samples = len(X)
        n_train = int(config['train_ratio'] * n_samples)
        n_val = int(config['val_ratio'] * n_samples)
        
        X_train = X[:n_train]
        X_val = X[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        
        y_train = y[:n_train]
        y_val = y[n_train:n_train + n_val]
        y_test = y[n_train + n_val:]
        
        print(f"   📊 Train: {len(X_train)} samples")
        print(f"   📊 Val:   {len(X_val)} samples")
        print(f"   📊 Test:  {len(X_test)} samples")
        
        # Log dataset split info
        session.tracker.log_metric("n_train", len(X_train))
        session.tracker.log_metric("n_val", len(X_val))
        session.tracker.log_metric("n_test", len(X_test))
        
        # 5. Train model
        print("\n5️⃣ Training model...")
        train_start = time.time()
        result = model.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - train_start
        
        print(f"   ✅ Training completed in {train_time:.2f}s!")
        print(f"   📈 Final train loss: {result.get('final_train_loss', 'N/A'):.6f}")
        print(f"   📈 Final val loss: {result.get('final_val_loss', 'N/A'):.6f}")
        
        # Log training metrics
        training_metrics = {
            "training_time": train_time,
            "final_train_loss": result.get('final_train_loss', 0),
            "final_val_loss": result.get('final_val_loss', 0)
        }
        session.log_model_performance(training_metrics)
        
        # 6. Test model
        print("\n6️⃣ Testing model...")
        predictions = model.predict(X_test)
        test_mse = np.mean((y_test - predictions) ** 2)
        test_r2 = 1 - np.var(y_test - predictions) / np.var(y_test) if np.var(y_test) > 0 else 0
        
        print(f"   📊 Test MSE: {test_mse:.6f}")
        print(f"   📊 Test R²: {test_r2:.6f}")
        
        # Log test metrics
        test_metrics = {
            "test_mse": float(test_mse),
            "test_r2": float(test_r2)
        }
        session.log_model_performance(test_metrics)
        
        # 7. Save results and generate plots
        print("\n7️⃣ Saving results...")
        save_dir = Path("experiments/results/simple_training")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 7.5. Generate plots
        print("\n📊 Generating plots...")
        try:
            # Training curves (if available)
            if 'training_history' in result:
                plot_training_curves(
                    result['training_history'],
                    save_path=save_dir / "training_curves.png"
                )
                print(f"   ✅ Training curves saved")
            
            # Prediction scatter plot
            plot_prediction_scatter(
                y_test, predictions,
                title="Test Predictions vs True Values",
                save_path=save_dir / "predictions_scatter.png"
            )
            print(f"   ✅ Prediction scatter plot saved")
            
            # Log plots to MLFlow
            session.log_plots(save_dir)
            
        except Exception as e:
            print(f"   ⚠️ Plotting failed: {e}")
        
        # 8. Register model in MLFlow
        print("\n🤖 Registering model...")
        try:
            # Create input example for model signature
            input_example = X_test[:5] if len(X_test) >= 5 else X_test
            
            model_version = session.register_model(
                model=model,
                model_name="simple_mlp_electromagnetic",
                input_example=input_example,
                performance_metrics=test_metrics,
                auto_promote=True
            )
            
            if model_version:
                print(f"   ✅ Model registered as version {model_version}")
            
        except Exception as e:
            print(f"   ⚠️ Model registration failed: {e}")
        
        # 9. Save artifacts locally
        model.save(save_dir / "model")
        preprocessing.save(save_dir / "preprocessing") 
    
        # Save run summary (convert numpy types for JSON serialization)
        summary = {
            'config': config,
            'data_generation_time': float(data_time),
            'training_time': float(train_time),
            'test_mse': float(test_mse),
            'test_r2': float(test_r2),
            'final_train_loss': float(result.get('final_train_loss', 0)),
            'final_val_loss': float(result.get('final_val_loss', 0)),
            'model_path': str(save_dir)
        }
        
        import json
        with open(save_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   💾 Results saved to: {save_dir}")
        
        # Log local artifacts to MLFlow
        session.log_artifacts_directory(save_dir, "experiment_results")
        
        # Final summary
        total_time = data_time + train_time
        print(f"\n🎉 Training Complete with MLFlow Tracking!")
        print("=" * 50)
        print(f"📊 Test MSE: {test_mse:.6f}")
        print(f"📊 Test R²: {test_r2:.6f}")
        print(f"⏱️ Total Time: {total_time:.2f}s (Data: {data_time:.2f}s, Training: {train_time:.2f}s)")
        print(f"🚀 Samples/sec: {config['n_samples']/total_time:.1f}")
        
        # Show MLFlow experiment URL if available
        experiment_url = session.get_experiment_url()
        if experiment_url:
            print(f"📈 View in MLFlow: {experiment_url}")
        
        return {
            'test_mse': float(test_mse),
            'test_r2': float(test_r2),
            'training_time': float(train_time),
            'model_path': save_dir,
            'experiment_url': experiment_url
        }

if __name__ == "__main__":
    try:
        result = main()
        print(f"\n✅ Success! MSE: {result['test_mse']:.6f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()