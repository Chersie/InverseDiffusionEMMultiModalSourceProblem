#!/usr/bin/env python3
"""
Quick Example: Train an MLP Model

This is a simple example showing how to train an MLP model
using the new experiments framework.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.registry import create_mlp
from src.core.data_generator import DataGenerator, pack_coefficients
from src.api.preprocessing import PreprocessingPipeline
import numpy as np


def quick_mlp_example():
    """Quick example of training an MLP model."""
    print("🧠 Quick MLP Training Example")
    print("="*50)
    
    # 1. Create a small MLP for testing (we'll set input_dim after preprocessing)
    print("1️⃣ Creating MLP model...")
    
    # 2. Generate synthetic training data
    print("2️⃣ Generating synthetic data...")
    generator = DataGenerator.for_ml_training()
    dataset = generator.generate_batch(maxorder=3, n_samples=200)  # Small dataset
    
    print(f"   ✅ Generated {dataset['n_samples']} samples")
    print(f"   📊 Field shape: {dataset['amplitude'].shape}")
    print(f"   📈 Coefficients E: {dataset['coefficients_e'].shape}")
    print(f"   📈 Coefficients M: {dataset['coefficients_m'].shape}")
    
    # 3. Set up preprocessing
    print("\n3️⃣ Setting up preprocessing...")
    preprocessing = PreprocessingPipeline()
    
    E_theta = dataset['amplitude'][..., 0]  # Shape: (n_samples, n_phi, n_theta)
    E_phi = dataset['amplitude'][..., 1]
    
    # Fit preprocessing pipeline
    a_e = dataset['coefficients_e']
    a_m = dataset['coefficients_m']
    targets = pack_coefficients(a_e, a_m)
    
    preprocessing.fit(E_theta, E_phi, targets=targets)
    X = preprocessing.transform_features(E_theta, E_phi)
    y = preprocessing.process_coefficients(a_e, a_m)
    
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Targets shape: {y.shape}")
    
    # Now create MLP model with correct input dimension
    model = create_mlp(
        maxorder=3,           # Small for quick testing
        input_dim=X.shape[1], # Match preprocessing output
        hidden_size=128,      # Small network
        n_hidden_layers=2,
        epochs=5,            # Few epochs for quick test
        batch_size=32,
        learning_rate=0.001
    )
    print(f"   ✅ Created: {model}")
    
    # 4. Split data
    print("\n4️⃣ Splitting data...")
    n_train = int(0.7 * len(X))
    n_val = int(0.2 * len(X))
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    
    print(f"   📊 Train: {X_train.shape[0]} samples")
    print(f"   📊 Val:   {X_val.shape[0]} samples") 
    print(f"   📊 Test:  {X_test.shape[0]} samples")
    
    # 5. Train the model
    print("\n5️⃣ Training model...")
    result = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   ✅ Training completed!")
    print(f"   📈 Final train loss: {result.get('final_train_loss', 'N/A')}")
    print(f"   📈 Final val loss: {result.get('final_val_loss', 'N/A')}")
    print(f"   ⏱️ Training time: {result.get('training_time', 0):.2f}s")
    
    # 6. Test the model
    print("\n6️⃣ Testing model...")
    predictions = model.predict(X_test)
    test_mse = np.mean((y_test - predictions) ** 2)
    
    print(f"   📊 Test MSE: {test_mse:.6f}")
    
    # 7. Save the model
    print("\n7️⃣ Saving model...")
    save_path = Path("experiments/results/quick_example")
    model.save(save_path)
    preprocessing.save(save_path / "preprocessing")
    
    print(f"   💾 Model saved to: {save_path}")
    
    print("\n🎉 Example completed successfully!")
    print("="*50)
    
    return {
        'test_mse': test_mse,
        'training_time': result.get('training_time', 0),
        'model_path': save_path
    }


if __name__ == "__main__":
    try:
        result = quick_mlp_example()
        print(f"\nFinal Results: MSE={result['test_mse']:.6f}, Time={result['training_time']:.2f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()