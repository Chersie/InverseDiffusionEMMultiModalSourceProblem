#!/usr/bin/env python3
"""
Quick test to validate the fixed training pipeline.
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.registry import create_mlp
from src.core.data_generator import DataGenerator, pack_coefficients
from src.api.preprocessing import PreprocessingPipeline

def test_dimension_matching():
    """Test that model dimensions match preprocessing output."""
    print("🧪 Testing dimension matching fix...")
    
    # 1. Generate small dataset
    generator = DataGenerator.for_ml_training()
    dataset = generator.generate_batch(maxorder=3, n_samples=50)
    
    # 2. Setup preprocessing
    preprocessing = PreprocessingPipeline()
    E_theta = dataset['amplitude'][..., 0]
    E_phi = dataset['amplitude'][..., 1]
    a_e = dataset['coefficients_e']
    a_m = dataset['coefficients_m']
    
    targets = pack_coefficients(a_e, a_m)
    preprocessing.fit(E_theta, E_phi, targets=targets)
    X = preprocessing.transform_features(E_theta, E_phi)
    y = preprocessing.process_coefficients(a_e, a_m)
    
    print(f"   📊 Preprocessed data: X={X.shape}, y={y.shape}")
    
    # 3. Create model with CORRECT dimensions
    from src.models.registry import get_model_registry
    registry = get_model_registry()
    
    model = registry.create_model(
        model_type="mlp",
        input_dim=X.shape[1],  # Use actual input dimension
        output_dim=y.shape[1], # Use actual output dimension
        hidden_size=64,
        n_hidden_layers=2,
        epochs=3,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"   ✅ Created model: {model}")
    
    # 4. Test training (should work without dimension errors)
    start_time = time.time()
    result = model.fit(X[:35], y[:35], X[35:40], y[35:40])  # Small train/val split
    train_time = time.time() - start_time
    
    print(f"   ✅ Training successful in {train_time:.2f}s!")
    print(f"   📈 Final train loss: {result.get('final_train_loss', 'N/A')}")
    
    # 5. Test prediction
    predictions = model.predict(X[40:])
    print(f"   ✅ Predictions shape: {predictions.shape}")
    
    print("🎉 All tests passed! Dimension matching is fixed.")
    return True

if __name__ == "__main__":
    try:
        test_dimension_matching()
        print("\n✅ Fix validated successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()