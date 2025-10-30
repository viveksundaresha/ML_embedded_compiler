#!/usr/bin/env python3
"""
Quick test - Creates a model and compiles it
Run this file to test the compiler end-to-end
"""

import torch
import torch.nn as nn
import os
import sys

# Fix Windows encoding issues
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("ML Embedded Compiler - Quick Test")
print("="*60)

# Step 1: Create a simple model
print("\n[STEP 1] Creating simple model...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 112 * 112, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleModel()
model_path = 'test_model.pth'
torch.save(model, model_path)  # Save full model, not just state_dict
print(f"[OK] Model created and saved as '{model_path}'")

# Step 2: Import compiler
print("\n[STEP 2] Importing compiler...")
try:
    from compiler import MLEmbeddedCompiler, ModelConfig, FrameworkType, ARMTarget
    print("[OK] Compiler imported successfully")
except ImportError as e:
    print(f"[ERROR] Error importing compiler: {e}")
    print("Make sure compiler.py is in the same directory")
    sys.exit(1)

# Step 3: Configure compilation
print("\n[STEP 3] Configuring compilation...")
config = ModelConfig(
    framework=FrameworkType.PYTORCH,
    input_shape=(1, 3, 224, 224),
    quantization="int8",
    pruning=10.0,
    target=ARMTarget.ARMV8_NEON
)
print(f"[OK] Configuration created")
print(f"   - Framework: PyTorch")
print(f"   - Input Shape: (1, 3, 224, 224)")
print(f"   - Quantization: INT8")
print(f"   - Pruning: 10%")
print(f"   - Target: ARMv8 with NEON")

# Step 4: Compile
print("\n[STEP 4] Compiling model...")
print("-" * 60)

try:
    compiler = MLEmbeddedCompiler(config)
    result = compiler.compile(model_path, './build')
    
    print("-" * 60)
    print("[OK] Compilation successful!")
    print(f"\n[RESULTS]")
    print(f"   - Original Size: {result['original_size_mb']:.2f} MB")
    print(f"   - Optimized Size: {result['optimized_size_mb']:.2f} MB")
    print(f"   - Compression: {result['compression_ratio']:.2f}x")
    
    print(f"\n[OUTPUT FILES] in './build/':")
    if os.path.exists('./build'):
        for file in os.listdir('./build'):
            filepath = os.path.join('./build', file)
            size = os.path.getsize(filepath)
            print(f"   - {file} ({size} bytes)")
    
    print("\n" + "="*60)
    print("[SUCCESS] TEST COMPLETED!")
    print("="*60)
    print("\nYour compiler is working! Next steps:")
    print("1. Try with your own model:")
    print("   python cli.py --framework pytorch --model your_model.pth \\")
    print("     --output ./my_build --input-shape 1,3,224,224 \\")
    print("     --quantization int8 --target armv8_neon")
    print("\n2. View generated C++ code:")
    print("   cat ./build/ml_inference.h")
    
except Exception as e:
    print("-" * 60)
    print(f"[ERROR] Compilation failed: {e}")
    print(f"\nError details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
