#!/usr/bin/env python3
"""
Example usage of ML Embedded Compiler
Demonstrates various compilation scenarios
"""

import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from compiler import (
    MLEmbeddedCompiler, ModelConfig, FrameworkType,
    ARMTarget, OptimizationStrategy
)
from benchmark import run_benchmark_suite
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Example Models ====================

class SimpleImageClassifier(nn.Module):
    """Simple CNN for image classification"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetLike(nn.Module):
    """Lightweight mobile-friendly model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.depthwise = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.depthwise(x)))
        x = torch.relu(self.bn3(self.pointwise(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== Example 1: Basic Compilation ====================

def example_1_basic_compilation():
    """Basic model compilation with default settings"""
    logger.info("\n" + "="*60)
    logger.info("Example 1: Basic Model Compilation")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        # Configure compilation
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=0.0,
            target=ARMTarget.ARMV8_NEON,
            optimization_strategy=OptimizationStrategy.BALANCED
        )
        
        # Compile
        compiler = MLEmbeddedCompiler(config)
        result = compiler.compile(model_path, os.path.join(tmpdir, 'output'))
        
        logger.info("\nResults:")
        logger.info(f"  Original Size: {result['original_size_mb']:.2f} MB")
        logger.info(f"  Optimized Size: {result['optimized_size_mb']:.2f} MB")
        logger.info(f"  Compression: {result['compression_ratio']:.2f}x")


# ==================== Example 2: Aggressive Optimization ====================

def example_2_aggressive_optimization():
    """Aggressive optimization with maximum compression"""
    logger.info("\n" + "="*60)
    logger.info("Example 2: Aggressive Optimization")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        # Configure for aggressive optimization
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=30.0,  # 30% pruning
            target=ARMTarget.CORTEX_M7,
            optimization_strategy=OptimizationStrategy.AGGRESSIVE
        )
        
        # Compile
        compiler = MLEmbeddedCompiler(config)
        result = compiler.compile(model_path, os.path.join(tmpdir, 'output'))
        
        logger.info("\nResults:")
        logger.info(f"  Quantization: INT8")
        logger.info(f"  Pruning: 30%")
        logger.info(f"  Compression Ratio: {result['compression_ratio']:.2f}x")


# ==================== Example 3: Mobile-Friendly Model ====================

def example_3_mobile_model():
    """Compile mobile-friendly model"""
    logger.info("\n" + "="*60)
    logger.info("Example 3: Mobile-Friendly Model Compilation")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save lightweight model
        model = MobileNetLike()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        # Compile for ARM v7 (common in mobile devices)
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=20.0,
            target=ARMTarget.ARMV7,
            optimization_strategy=OptimizationStrategy.BALANCED
        )
        
        # Compile
        compiler = MLEmbeddedCompiler(config)
        result = compiler.compile(model_path, os.path.join(tmpdir, 'output'))
        
        logger.info("\nResults:")
        logger.info(f"  Target: ARMv7 (Mobile)")
        logger.info(f"  Compression Ratio: {result['compression_ratio']:.2f}x")


# ==================== Example 4: Different ARM Targets ====================

def example_4_different_arm_targets():
    """Compile same model for different ARM targets"""
    logger.info("\n" + "="*60)
    logger.info("Example 4: Multiple ARM Targets")
    logger.info("="*60 + "\n")
    
    targets = [
        ARMTarget.ARMV7,
        ARMTarget.ARMV8_NEON,
        ARMTarget.CORTEX_M4,
        ARMTarget.CORTEX_M7,
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model once
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        logger.info("\nCompiling for different ARM targets:\n")
        
        for target in targets:
            logger.info(f"Compiling for {target.value}...")
            
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                quantization="int8",
                pruning=10.0,
                target=target,
                optimization_strategy=OptimizationStrategy.BALANCED
            )
            
            compiler = MLEmbeddedCompiler(config)
            output_path = os.path.join(tmpdir, f'output_{target.value}')
            result = compiler.compile(model_path, output_path)
            
            logger.info(f"  ✓ {target.value}: {result['compression_ratio']:.2f}x compression\n")


# ==================== Example 5: Conservative Optimization ====================

def example_5_conservative_optimization():
    """Conservative optimization for maximum compatibility"""
    logger.info("\n" + "="*60)
    logger.info("Example 5: Conservative Optimization")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        # Configure for conservative (compatible) optimization
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="none",  # No quantization
            pruning=0.0,  # No pruning
            target=ARMTarget.ARMV8,
            optimization_strategy=OptimizationStrategy.CONSERVATIVE
        )
        
        # Compile
        compiler = MLEmbeddedCompiler(config)
        result = compiler.compile(model_path, os.path.join(tmpdir, 'output'))
        
        logger.info("\nResults (No Optimizations):")
        logger.info(f"  Compression Ratio: {result['compression_ratio']:.2f}x")
        logger.info("  Note: Conservative mode prioritizes compatibility over size")


# ==================== Example 6: Comparison ====================

def example_6_optimization_comparison():
    """Compare different optimization strategies on the same model"""
    logger.info("\n" + "="*60)
    logger.info("Example 6: Optimization Strategy Comparison")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        strategies = [
            ("none", 0.0, "No Optimization"),
            ("float16", 0.0, "FP16 Only"),
            ("int8", 0.0, "INT8 Only"),
            ("int8", 10.0, "INT8 + 10% Pruning"),
            ("int8", 20.0, "INT8 + 20% Pruning"),
            ("int8", 30.0, "INT8 + 30% Pruning"),
        ]
        
        logger.info("Quantization | Pruning | Compression | Description")
        logger.info("-" * 60)
        
        for quant, pruning, description in strategies:
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                quantization=quant,
                pruning=pruning,
                target=ARMTarget.ARMV8_NEON
            )
            
            compiler = MLEmbeddedCompiler(config)
            output_path = os.path.join(tmpdir, f'output_{quant}_{pruning}')
            result = compiler.compile(model_path, output_path)
            
            logger.info(f"{quant:12} | {pruning:7.1f}% | {result['compression_ratio']:11.2f}x | {description}")


# ==================== Example 7: Full Benchmark ====================

def example_7_full_benchmark():
    """Run full benchmark suite"""
    logger.info("\n" + "="*60)
    logger.info("Example 7: Full Benchmark Suite")
    logger.info("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        model = SimpleImageClassifier()
        model_path = os.path.join(tmpdir, 'model.pth')
        torch.save(model, model_path)
        
        # Run full benchmark
        benchmark_results = run_benchmark_suite(
            model,
            model_path,
            os.path.join(tmpdir, 'benchmarks')
        )
        
        logger.info("\nBenchmark Results:")
        logger.info(json.dumps(benchmark_results, indent=2, default=str))


# ==================== Main ====================

def main():
    """Run all examples"""
    logger.info("\n" + "="*70)
    logger.info("ML Embedded Compiler - Example Usage")
    logger.info("="*70)
    
    examples = [
        ("1", "Basic Compilation", example_1_basic_compilation),
        ("2", "Aggressive Optimization", example_2_aggressive_optimization),
        ("3", "Mobile-Friendly Model", example_3_mobile_model),
        ("4", "Different ARM Targets", example_4_different_arm_targets),
        ("5", "Conservative Optimization", example_5_conservative_optimization),
        ("6", "Optimization Comparison", example_6_optimization_comparison),
    ]
    
    logger.info("\nAvailable Examples:")
    for num, name, _ in examples:
        logger.info(f"  {num}. {name}")
    
    # Run all examples
    for num, name, func in examples:
        try:
            func()
        except Exception as e:
            logger.error(f"Example {num} failed: {e}", exc_info=True)
    
    logger.info("\n" + "="*70)
    logger.info("All Examples Completed!")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    import json
    main()
