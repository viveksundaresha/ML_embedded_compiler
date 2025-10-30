"""
Unit tests for ML Embedded Compiler
"""

import unittest
import tempfile
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from compiler import (
    ModelConfig, ModelOptimizer, CodeGenerator, MLEmbeddedCompiler,
    FrameworkType, ARMTarget, OptimizationStrategy
)


class SimpleModel(torch.nn.Module):
    """Simple PyTorch model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(224 * 224 * 3, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig"""
    
    def test_config_creation(self):
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=20.0,
            target=ARMTarget.ARMV8_NEON
        )
        
        self.assertEqual(config.framework, FrameworkType.PYTORCH)
        self.assertEqual(config.input_shape, (1, 3, 224, 224))
        self.assertEqual(config.quantization, "int8")
        self.assertEqual(config.pruning, 20.0)
        self.assertEqual(config.target, ARMTarget.ARMV8_NEON)


class TestModelOptimizer(unittest.TestCase):
    """Test ModelOptimizer"""
    
    def setUp(self):
        self.config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=20.0
        )
        self.optimizer = ModelOptimizer(self.config)
    
    def test_pytorch_model_loading(self):
        """Test loading PyTorch model"""
        # Create and save a simple model
        model = SimpleModel()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)  # Save only weights, not the class
            model_path = f.name
        
        try:
            # Load with a fresh model instance
            loaded_model = SimpleModel()
            loaded_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
            self.assertIsNotNone(loaded_model)
            self.assertTrue(hasattr(loaded_model, 'fc1'))
        except TypeError:
            # Fallback for older PyTorch versions
            loaded_model = SimpleModel()
            loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.assertIsNotNone(loaded_model)
        finally:
            Path(model_path).unlink()
    
    def test_quantization_none(self):
        """Test no quantization"""
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="none"
        )
        optimizer = ModelOptimizer(config)
        optimizer.model = SimpleModel()
        
        result = optimizer.quantize_pytorch()
        self.assertIsNotNone(result)
    
    def test_model_size_calculation(self):
        """Test model size calculation"""
        self.optimizer.model = SimpleModel()
        sizes = self.optimizer.get_model_size()
        
        self.assertIn('total_parameters', sizes)
        self.assertIn('estimated_size_mb', sizes)
        self.assertGreater(sizes['total_parameters'], 0)
        self.assertGreater(sizes['estimated_size_mb'], 0)


class TestCodeGenerator(unittest.TestCase):
    """Test CodeGenerator"""
    
    def setUp(self):
        self.config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            target=ARMTarget.ARMV8_NEON
        )
        self.generator = CodeGenerator(self.config)
    
    def test_header_generation(self):
        """Test C++ header generation"""
        header = self.generator.generate_inference_header()
        
        self.assertIn('#ifndef ML_INFERENCE_H', header)
        self.assertIn('#endif', header)
        self.assertIn('ml_inference', header)
        self.assertIn('InferenceEngine', header)
        self.assertIn('armv8_neon', header)
    
    def test_source_generation(self):
        """Test C++ source generation"""
        source = self.generator.generate_inference_source()
        
        self.assertIn('namespace ml_inference', source)
        self.assertIn('InferenceEngine::', source)
        self.assertIn('loadWeights', source)
        self.assertIn('infer', source)
    
    def test_cmake_generation(self):
        """Test CMakeLists.txt generation"""
        cmake = self.generator.generate_cmake()
        
        self.assertIn('cmake_minimum_required', cmake)
        self.assertIn('project(ml_inference)', cmake)
        self.assertIn('add_library(ml_inference', cmake)
        self.assertIn('armv8_neon', cmake)
    
    def test_activation_functions(self):
        """Test activation functions generation"""
        funcs = self.generator._generate_activation_functions()
        
        self.assertIn('relu', funcs)
        self.assertIn('sigmoid', funcs)
        self.assertIn('tanh_approx', funcs)
        self.assertIn('gelu_approx', funcs)
    
    def test_quantization_functions_int8(self):
        """Test INT8 quantization functions"""
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8"
        )
        generator = CodeGenerator(config)
        funcs = generator._generate_quantization_functions()
        
        self.assertIn('quantize_int8', funcs)
        self.assertIn('dequantize_int8', funcs)
        self.assertIn('matmul_int8', funcs)
    
    def test_quantization_functions_float(self):
        """Test float quantization functions"""
        config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="none"
        )
        generator = CodeGenerator(config)
        funcs = generator._generate_quantization_functions()
        
        self.assertIn('matmul_float', funcs)
    
    def test_weight_extraction(self):
        """Test weight extraction"""
        model = SimpleModel()
        self.generator.extract_weights(model)
        
        self.assertGreater(len(self.generator.weights), 0)
        self.assertIn('fc1.weight', self.generator.weights)


class TestMLEmbeddedCompiler(unittest.TestCase):
    """Test MLEmbeddedCompiler"""
    
    def setUp(self):
        self.config = ModelConfig(
            framework=FrameworkType.PYTORCH,
            input_shape=(1, 3, 224, 224),
            quantization="int8",
            pruning=10.0,
            target=ARMTarget.ARMV8_NEON
        )
    
    def test_compiler_initialization(self):
        """Test compiler initialization"""
        compiler = MLEmbeddedCompiler(self.config)
        
        self.assertIsNotNone(compiler.optimizer)
        self.assertIsNotNone(compiler.code_generator)
    
    def test_full_compilation_pipeline(self):
        """Test full compilation pipeline"""
        # Create and save model
        model = SimpleModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pth')
            torch.save(model, model_path)
            
            output_dir = os.path.join(tmpdir, 'output')
            
            compiler = MLEmbeddedCompiler(self.config)
            result = compiler.compile(model_path, output_dir)
            
            # Check output files
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'include', 'ml_inference.h')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'src', 'ml_inference.cpp')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'CMakeLists.txt')))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'config.json')))
            
            # Check result
            self.assertIn('original_size_mb', result)
            self.assertIn('optimized_size_mb', result)
            self.assertIn('compression_ratio', result)
            self.assertGreater(result['compression_ratio'], 0)


class TestARMTargets(unittest.TestCase):
    """Test ARM target support"""
    
    def test_all_arm_targets(self):
        """Test all ARM targets"""
        targets = [
            ARMTarget.ARMV7,
            ARMTarget.ARMV8,
            ARMTarget.ARMV8_NEON,
            ARMTarget.CORTEX_M4,
            ARMTarget.CORTEX_M7,
        ]
        
        for target in targets:
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                target=target
            )
            generator = CodeGenerator(config)
            header = generator.generate_inference_header()
            
            self.assertIn(target.value, header)


class TestOptimizationStrategies(unittest.TestCase):
    """Test optimization strategies"""
    
    def test_all_optimization_strategies(self):
        """Test all optimization strategies"""
        strategies = [
            OptimizationStrategy.AGGRESSIVE,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.CONSERVATIVE,
        ]
        
        for strategy in strategies:
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                optimization_strategy=strategy
            )
            generator = CodeGenerator(config)
            header = generator.generate_inference_header()
            
            self.assertIn(strategy.value, header)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
