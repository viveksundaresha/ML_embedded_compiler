"""
ML Embedded Compiler - Main compilation engine
Handles model loading, optimization, and C++ code generation
"""

import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import logging
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkType(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class ARMTarget(Enum):
    ARMV7 = "armv7"
    ARMV8 = "armv8"
    ARMV8_NEON = "armv8_neon"
    CORTEX_M4 = "cortex_m4"
    CORTEX_M7 = "cortex_m7"


class OptimizationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class ModelConfig:
    """Configuration for model optimization"""
    framework: FrameworkType
    input_shape: Tuple[int, ...]
    quantization: str = "int8"  # 'none', 'int8', 'float16'
    pruning: float = 0.0  # percentage of weights to prune
    target: ARMTarget = ARMTarget.ARMV8_NEON
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED


class ModelOptimizer:
    """Handles model optimization operations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.optimization_stats = {}
    
    def load_pytorch_model(self, model_path: str):
        """Load PyTorch model"""
        logger.info(f"Loading PyTorch model from {model_path}")
        try:
            # Try with weights_only=False for compatibility with PyTorch 2.6+
            try:
                self.model = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't have weights_only parameter
                self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
            logger.info("PyTorch model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def load_tensorflow_model(self, model_path: str):
        """Load TensorFlow model"""
        logger.info(f"Loading TensorFlow model from {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info("TensorFlow model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise
    
    def quantize_pytorch(self) -> torch.nn.Module:
        """Apply quantization to PyTorch model"""
        logger.info(f"Applying {self.config.quantization} quantization to PyTorch model")
        
        if self.config.quantization == "none":
            return self.model
        
        elif self.config.quantization == "int8":
            logger.info("Applying INT8 quantization")
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            self.optimization_stats['pytorch_quantization'] = 'int8'
            return quantized_model
        
        elif self.config.quantization == "float16":
            logger.info("Applying FP16 quantization")
            for module in self.model.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    module.half()
            self.optimization_stats['pytorch_quantization'] = 'float16'
            return self.model
    
    def prune_pytorch(self) -> torch.nn.Module:
        """Apply structured pruning to PyTorch model"""
        if self.config.pruning == 0.0:
            logger.info("No pruning applied")
            return self.model
        
        logger.info(f"Applying {self.config.pruning}% structured pruning")
        
        try:
            # Try to use the prune module if available
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.utils.prune.ln_structured(
                        module, name='weight',
                        amount=self.config.pruning / 100.0,
                        n=2, dim=0
                    )
        except (AttributeError, ImportError):
            # If prune module not available, use manual weight zeroing
            logger.warning("Prune module not available, using manual weight zeroing instead")
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    with torch.no_grad():
                        # Manually zero out the smallest weights
                        weights = module.weight.data.view(-1)
                        num_to_prune = int(len(weights) * (self.config.pruning / 100.0))
                        if num_to_prune > 0:
                            threshold = torch.topk(torch.abs(weights), num_to_prune, largest=False)[0][-1]
                            module.weight.data[torch.abs(module.weight.data) <= threshold] = 0
        
        self.optimization_stats['pruning_percentage'] = self.config.pruning
        return self.model
    
    def quantize_tensorflow(self) -> tf.keras.Model:
        """Apply quantization to TensorFlow model"""
        logger.info(f"Applying {self.config.quantization} quantization to TensorFlow model")
        
        if self.config.quantization == "none":
            return self.model
        
        elif self.config.quantization == "int8":
            logger.info("Applying INT8 quantization via quantization-aware training")
            return self._apply_qat_tf(self.model)
        
        elif self.config.quantization == "float16":
            logger.info("Applying FP16 quantization")
            return self._apply_mixed_precision_tf(self.model)
        
        return self.model
    
    @staticmethod
    def _apply_qat_tf(model: tf.keras.Model) -> tf.keras.Model:
        """Apply Quantization-Aware Training for TensorFlow"""
        import tensorflow_model_optimization as tfmot
        
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        return q_aware_model
    
    @staticmethod
    def _apply_mixed_precision_tf(model: tf.keras.Model) -> tf.keras.Model:
        """Apply mixed precision training"""
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        return model
    
    def optimize(self) -> Union[torch.nn.Module, tf.keras.Model]:
        """Apply all optimizations"""
        logger.info("Starting model optimization pipeline")
        
        if self.config.framework == FrameworkType.PYTORCH:
            self.model = self.quantize_pytorch()
            self.model = self.prune_pytorch()
        elif self.config.framework == FrameworkType.TENSORFLOW:
            self.model = self.quantize_tensorflow()
        
        logger.info("Model optimization completed")
        return self.model
    
    def get_model_size(self) -> Dict[str, float]:
        """Calculate model sizes before and after optimization"""
        sizes = {}
        
        if isinstance(self.model, torch.nn.Module):
            # PyTorch model size
            param_count = sum(p.numel() for p in self.model.parameters())
            sizes['total_parameters'] = param_count
            sizes['estimated_size_mb'] = (param_count * 4) / (1024 * 1024)  # Assume 4 bytes per param
        
        elif isinstance(self.model, tf.keras.Model):
            # TensorFlow model size
            param_count = self.model.count_params()
            sizes['total_parameters'] = param_count
            sizes['estimated_size_mb'] = (param_count * 4) / (1024 * 1024)
        
        return sizes


class CodeGenerator:
    """Generates optimized C++ inference code"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.weights = {}
        self.layer_configs = []
    
    def extract_weights(self, model: Union[torch.nn.Module, tf.keras.Model]):
        """Extract weights from model"""
        logger.info("Extracting model weights")
        
        if isinstance(model, torch.nn.Module):
            for name, param in model.named_parameters():
                self.weights[name] = param.data.cpu().numpy()
        
        elif isinstance(model, tf.keras.Model):
            for layer in model.layers:
                if layer.weights:
                    self.weights[layer.name] = [w.numpy() for w in layer.weights]
        
        logger.info(f"Extracted {len(self.weights)} weight tensors")
    
    def generate_inference_header(self) -> str:
        """Generate C++ header file for inference"""
        target_flags = self._get_target_flags()
        
        header = f'''#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#include <vector>
#include <array>
#include <cstring>
#include <cmath>
#include <algorithm>

// ARM Target: {self.config.target.value}
// Optimization Level: {self.config.optimization_strategy.value}
// Quantization: {self.config.quantization}
// Compiler Flags: {target_flags}

namespace ml_inference {{

// ==================== Data Types ====================

{self._generate_data_types()}

// ==================== Model Configuration ====================

{self._generate_model_config()}

// ==================== Inference Engine ====================

class InferenceEngine {{
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Initialize weights from external storage
    bool loadWeights(const uint8_t* weights_data, size_t size);
    
    // Main inference function
    bool infer(const float* input, float* output);
    
    // Get model input/output shapes
    const std::vector<size_t>& getInputShape() const {{ return input_shape_; }}
    const std::vector<size_t>& getOutputShape() const {{ return output_shape_; }}
    
    // Performance monitoring
    float getLastInferenceTime() const {{ return last_inference_time_; }}
    size_t getWeightsMemory() const {{ return weights_memory_; }}

private:
    // Layer implementations
{self._generate_layer_declarations()}
    
    // Internal state
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape_;
    float last_inference_time_;
    size_t weights_memory_;
    bool weights_loaded_;
}};

}} // namespace ml_inference

#endif // ML_INFERENCE_H
'''
        return header
    
    def generate_inference_source(self) -> str:
        """Generate C++ source file for inference"""
        source = f'''#include "ml_inference.h"
#include <chrono>
#include <cstring>

namespace ml_inference {{

// ==================== Activation Functions ====================

{self._generate_activation_functions()}

// ==================== Quantization Utilities ====================

{self._generate_quantization_functions()}

// ==================== InferenceEngine Implementation ====================

InferenceEngine::InferenceEngine()
    : last_inference_time_(0.0f),
      weights_memory_(0),
      weights_loaded_(false)
{{
    input_shape_ = {{{', '.join(map(str, self.config.input_shape))}}};
{self._generate_output_shape_init()}
}}

InferenceEngine::~InferenceEngine() {{
    // Cleanup if needed
}}

bool InferenceEngine::loadWeights(const uint8_t* weights_data, size_t size) {{
    if (!weights_data || size == 0) {{
        return false;
    }}
    weights_loaded_ = true;
    weights_memory_ = size;
    return true;
}}

bool InferenceEngine::infer(const float* input, float* output) {{
    if (!weights_loaded_ || !input || !output) {{
        return false;
    }}
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
{self._generate_forward_pass()}
    
    auto end = std::chrono::high_resolution_clock::now();
    last_inference_time_ = std::chrono::duration<float, std::milli>(end - start).count();
    
    return true;
}}

}} // namespace ml_inference
'''
        return source
    
    def generate_cmake(self) -> str:
        """Generate CMakeLists.txt for building"""
        target_flags = self._get_target_flags()
        
        cmake = f'''cmake_minimum_required(VERSION 3.10)
project(ml_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target-specific flags
set(ARM_TARGET "{self.config.target.value}")
set(ARM_FLAGS "{target_flags}")

if(ARM_TARGET STREQUAL "cortex_m4")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=softfp")
elseif(ARM_TARGET STREQUAL "cortex_m7")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=softfp")
elseif(ARM_TARGET STREQUAL "armv8_neon")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -march=armv8-a+simd -mtune=cortex-a72")
elseif(ARM_TARGET STREQUAL "armv8")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -march=armv8-a")
elseif(ARM_TARGET STREQUAL "armv7")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -march=armv7-a -mfpu=neon -mfloat-abi=softfp")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -O3 -ffast-math -funroll-loops")

# Library
add_library(ml_inference STATIC
    src/ml_inference.cpp
)

target_include_directories(ml_inference PUBLIC include)

# Test executable
add_executable(inference_test
    test/inference_test.cpp
)

target_link_libraries(inference_test ml_inference)

# Benchmark executable
add_executable(benchmark
    test/benchmark.cpp
)

target_link_libraries(benchmark ml_inference)
'''
        return cmake
    
    def _get_target_flags(self) -> str:
        """Get compiler flags for target"""
        flags = {
            ARMTarget.ARMV7: "-march=armv7-a -mfpu=neon",
            ARMTarget.ARMV8: "-march=armv8-a",
            ARMTarget.ARMV8_NEON: "-march=armv8-a+simd",
            ARMTarget.CORTEX_M4: "-mcpu=cortex-m4 -mthumb",
            ARMTarget.CORTEX_M7: "-mcpu=cortex-m7 -mthumb",
        }
        return flags.get(self.config.target, "")
    
    def _generate_data_types(self) -> str:
        """Generate data type definitions"""
        if self.config.quantization == "int8":
            return '''using WeightType = int8_t;
using ActivationType = int32_t;

struct QuantizationParams {
    float scale;
    int32_t zero_point;
};
'''
        elif self.config.quantization == "float16":
            return '''using WeightType = uint16_t;  // float16
using ActivationType = float;
'''
        else:
            return '''using WeightType = float;
using ActivationType = float;
'''
    
    def _generate_model_config(self) -> str:
        """Generate model configuration constants"""
        return f'''constexpr size_t INPUT_SIZE = {np.prod(self.config.input_shape)};
constexpr size_t BATCH_SIZE = {self.config.input_shape[0] if self.config.input_shape else 1};
constexpr const char* MODEL_NAME = "optimized_model_{self.config.quantization}";
'''
    
    def _generate_activation_functions(self) -> str:
        """Generate optimized activation functions"""
        return '''// ReLU activation
inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Optimized tanh
inline float tanh_approx(float x) {
    // Using Padé approximation for faster computation
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Sigmoid
inline float sigmoid(float x) {
    if (x < -45.0f) return 0.0f;
    if (x > 45.0f) return 1.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// GELU approximation
inline float gelu_approx(float x) {
    const float cdf = 0.5f * (1.0f + tanh_approx(0.7978845608f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}
'''
    
    def _generate_quantization_functions(self) -> str:
        """Generate quantization/dequantization utilities"""
        if self.config.quantization == "int8":
            return '''// Quantize float to int8
inline int8_t quantize_int8(float value, float scale, int32_t zero_point) {
    int32_t quantized = static_cast<int32_t>(std::round(value / scale)) + zero_point;
    return static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
}

// Dequantize int8 to float
inline float dequantize_int8(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<int32_t>(value) - zero_point) * scale;
}

// Optimized int8 matrix multiplication
void matmul_int8(const int8_t* A, const int8_t* B, int32_t* C,
                 size_t M, size_t N, size_t K,
                 float scale_A, float scale_B, float scale_C) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
            }
            C[m * N + n] = sum;
        }
    }
}
'''
        else:
            return '''// Optimized float matrix multiplication
void matmul_float(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}
'''
    
    def _generate_layer_declarations(self) -> str:
        """Generate layer implementation declarations"""
        return '''    // Conv2D layers
    void conv2d_layer_1(const float* input, float* output);
    
    // Linear layers
    void linear_layer_1(const float* input, float* output);
    
    // Intermediate buffers
    std::vector<float> buffer_1;
    std::vector<float> buffer_2;
'''
    
    def _generate_output_shape_init(self) -> str:
        """Generate output shape initialization"""
        return "    output_shape_ = {1, 1000};  // Update based on your model"
    
    def _generate_forward_pass(self) -> str:
        """Generate forward pass implementation"""
        return '''    // Layer 1: Input normalization
    std::vector<float> normalized_input(INPUT_SIZE);
    for (size_t i = 0; i < INPUT_SIZE; ++i) {
        normalized_input[i] = input[i] / 255.0f;  // Normalize to [0,1]
    }
    
    // Layer 2: First conv layer
    conv2d_layer_1(normalized_input.data(), buffer_1.data());
    
    // Layer 3: Linear output layer
    linear_layer_1(buffer_1.data(), output);
    
    return true;
'''


class MLEmbeddedCompiler:
    """Main compiler orchestrating the entire process"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.optimizer = ModelOptimizer(config)
        self.code_generator = CodeGenerator(config)
    
    def compile(self, model_path: str, output_dir: str) -> Dict:
        """Compile model end-to-end"""
        logger.info(f"Starting ML model compilation from {model_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if self.config.framework == FrameworkType.PYTORCH:
            self.optimizer.load_pytorch_model(model_path)
        else:
            self.optimizer.load_tensorflow_model(model_path)
        
        # Get original size
        original_size = self.optimizer.get_model_size()
        logger.info(f"Original model size: {original_size}")
        
        # Optimize
        optimized_model = self.optimizer.optimize()
        
        # Get optimized size
        optimized_size = self.optimizer.get_model_size()
        logger.info(f"Optimized model size: {optimized_size}")
        
        # Extract weights
        self.code_generator.extract_weights(optimized_model)
        
        # Generate code
        logger.info("Generating C++ code")
        header_code = self.code_generator.generate_inference_header()
        source_code = self.code_generator.generate_inference_source()
        cmake_code = self.code_generator.generate_cmake()
        
        # Write files
        include_dir = output_dir / "include"
        src_dir = output_dir / "src"
        include_dir.mkdir(exist_ok=True)
        src_dir.mkdir(exist_ok=True)
        
        with open(include_dir / "ml_inference.h", "w") as f:
            f.write(header_code)
        
        with open(src_dir / "ml_inference.cpp", "w") as f:
            f.write(source_code)
        
        with open(output_dir / "CMakeLists.txt", "w") as f:
            f.write(cmake_code)
        
        # Save configuration
        original_mb = original_size.get('estimated_size_mb', 1)
        optimized_mb = optimized_size.get('estimated_size_mb', 1)
        
        # Avoid division by zero
        if optimized_mb == 0:
            optimized_mb = 1
        
        config_data = {
            "framework": self.config.framework.value,
            "quantization": self.config.quantization,
            "pruning": self.config.pruning,
            "target": self.config.target.value,
            "optimization_strategy": self.config.optimization_strategy.value,
            "original_size_mb": original_mb,
            "optimized_size_mb": optimized_mb,
            "compression_ratio": original_mb / optimized_mb,
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Compilation completed. Output saved to {output_dir}")
        
        return config_data


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(
        framework=FrameworkType.PYTORCH,
        input_shape=(1, 3, 224, 224),
        quantization="int8",
        pruning=20.0,
        target=ARMTarget.ARMV8_NEON,
        optimization_strategy=OptimizationStrategy.BALANCED
    )
    
    compiler = MLEmbeddedCompiler(config)
    logger.info("ML Embedded Compiler initialized successfully")
