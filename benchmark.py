"""
Benchmark utilities for ML Embedded Compiler
"""

import torch
import time
import logging
from typing import Dict, Tuple
from compiler import (
    MLEmbeddedCompiler, ModelConfig, FrameworkType,
    ARMTarget, OptimizationStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompilerBenchmark:
    """Benchmark compiler performance"""
    
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        self.results = {}
    
    def measure_compilation_time(self, model_path: str, output_dir: str) -> float:
        """Measure compilation time"""
        logger.info("Measuring compilation time...")
        start = time.time()
        
        compiler = MLEmbeddedCompiler(self.config)
        compiler.compile(model_path, output_dir)
        
        elapsed = time.time() - start
        logger.info(f"Compilation time: {elapsed:.2f}s")
        
        self.results['compilation_time'] = elapsed
        return elapsed
    
    def measure_model_inference(self) -> float:
        """Measure original model inference time"""
        logger.info("Measuring original model inference...")
        
        input_data = torch.randn(self.config.input_shape)
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            self.model(input_data)
        
        # Measure
        iterations = 10
        start = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                self.model(input_data)
        
        elapsed = (time.time() - start) / iterations
        logger.info(f"Inference time per iteration: {elapsed*1000:.2f}ms")
        
        self.results['inference_time_ms'] = elapsed * 1000
        return elapsed
    
    def get_model_complexity(self) -> Dict:
        """Calculate model complexity metrics"""
        logger.info("Calculating model complexity...")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate FLOPs
        macs = self._estimate_macs()
        
        complexity = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_flops': macs * 2,  # MACs * 2 = FLOPs
        }
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Estimated FLOPs: {macs*2:,.0f}")
        
        self.results['complexity'] = complexity
        return complexity
    
    def _estimate_macs(self) -> float:
        """Estimate MACs (Multiply-Accumulate operations)"""
        macs = 0
        input_size = 1
        for dim in self.config.input_shape:
            input_size *= dim
        
        for param in self.model.parameters():
            macs += param.numel()
        
        return macs * input_size


class OptimizationComparison:
    """Compare different optimization strategies"""
    
    @staticmethod
    def compare_quantization_strategies(model, model_path: str, 
                                       output_base: str) -> Dict:
        """Compare different quantization strategies"""
        logger.info("\n" + "="*60)
        logger.info("Comparing Quantization Strategies")
        logger.info("="*60 + "\n")
        
        strategies = {
            'none': 'No Quantization',
            'float16': 'FP16 Quantization',
            'int8': 'INT8 Quantization',
        }
        
        results = {}
        
        for quant_type, description in strategies.items():
            logger.info(f"Testing: {description}")
            logger.info("-" * 40)
            
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                quantization=quant_type,
                pruning=0.0
            )
            
            benchmark = CompilerBenchmark(model, config)
            
            # Measure compilation time
            compilation_time = benchmark.measure_compilation_time(
                model_path,
                f"{output_base}/quant_{quant_type}"
            )
            
            # Get model size
            model_size = benchmark.results.get('size', 'N/A')
            
            results[quant_type] = {
                'description': description,
                'compilation_time': compilation_time,
            }
            
            logger.info("")
        
        return results
    
    @staticmethod
    def compare_pruning_strategies(model, model_path: str,
                                  output_base: str) -> Dict:
        """Compare different pruning strategies"""
        logger.info("\n" + "="*60)
        logger.info("Comparing Pruning Strategies")
        logger.info("="*60 + "\n")
        
        pruning_amounts = [0.0, 10.0, 20.0, 30.0]
        results = {}
        
        for pruning in pruning_amounts:
            logger.info(f"Testing: {pruning}% Pruning")
            logger.info("-" * 40)
            
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                quantization="int8",
                pruning=pruning
            )
            
            benchmark = CompilerBenchmark(model, config)
            
            # Measure compilation time
            compilation_time = benchmark.measure_compilation_time(
                model_path,
                f"{output_base}/prune_{pruning}"
            )
            
            results[f"{pruning}%"] = {
                'compilation_time': compilation_time,
            }
            
            logger.info("")
        
        return results
    
    @staticmethod
    def compare_arm_targets(model, model_path: str,
                           output_base: str) -> Dict:
        """Compare different ARM targets"""
        logger.info("\n" + "="*60)
        logger.info("Comparing ARM Targets")
        logger.info("="*60 + "\n")
        
        targets = [
            ARMTarget.ARMV7,
            ARMTarget.ARMV8,
            ARMTarget.ARMV8_NEON,
            ARMTarget.CORTEX_M4,
            ARMTarget.CORTEX_M7,
        ]
        
        results = {}
        
        for target in targets:
            logger.info(f"Testing: {target.value}")
            logger.info("-" * 40)
            
            config = ModelConfig(
                framework=FrameworkType.PYTORCH,
                input_shape=(1, 3, 224, 224),
                target=target,
                quantization="int8"
            )
            
            benchmark = CompilerBenchmark(model, config)
            
            # Generate code
            compiler = MLEmbeddedCompiler(config)
            compiler.compile(model_path, f"{output_base}/target_{target.value}")
            
            results[target.value] = {
                'status': 'success'
            }
            
            logger.info(f"✓ {target.value} compilation successful")
            logger.info("")
        
        return results


def run_benchmark_suite(model, model_path: str, output_base: str):
    """Run complete benchmark suite"""
    logger.info("\n" + "="*60)
    logger.info("ML Embedded Compiler Benchmark Suite")
    logger.info("="*60 + "\n")
    
    # Create base config
    config = ModelConfig(
        framework=FrameworkType.PYTORCH,
        input_shape=(1, 3, 224, 224)
    )
    
    # Run benchmarks
    benchmark = CompilerBenchmark(model, config)
    
    logger.info("\n1. Model Complexity Analysis")
    logger.info("-" * 60)
    complexity = benchmark.get_model_complexity()
    
    logger.info("\n2. Inference Performance")
    logger.info("-" * 60)
    benchmark.measure_model_inference()
    
    logger.info("\n3. Optimization Comparison")
    logger.info("-" * 60)
    
    # Compare strategies
    quant_results = OptimizationComparison.compare_quantization_strategies(
        model, model_path, output_base
    )
    
    prune_results = OptimizationComparison.compare_pruning_strategies(
        model, model_path, output_base
    )
    
    arm_results = OptimizationComparison.compare_arm_targets(
        model, model_path, output_base
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Benchmark Summary")
    logger.info("="*60)
    logger.info(f"\nTotal Benchmarks Run: {1 + len(quant_results) + len(prune_results) + len(arm_results)}")
    logger.info(f"Quantization Strategies: {len(quant_results)}")
    logger.info(f"Pruning Strategies: {len(prune_results)}")
    logger.info(f"ARM Targets: {len(arm_results)}")
    
    return {
        'complexity': complexity,
        'quantization': quant_results,
        'pruning': prune_results,
        'arm_targets': arm_results,
    }


if __name__ == "__main__":
    logger.info("Benchmark utilities loaded successfully")
