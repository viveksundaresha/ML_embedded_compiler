#!/usr/bin/env python3
"""
ML Embedded Compiler - Command Line Interface
"""

import argparse
import sys
from pathlib import Path
from compiler import (
    MLEmbeddedCompiler, ModelConfig, FrameworkType,
    ARMTarget, OptimizationStrategy
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="ML Model Compiler for Embedded Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PyTorch model with INT8 quantization for ARM v8 with NEON
  %(prog)s --framework pytorch --model model.pth --output ./build \\
    --input-shape 1,3,224,224 --quantization int8 --target armv8_neon

  # TensorFlow model with aggressive optimization for Cortex-M7
  %(prog)s --framework tensorflow --model model.h5 --output ./build \\
    --input-shape 1,28,28,1 --quantization int8 --pruning 30 \\
    --target cortex_m7 --strategy aggressive

  # Conservative optimization (best compatibility)
  %(prog)s --framework pytorch --model model.pth --output ./build \\
    --quantization none --pruning 0 --strategy conservative
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--framework',
        choices=['pytorch', 'tensorflow'],
        required=True,
        help='Deep learning framework'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the pre-trained model'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for compiled code'
    )
    
    # Input shape
    parser.add_argument(
        '--input-shape',
        type=str,
        required=True,
        help='Input shape as comma-separated values (e.g., 1,3,224,224)'
    )
    
    # Optimization options
    parser.add_argument(
        '--quantization',
        choices=['none', 'int8', 'float16'],
        default='int8',
        help='Quantization strategy (default: int8)'
    )
    
    parser.add_argument(
        '--pruning',
        type=float,
        default=0.0,
        help='Pruning percentage (0-50, default: 0)'
    )
    
    # Target platform
    parser.add_argument(
        '--target',
        choices=['armv7', 'armv8', 'armv8_neon', 'cortex_m4', 'cortex_m7'],
        default='armv8_neon',
        help='ARM target architecture (default: armv8_neon)'
    )
    
    # Optimization strategy
    parser.add_argument(
        '--strategy',
        choices=['aggressive', 'balanced', 'conservative'],
        default='balanced',
        help='Optimization strategy (default: balanced)'
    )
    
    # Verbose
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments"""
    # Check model file exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Validate input shape
    try:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
        if len(input_shape) < 2:
            raise ValueError("Input shape must have at least 2 dimensions")
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid input shape: {e}")
        sys.exit(1)
    
    # Validate pruning percentage
    if not 0 <= args.pruning <= 50:
        logger.error("Pruning percentage must be between 0 and 50")
        sys.exit(1)
    
    return input_shape


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("ML Embedded Model Compiler")
    logger.info("=" * 60)
    
    # Validate arguments
    input_shape = validate_arguments(args)
    
    # Create configuration
    config = ModelConfig(
        framework=FrameworkType[args.framework.upper()],
        input_shape=input_shape,
        quantization=args.quantization,
        pruning=args.pruning,
        target=ARMTarget[args.target.upper().replace('-', '_')],
        optimization_strategy=OptimizationStrategy[args.strategy.upper()]
    )
    
    logger.info("Configuration:")
    logger.info(f"  Framework: {config.framework.value}")
    logger.info(f"  Input Shape: {config.input_shape}")
    logger.info(f"  Quantization: {config.quantization}")
    logger.info(f"  Pruning: {config.pruning}%")
    logger.info(f"  Target: {config.target.value}")
    logger.info(f"  Strategy: {config.optimization_strategy.value}")
    logger.info("")
    
    # Run compilation
    try:
        compiler = MLEmbeddedCompiler(config)
        result = compiler.compile(args.model, args.output)
        
        logger.info("=" * 60)
        logger.info("Compilation Results:")
        logger.info(f"  Original Size: {result['original_size_mb']:.2f} MB")
        logger.info(f"  Optimized Size: {result['optimized_size_mb']:.2f} MB")
        logger.info(f"  Compression Ratio: {result['compression_ratio']:.2f}x")
        logger.info("=" * 60)
        logger.info(f"✓ Compilation successful!")
        logger.info(f"✓ Output saved to: {Path(args.output).absolute()}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. cd {args.output}")
        logger.info(f"  2. mkdir build && cd build")
        logger.info(f"  3. cmake ..")
        logger.info(f"  4. make")
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
