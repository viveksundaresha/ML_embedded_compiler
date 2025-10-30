# ML Embedded Compiler

Convert PyTorch and TensorFlow models to optimized C++ code for ARM devices.

## Overview

This tool compiles machine learning models to efficient C++ inference engines. It applies quantization, pruning, and code generation to reduce model size and improve performance on ARM embedded systems.

## Installation

```bash
git clone https://github.com/yourusername/ml-embedded-compiler.git
cd ml-embedded-compiler
pip install -r requirements.txt
```

Verify installation:
```bash
python test_compiler.py
```

## Usage

Basic compilation:
```bash
python cli.py --framework pytorch --model model.pth \
  --output ./build --input-shape 1,3,224,224 \
  --quantization int8 --target armv8_neon
```

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--framework` | pytorch, tensorflow | Model framework |
| `--model` | path | Path to model file |
| `--output` | path | Output directory |
| `--input-shape` | shape | Input dimensions (batch,channels,height,width) |
| `--quantization` | none, int8, float16 | Quantization method |
| `--pruning` | 0-100 | Pruning percentage |
| `--target` | armv7, armv8_neon, cortex_m7, cortex_m4, cortex_a72 | Target architecture |
| `--strategy` | balanced, aggressive, conservative | Optimization level |

## Examples

Compile PyTorch model for mobile:
```bash
python cli.py --framework pytorch --model model.pth \
  --output ./build --input-shape 1,3,224,224 \
  --quantization int8 --target armv8_neon
```

Compile for Raspberry Pi:
```bash
python cli.py --framework pytorch --model model.pth \
  --output ./build --input-shape 1,3,224,224 \
  --quantization int8 --target cortex_a72
```

Compile for microcontroller:
```bash
python cli.py --framework pytorch --model model.pth \
  --output ./build --input-shape 1,224,224 \
  --quantization int8 --target cortex_m7
```

## Output

The compiler generates:
- `ml_inference.h` - C++ header
- `ml_inference.cpp` - Implementation
- `CMakeLists.txt` - Build configuration
- `config.json` - Model metadata

Build the generated code:
```bash
cd build
mkdir cmake_build && cd cmake_build
cmake ..
make
```

## Using Generated Code

```cpp
#include "ml_inference.h"
#include <vector>

int main() {
    ml_inference::InferenceEngine engine;
    
    std::vector<float> input(224*224*3, 0.5f);
    std::vector<float> output(1000, 0.0f);
    
    engine.infer(input.data(), output.data());
    
    return 0;
}
```

## Testing

Run tests:
```bash
python test_compiler.py
```

Run examples:
```bash
python examples.py
```

Run benchmarks:
```bash
python benchmark.py
```

## Requirements

- Python 3.8+
- PyTorch 1.9+ or TensorFlow 2.0+
- CMake 3.10+
- C++ compiler (for building generated code)

## Supported Targets

| Target | Architecture | Use Case |
|--------|--------------|----------|
| armv7 | 32-bit ARM | Mobile devices |
| armv8_neon | 64-bit ARM with NEON | Smartphones |
| cortex_a72 | ARMv8 | Raspberry Pi 4 |
| cortex_m7 | Microcontroller | Embedded systems |
| cortex_m4 | Low-power MCU | IoT devices |

## Performance

Typical results with INT8 quantization:
- Model size reduction: 4-7x
- Inference speedup: 2-3x
- Accuracy impact: < 1% for most models

## Known Issues

- Windows build requires cross-compiler for ARM output
- Custom operators not supported, use standard layers
- Dynamic shapes not supported, fixed sizes only

## License

MIT
