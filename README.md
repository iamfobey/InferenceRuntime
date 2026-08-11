# InferenceRuntime

[Русская версия](README_RU.md)

An experimental C++20 runtime for running language models on the CPU. It currently loads the **SmolLM2** architecture from a Hugging Face-compatible model directory and performs `float32`/`float16` computation with OpenMP and AVX2 acceleration.

> This project is under development: the current executable performs prefill and decoding and reports average token latency and throughput, but does not yet return generated text.

## Requirements

- CMake 3.25.1 or later
- A C++20-capable compiler
- OpenMP (enabled by default and required by the standard configuration)
- A CPU with AVX2, FMA, and F16C for the accelerated build

## Build

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If OpenMP or AVX2 is unavailable, disable it during configuration:

```powershell
cmake -S . -B build -DENABLE_OPENMP=OFF -DENABLE_AVX2=OFF
```

## Run

Pass the architecture, model directory, and prompt:

```powershell
.\build\Release\InferenceRuntime.exe smollm2 C:\models\SmolLM2-135M "Hello, world!"
```

The model directory must contain:

```text
config.json
model.safetensors
tokenizer.json
```

Only the `smollm2` architecture identifier is supported. The loader expects weights in a single `model.safetensors` file with `float16` tensors.

## Project layout

- `src/Backend/CPU` — CPU backend and tensor operations
- `src/Math` — computational kernels, including AVX2 variants
- `src/Model/SmolLM2` — SmolLM2 weights loader, tokenizer, and implementation
- `src/Runtime` — high-level inference API
- `thirdparty/simdjson` — bundled JSON parser

## Limitations

- CPU only; the demo application's thread count is currently set in `src/Main.cpp`.
- Sampling is greedy (`argmax`).
- Tests in `tests/` are not yet wired into CMake.
