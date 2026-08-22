# InferenceRuntime

Experimental C++20 CPU runtime for the SmolLM2 architecture. It loads a
Hugging Face-compatible model directory and executes `float32`/`float16`
inference and optional AVX2/FMA/F16C acceleration.

## Supported build environments

| Environment | Compiler | Where to run commands | Presets |
|---|---|---|---|
| Windows | MSVC | Visual Studio Developer PowerShell or Developer Command Prompt | `*-msvc` |
| WSL | GCC or Clang | WSL terminal or a CLion WSL toolchain | `*-wsl` |

Do not use a Windows preset from WSL or a WSL preset from Windows. Conan runs
inside the same environment as CMake, so Windows and WSL intentionally have
separate build directories and dependency packages.

## Prerequisites

Install these tools in every environment you use:

- CMake 3.25.1 or newer;
- Conan 2, available as `conan` on `PATH`;
- a C++20 compiler: MSVC on Windows, GCC or Clang in WSL;
- Ninja for Windows MSVC builds; `make` for WSL builds.

AVX2 are enabled by default. The accelerated build requires a CPU
with AVX2, FMA, and F16C support.

## Dependency management

Dependencies are declared in [conanfile.py](conanfile.py).

## Configure, build, and test

### Windows with MSVC

Open a Visual Studio Developer PowerShell or Developer Command Prompt, then:

```powershell
cmake --preset release-msvc
cmake --build --preset release-msvc
ctest --preset release-msvc
```

### WSL

Run from the project directory inside the WSL distribution:

```bash
cmake --preset release-wsl
cmake --build --preset release-wsl
ctest --preset release-wsl
```

Replace `release` with `debug` or `relwithdebinfo` when needed.

### Disable optional CPU features

Pass CMake cache variables during configuration:

```powershell
cmake --preset release-msvc -DENABLE_AVX2=OFF
cmake --build --preset release-msvc
```

Use the corresponding `*-wsl` preset in WSL.

## CLion

Open the project, select the matching CMake profile for the active toolchain,
and reload the CMake project:

- use an `*-MSVC` profile with the Windows MSVC toolchain;
- use an `*-WSL` profile with the WSL toolchain.

The reload/configure step runs Conan automatically. Build or run a CMake
target only after it finishes. No generated Conan toolchain path needs to be
entered in CLion settings.

## Run the application

The model directory must contain:

```text
config.json
model.safetensors
tokenizer.json
```

Windows example:

```powershell
.\build\msvc-release\InferenceRuntime.exe smollm2 C:\models\SmolLM2-135M "Hello, world!" 1
```

WSL example:

```bash
./build/wsl-release/InferenceRuntime smollm2 /mnt/c/models/SmolLM2-135M 'Hello, world!' 1
```

Only the `smollm2` architecture identifier is currently supported. Weights
must be in one `model.safetensors` file with `float16` tensors.

## Benchmarks

Benchmarks are enabled by default. The executable takes an architecture, model
directory, and thread count. It runs ten 300-token generation iterations and
reports elapsed time and tokens per second.

Windows:

```powershell
.\build\msvc-release\RuntimeBenchmark.exe smollm2 C:\models\SmolLM2-135M 1
```

WSL:

```bash
./build/wsl-release/RuntimeBenchmark smollm2 /mnt/c/models/SmolLM2-135M 1
```

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `conan` is not found | Install Conan 2 in the same Windows or WSL environment that runs CMake, then ensure it is on `PATH`. |
| CMake cannot find a dependency | Delete the affected `build/<profile>` directory and configure that preset again. Do not manually copy generated Conan files between profiles. |
| MSVC compiler is not found | Start the Windows commands from a Visual Studio Developer shell. |
| CLion uses the wrong packages | Ensure the CMake profile matches its toolchain: MSVC ↔ `*-MSVC`, WSL ↔ `*-WSL`; then reload CMake. |
| `cmake/conan_provider.cmake` is missing | Restore it from the repository. It is required for automatic Conan integration. |

## Current limitations

- CPU execution only;
- greedy (`argmax`) sampling only;
- demo thread count is currently provided on the command line.
