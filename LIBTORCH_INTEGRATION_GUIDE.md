# LibTorch Integration Guide for VTM Neural Network Super Resolution

## Overview
This guide explains how to properly link LibTorch with VTM for Neural Network Super Resolution functionality.

## Prerequisites

### 1. LibTorch Installation
- Download LibTorch 2.1.0 CPU Release from: https://pytorch.org/get-started/locally/
- Extract to: `C:/Users/User/Desktop/LibTorch/libtorch/`
- **Important**: Use LibTorch 2.1.0 (not 2.9.0) for Windows compatibility

### 2. System Requirements
- Windows 10/11
- Visual Studio 2022 Community
- CMake 3.18+ (4.0.1 recommended)
- Visual C++ Redistributable

### 3. CMake Configuration
The integration uses the following CMake variables:
- `VTM_NN_SR_ENABLE`: Enable/disable NN Super Resolution (ON/OFF)
- `Torch_DIR`: Path to LibTorch CMake configuration

## Build Commands

### With Neural Network Super Resolution
```bash
cmake -B build-sr -S . -DVTM_NN_SR_ENABLE=ON -DTorch_DIR="C:/Users/User/Desktop/LibTorch/libtorch/share/cmake/Torch" -G "Visual Studio 17 2022"
cmake --build build-sr --config Release

# Copy LibTorch DLLs to executable directory
copy "C:\Users\User\Desktop\LibTorch\libtorch\lib\*.dll" "build-sr\source\App\EncoderApp\Release\"
```

### Without Neural Network Super Resolution (Standard VTM)
```bash
cmake -B build-standard -S . -DVTM_NN_SR_ENABLE=OFF -G "Visual Studio 17 2022"
cmake --build build-standard --config Release
```

## Configuration Details

### Main CMakeLists.txt Changes
- Added `option(VTM_NN_SR_ENABLE "Enable Neural Network Super Resolution" OFF)`
- Added LibTorch `find_package(Torch REQUIRED)`
- Set C++ standard to 17 (required by LibTorch 2.1.0)
- Added compile definition `-DVTM_NN_SR_ENABLE`

### CommonLib CMakeLists.txt Changes
- Added LibTorch linking: `target_link_libraries(${LIB_NAME} ${TORCH_LIBRARIES})`
- Added compile definition: `target_compile_definitions(${LIB_NAME} PUBLIC VTM_NN_SR_ENABLE=1)`

## Build Scripts

### build_with_sr.bat
```batch
@echo off
echo Building VTM with Super Resolution support...
cmake -B build-sr -S . -DVTM_NN_SR_ENABLE=ON -DTorch_DIR="C:/Users/User/Desktop/LibTorch/libtorch/share/cmake/Torch" -G "Visual Studio 17 2022"
cmake --build build-sr --target EncoderApp --config Release
```

### build_without_sr.bat
```batch
@echo off
echo Building VTM without Super Resolution...
cmake -B build-standard -S . -DVTM_NN_SR_ENABLE=OFF -G "Visual Studio 17 2022"
cmake --build build-standard --target EncoderApp --config Release
```

## Usage

### With Super Resolution
```bash
EncoderApp.exe -c cfg/encoder_randomaccess_vtm.cfg -c cfg/rpr/scale1.5x.cfg -c cfg/super_resolution.cfg -i input.yuv -o output.bin
```

### Without Super Resolution (Standard VTM)
```bash
EncoderApp.exe -c cfg/encoder_randomaccess_vtm.cfg -c cfg/rpr/scale1.5x.cfg -i input.yuv -o output.bin
```

## Troubleshooting

### Common Issues

1. **LibTorch Not Found**
   - Ensure `Torch_DIR` points to the correct LibTorch installation
   - Check that `libtorch/share/cmake/Torch` exists
   - Use LibTorch 2.1.0 (not 2.9.0) for Windows compatibility

2. **C++ Standard Conflicts**
   - LibTorch 2.1.0 requires C++17 (not C++14)
   - VTM uses C++11 by default
   - The configuration sets C++17 only when NN SR is enabled

3. **Linking Errors**
   - Ensure all LibTorch libraries are available
   - Use Visual Studio build system instead of command-line CMake
   - Check that the LibTorch version is compatible

4. **Runtime Errors - Missing DLLs**
   - Copy all LibTorch DLLs to executable directory:
     ```bash
     copy "C:\Users\User\Desktop\LibTorch\libtorch\lib\*.dll" "build-sr\source\App\EncoderApp\Release\"
     ```
   - Required DLLs: asmjit.dll, c10.dll, fbgemm.dll, torch_cpu.dll, torch.dll, etc.

5. **Model Loading Errors**
   - Ensure the model file exists at `models/sr_model.pt`
   - Check that the model is compatible with LibTorch 2.1.0
   - Verify the model was trained with the correct PyTorch version

6. **Build Failures**
   - Use fresh build directory (delete existing build folders)
   - Use Visual Studio instead of command-line CMake for complex projects
   - Ensure CMake version is 3.18+ (4.0.1 recommended)

## File Structure
```
VTM_Project/
├── source/Lib/CommonLib/
│   ├── SuperResolutionNN.h
│   ├── SuperResolutionNN.cpp
│   └── Slice.cpp (modified)
├── cfg/
│   └── super_resolution.cfg
├── build_with_sr.bat
├── build_without_sr.bat
└── CMakeLists.txt (modified)
```

## Notes
- The integration is designed to be optional - VTM can build and run without LibTorch
- Only luma (Y) component is processed by the neural network
- Chroma (U, V) components always use VTM's default upsampling
- Downsampling always uses VTM's default methods
