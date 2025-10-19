#ifndef __SUPERRESOLUTIONNN__
#define __SUPERRESOLUTIONNN__

#include "CommonDef.h"
#include <torch/script.h>

/**
 * @brief Neural Network-based Super Resolution class for VTM
 * 
 * This class provides neural network-based super resolution functionality
 * using LibTorch for inference. It can be used as an alternative to VTM's
 * default upsampling methods in RPR configurations.
 */
class SuperResolutionNN
{
private:
  torch::jit::script::Module m_model;
  bool m_modelLoaded;

public:
  SuperResolutionNN();
  ~SuperResolutionNN();

  // Model management
  bool loadModel(const char* modelPath);
  bool isModelLoaded() const { return m_modelLoaded; }
  
  // Neural network inference
  bool performInference(const Pel* inputData, int inputWidth, int inputHeight, 
                       Pel* outputData, int outputWidth, int outputHeight, 
                       int bitDepth);
  
  // Utility functions
  double calculateMSE(const Pel* block1, const Pel* block2, int width, int height);
  
  // Exhaustive search algorithm
  bool exhaustiveSearch(const Pel* refBlock, int refWidth, int refHeight,
                       Pel* targetBlock, int targetWidth, int targetHeight,
                       int bitDepth, Pel* vtmResult, Pel* nnResult);
  
private:
  // LibTorch-specific helper functions
  torch::Tensor pelArrayToTensor(const Pel* pelArray, int width, int height, int bitDepth);
  void tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth);
};

#endif
