#include "SuperResolutionNN.h"

SuperResolutionNN::SuperResolutionNN()
  : m_modelLoaded(false)
{
}

SuperResolutionNN::~SuperResolutionNN()
{
}

bool SuperResolutionNN::loadModel(const char* modelPath)
{
  m_model = torch::jit::load(modelPath);
  m_model.eval();
  m_modelLoaded = true;
  return true;
}

bool SuperResolutionNN::performInference(const Pel* inputData, int inputWidth, int inputHeight,
                                       Pel* outputData, int outputWidth, int outputHeight,
                                       int bitDepth)
{
  if (!m_modelLoaded)
  {
    printf("ERROR: Model not loaded!\n");
    return false;
  }
  
  if (!inputData || !outputData)
  {
    printf("ERROR: Null pointer in performInference!\n");
    return false;
  }
  
  if (inputWidth <= 0 || inputHeight <= 0 || outputWidth <= 0 || outputHeight <= 0)
  {
    printf("ERROR: Invalid dimensions in performInference!\n");
    printf("  inputWidth=%d, inputHeight=%d, outputWidth=%d, outputHeight=%d\n", 
           inputWidth, inputHeight, outputWidth, outputHeight);
    return false;
  }

  // Convert input to tensor
  torch::Tensor inputTensor = pelArrayToTensor(inputData, inputWidth, inputHeight, bitDepth);
  
  // Print input tensor shape
  printf("Input tensor shape: [");
  for (int i = 0; i < inputTensor.dim(); i++) {
    printf("%ld", inputTensor.size(i));
    if (i < inputTensor.dim() - 1) printf(", ");
  }
  printf("]\n");
  
  // Perform inference
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(inputTensor);
  
  torch::Tensor outputTensor;
  try {
    outputTensor = m_model.forward(inputs).toTensor();
  } catch (const std::exception& e) {
    printf("ERROR: Model forward pass failed: %s\n", e.what());
    return false;
  }
  
  // Print output tensor shape
  printf("Output tensor shape: [");
  for (int i = 0; i < outputTensor.dim(); i++) {
    printf("%ld", outputTensor.size(i));
    if (i < outputTensor.dim() - 1) printf(", ");
  }
  printf("]\n");
  
  // Convert output back to Pel array
  tensorToPelArray(outputTensor, outputData, outputWidth, outputHeight, bitDepth);
  
  return true;
}

double SuperResolutionNN::calculateMSE(const Pel* block1, const Pel* block2, int width, int height)
{
  // Safety checks
  if (!block1 || !block2)
  {
    printf("ERROR: Null pointer in calculateMSE! block1=%p, block2=%p\n", block1, block2);
    return 999999.0; // Return high MSE to indicate error
  }
  
  if (width <= 0 || height <= 0)
  {
    printf("ERROR: Invalid dimensions in calculateMSE: %dx%d\n", width, height);
    return 999999.0;
  }
  
  double mse = 0.0;
  int pixelCount = 0;
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = y * width + x;
      double diff = static_cast<double>(block1[idx]) - static_cast<double>(block2[idx]);
      mse += diff * diff;
      pixelCount++;
    }
  }
  
  return (pixelCount > 0) ? (mse / pixelCount) : 0.0;
}

bool SuperResolutionNN::exhaustiveSearch(const Pel* refBlock, int refWidth, int refHeight,
                                       const Pel* targetFrame, int targetWidth, int targetHeight,
                                       int bitDepth, Pel* vtmResult, Pel* nnResult)
{
  if (!m_modelLoaded)
  {
    printf("ERROR: Model not loaded in exhaustiveSearch!\n");
    return false;
  }
  
  // Critical null pointer checks
  if (!targetFrame || !vtmResult || !nnResult)
  {
    printf("ERROR: Null pointer in exhaustiveSearch! targetFrame=%p, vtmResult=%p, nnResult=%p\n", 
           targetFrame, vtmResult, nnResult);
    return false;
  }
  
  if (targetWidth <= 0 || targetHeight <= 0)
  {
    printf("ERROR: Invalid target dimensions: %dx%d\n", targetWidth, targetHeight);
    return false;
  }

  // Exhaustive search: compare VTM vs NN upsampling results
  // In RPR context, we compare the quality of the two upsampling methods
  // We use the targetFrame as reference (this should be the target frame we're trying to match)
  
  // Debug: Print dimensions before MSE calculation
  printf("exhaustiveSearch - Dimensions: target=%dx%d, vtmResult=%p, nnResult=%p\n", 
         targetWidth, targetHeight, vtmResult, nnResult);
  
  // Calculate MSE between each result and the target frame
  printf("Calculating VTM MSE...\n");
  double vtmMSE = calculateMSE(targetFrame, vtmResult, targetWidth, targetHeight);
  printf("Calculating NN MSE...\n");
  double nnMSE = calculateMSE(targetFrame, nnResult, targetWidth, targetHeight);
  
  // Print both MSE values for comparison
  printf("Exhaustive Search MSE Comparison:\n");
  printf("  VTM MSE: %.6f\n", vtmMSE);
  printf("  NN MSE:  %.6f\n", nnMSE);
  printf("  Decision: %s (NN %s VTM)\n", 
         (nnMSE < vtmMSE) ? "Use NN" : "Use VTM",
         (nnMSE < vtmMSE) ? "better than" : "worse than");
  
  // Return true if NN result has lower MSE (better quality) than VTM result
  return nnMSE < vtmMSE;
}

torch::Tensor SuperResolutionNN::pelArrayToTensor(const Pel* pelArray, int width, int height, int bitDepth)
{
  float* floatData = new float[width * height];
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      float value = static_cast<float>(pelArray[y * width + x]);
      float normalizedValue = value / ((1 << bitDepth) - 1.0f);
      floatData[y * width + x] = normalizedValue;
    }
  }
  
  auto tensor = torch::from_blob(floatData, {height, width, 1}, torch::kFloat).clone();
  tensor = tensor.permute({2, 0, 1}); // CHW format -> [1, H, W]
  tensor = tensor.unsqueeze(0); // Add batch dimension -> [1, 1, H, W]
  
  // Safe to delete after clone() - tensor now owns its own data
  delete[] floatData;
  return tensor;
}

void SuperResolutionNN::tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth)
{
  torch::Tensor cpuTensor = tensor.cpu();
  
  // Debug: Print tensor shape before processing
  printf("tensorToPelArray - Input tensor shape: [");
  for (int i = 0; i < cpuTensor.dim(); i++) {
    printf("%ld", cpuTensor.size(i));
    if (i < cpuTensor.dim() - 1) printf(", ");
  }
  printf("]\n");
  
  // Remove batch dimension if present: [1, 1, H, W] -> [1, H, W]
  if (cpuTensor.dim() == 4) {
    cpuTensor = cpuTensor.squeeze(0); // Remove first batch dimension
  }
  
  // Convert from CHW to HWC for easier indexing
  if (cpuTensor.dim() == 3) {
    cpuTensor = cpuTensor.permute({1, 2, 0}); // CHW -> HWC
  }
  
  // Verify tensor dimensions match expected output
  if (cpuTensor.dim() == 3) {
    int tensorHeight = cpuTensor.size(0);
    int tensorWidth = cpuTensor.size(1);
    if (tensorHeight != height || tensorWidth != width) {
      printf("ERROR: Tensor dimensions mismatch! Expected [%d, %d], got [%d, %d]\n", 
             height, width, tensorHeight, tensorWidth);
      return;
    }
  }
  
  float* data = cpuTensor.data_ptr<float>();
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      float normalizedValue = data[y * width + x];
      float denormalizedValue = normalizedValue * ((1 << bitDepth) - 1.0f);
      denormalizedValue = std::max(0.0f, std::min(denormalizedValue, (1 << bitDepth) - 1.0f));
      pelArray[y * width + x] = static_cast<Pel>(std::round(denormalizedValue));
    }
  }
}
