#include "SuperResolutionNN.h"
#include <algorithm>

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

bool SuperResolutionNN::performInference(const Pel* inputData, int inputWidth, int inputHeight, int inputStride,
                                         Pel* outputData, int outputWidth, int outputHeight, int outputStride,
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

  // Debug: Print input data info
  printf("performInference input data:\n");
  printf("  inputData pointer: %p\n", inputData);
  printf("  inputWidth=%d, inputHeight=%d, bitDepth=%d\n", inputWidth, inputHeight, bitDepth);
  printf("  First 10 input pixels: ");
  for (int i = 0; i < std::min(10, inputWidth * inputHeight); i++) {
    printf("%d ", inputData[i]);
  }
  printf("\n");

  // Convert input to tensor using the provided stride
  torch::Tensor inputTensor = pelArrayToTensor(inputData, inputWidth, inputHeight, bitDepth, inputStride);
  
  // Print input tensor shape before padding
  printf("Input tensor shape before padding: [");
  for (int i = 0; i < inputTensor.dim(); i++) {
    printf("%ld", inputTensor.size(i));
    if (i < inputTensor.dim() - 1) printf(", ");
  }
  printf("]\n");

  int padding_h = ((inputHeight >> 3) << 3) + 4 - inputHeight;
  int padding_w = ((inputWidth >> 3) << 3) + 4 - inputWidth;
  printf("Padding values: padding_h=%d, padding_w=%d\n", padding_h, padding_w);

  // Pad the input tensor to the nearest multiple of 8
  if (padding_h > 0 || padding_w > 0) {
    // Pad format: {left, right, top, bottom}
    inputTensor = torch::reflection_pad2d(inputTensor, {padding_w, padding_w, padding_h, padding_h});
    
    // Print input tensor shape after padding
    printf("Input tensor shape after padding: [");
    for (int i = 0; i < inputTensor.dim(); i++) {
      printf("%ld", inputTensor.size(i));
      if (i < inputTensor.dim() - 1) printf(", ");
    }
    printf("]\n");
  }

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

  // Print output tensor shape before unpadding
  printf("Output tensor shape before unpadding: [");
  for (int i = 0; i < outputTensor.dim(); i++) {
    printf("%ld", outputTensor.size(i));
    if (i < outputTensor.dim() - 1) printf(", ");
  }
  printf("]\n");

  // Remove padding from output tensor
  // Calculate scaling factor dynamically
  int scaling = outputWidth / inputWidth;  // or outputHeight / inputHeight
  padding_h *= scaling;
  padding_w *= scaling;
  if (padding_h > 0 || padding_w > 0) {
    outputTensor = outputTensor.slice(2, padding_h, outputTensor.size(2) - padding_h)
                              .slice(3, padding_w, outputTensor.size(3) - padding_w);
    
    // Print output tensor shape after unpadding
    printf("Output tensor shape after unpadding: [");
    for (int i = 0; i < outputTensor.dim(); i++) {
      printf("%ld", outputTensor.size(i));
      if (i < outputTensor.dim() - 1) printf(", ");
    }
    printf("]\n");
  }
  
  // Print output tensor shape
  printf("Output tensor shape: [");
  for (int i = 0; i < outputTensor.dim(); i++) {
    printf("%ld", outputTensor.size(i));
    if (i < outputTensor.dim() - 1) printf(", ");
  }
  printf("]\n");
  
  // Convert output back to Pel array
  // Use the actual output stride from VTM buffer
  tensorToPelArray(outputTensor, outputData, outputWidth, outputHeight, bitDepth, outputStride);
  
  return true;
}

double SuperResolutionNN::calculateMSE(const Pel* a, int aStride, const Pel* b, int bStride, int width, int height)
{
  // Safety checks
  if (!a || !b)
  {
    printf("ERROR: Null pointer in calculateMSE! a=%p, b=%p\n", a, b);
    return 999999.0; // Return high MSE to indicate error
  }
  if (width <= 0 || height <= 0)
  {
    printf("ERROR: Invalid dimensions in calculateMSE: %dx%d\n", width, height);
    return 999999.0;
  }
  
  double mse = 0.0;
  int pixelCount = 0;
  for (int y = 0; y < height; ++y)
  {
    const Pel* rowA = a + y * aStride;
    const Pel* rowB = b + y * bStride;
    for (int x = 0; x < width; ++x)
    {
      const int d = int(rowA[x]) - int(rowB[x]);
      mse += double(d) * double(d);
      ++pixelCount;
    }
  }
  return (pixelCount > 0) ? (mse / pixelCount) : 0.0;
}

bool SuperResolutionNN::exhaustiveSearch(const Pel* refBlock, int refWidth, int refHeight,
                                       const Pel* targetFrame, int targetWidth, int targetHeight, int targetStride,
                                       int bitDepth, const Pel* vtmResult, int vtmStride,
                                       const Pel* nnResult, int nnStride)
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
  // printf("exhaustiveSearch - Dimensions: target=%dx%d, vtmResult=%p, nnResult=%p\n", 
  //        targetWidth, targetHeight, vtmResult, nnResult);
  
  // Calculate MSE between each result and the target frame
  // printf("Calculating VTM MSE...\n");
  double vtmMSE = calculateMSE(targetFrame, targetStride, vtmResult, vtmStride, targetWidth, targetHeight);
  // printf("Calculating NN MSE...\n");
  double nnMSE = calculateMSE(targetFrame, targetStride, nnResult, nnStride, targetWidth, targetHeight);
  
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

torch::Tensor SuperResolutionNN::pelArrayToTensor(const Pel* pelArray, int width, int height, int bitDepth, int stride)
{
  // Debug: Print input parameters
  printf("pelArrayToTensor: width=%d, height=%d, bitDepth=%d, stride=%d\n", width, height, bitDepth, stride);
  printf("pelArray pointer: %p\n", pelArray);
  
  // Debug: Print first few pixel values (using stride)
  printf("First 10 pixels (stride-aware): ");
  for (int i = 0; i < std::min(10, width * height); i++) {
    int y = i / width;
    int x = i % width;
    printf("%d ", pelArray[y * stride + x]);
  }
  printf("\n");
  
  float* floatData = new float[width * height];
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      float value = static_cast<float>(pelArray[y * stride + x]);  // Use stride for input!
      float normalizedValue = value / ((1 << bitDepth) - 1.0f);
      floatData[y * width + x] = normalizedValue;  // Output buffer is contiguous, so use width
    }
  }
  
  // Debug: Print first few normalized values
  printf("First 10 normalized values: ");
  for (int i = 0; i < std::min(10, width * height); i++) {
    printf("%.3f ", floatData[i]);
  }
  printf("\n");
  
  auto tensor = torch::from_blob(floatData, {height, width, 1}, torch::kFloat).clone();
  tensor = tensor.permute({2, 0, 1}); // CHW format -> [1, H, W]
  tensor = tensor.unsqueeze(0); // Add batch dimension -> [1, 1, H, W]
  
  // Debug: Print tensor values
  printf("Tensor first few values: ");
  auto tensor_data = tensor.data_ptr<float>();
  for (int i = 0; i < std::min(10, (int)tensor.numel()); i++) {
    printf("%.3f ", tensor_data[i]);
  }
  printf("\n");
  
  // Safe to delete after clone() - tensor now owns its own data
  delete[] floatData;
  return tensor;
}

void SuperResolutionNN::tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth, int stride)
{
  torch::Tensor cpuTensor = tensor.cpu();
  
  // Debug: Print tensor shape before processing
  // printf("tensorToPelArray - Input tensor shape: [");
  // for (int i = 0; i < cpuTensor.dim(); i++) {
  //   printf("%ld", cpuTensor.size(i));
  //   if (i < cpuTensor.dim() - 1) printf(", ");
  // }
  // printf("]\n");
  
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
      // Read from tensor data (contiguous): data[y * width + x]
      float normalizedValue = data[y * width + x];
      float denormalizedValue = normalizedValue * ((1 << bitDepth) - 1.0f);
      denormalizedValue = std::max(0.0f, std::min(denormalizedValue, (1 << bitDepth) - 1.0f));
      // Write to output buffer (might have stride): pelArray[y * stride + x]
      pelArray[y * stride + x] = static_cast<Pel>(std::round(denormalizedValue));
    }
  }
}
