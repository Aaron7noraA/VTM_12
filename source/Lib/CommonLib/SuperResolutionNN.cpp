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
    return false;
  }

  // Convert input to tensor
  torch::Tensor inputTensor = pelArrayToTensor(inputData, inputWidth, inputHeight, bitDepth);
  
  // Perform inference
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(inputTensor);
  torch::Tensor outputTensor = m_model.forward(inputs).toTensor();
  
  // Convert output back to Pel array
  tensorToPelArray(outputTensor, outputData, outputWidth, outputHeight, bitDepth);
  
  return true;
}

double SuperResolutionNN::calculateMSE(const Pel* block1, const Pel* block2, int width, int height)
{
  double mse = 0.0;
  int pixelCount = 0;
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      double diff = static_cast<double>(block1[y * width + x]) - static_cast<double>(block2[y * width + x]);
      mse += diff * diff;
      pixelCount++;
    }
  }
  
  return (pixelCount > 0) ? (mse / pixelCount) : 0.0;
}

bool SuperResolutionNN::exhaustiveSearch(const Pel* refBlock, int refWidth, int refHeight,
                                       Pel* targetBlock, int targetWidth, int targetHeight,
                                       int bitDepth, Pel* vtmResult, Pel* nnResult)
{
  if (!m_modelLoaded)
  {
    return false;
  }

  // Exhaustive search: compare VTM vs NN upsampling results
  // In RPR context, we compare the quality of the two upsampling methods
  // We use the targetBlock as reference (this should be the target frame we're trying to match)
  
  // Calculate MSE between each result and the target frame
  double vtmMSE = calculateMSE(targetBlock, vtmResult, targetWidth, targetHeight);
  double nnMSE = calculateMSE(targetBlock, nnResult, targetWidth, targetHeight);
  
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
  tensor = tensor.permute({2, 0, 1}); // CHW format
  
  delete[] floatData;
  return tensor;
}

void SuperResolutionNN::tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth)
{
  torch::Tensor cpuTensor = tensor.cpu();
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
