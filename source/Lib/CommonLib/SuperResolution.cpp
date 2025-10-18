#include "SuperResolution.h"
#include "CommonDef.h"
#include <iostream>

bool SuperResolution::loadModel(const std::string& modelPath)
{
  m_model = torch::jit::load(modelPath);
  m_model.eval();
  m_modelLoaded = true;
  std::cout << "Model loaded successfully from: " << modelPath << std::endl;
  return true;
}

torch::Tensor SuperResolution::performInference(const torch::Tensor& input)
{
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  torch::Tensor output = m_model.forward(inputs).toTensor();
  return output;
}

torch::Tensor SuperResolution::pelArrayToTensor(const Pel* pelArray, int width, int height, int bitDepth)
{
  std::vector<float> floatData;
  floatData.reserve(width * height);
  
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      float value = static_cast<float>(pelArray[y * width + x]);
      float normalizedValue = value / ((1 << bitDepth) - 1.0f);
      floatData.push_back(normalizedValue);
    }
  }
  
  auto tensor = torch::from_blob(floatData.data(), {height, width, 1}, torch::kFloat).clone();
  tensor = tensor.permute({2, 0, 1});
  
  return tensor;
}

void SuperResolution::tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth)
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
