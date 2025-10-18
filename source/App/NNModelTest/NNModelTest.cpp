#include "NNModelLoader.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "usage: NNModelTest <path-to-exported-script-module>\n";
    return -1;
  }

  NNModelLoader loader;

  loader.loadModel(argv[1]);
  std::cout << "Model loaded successfully." << std::endl;

  int width = 64;
  int height = 64;
  int bitDepth = 8;
  
  std::vector<Pel> inputData(width * height);
  for (int i = 0; i < width * height; ++i)
  {
    inputData[i] = static_cast<Pel>(i % 256);
  }

  torch::Tensor inputTensor = loader.pelArrayToTensor(inputData.data(), width, height, bitDepth);
  std::cout << "Input tensor shape: [";
  for (int i = 0; i < inputTensor.dim(); ++i)
  {
    std::cout << inputTensor.size(i);
    if (i < inputTensor.dim() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  torch::Tensor outputTensor = loader.performInference(inputTensor);
  
  std::cout << "Output tensor shape: [";
  for (int i = 0; i < outputTensor.dim(); ++i)
  {
    std::cout << outputTensor.size(i);
    if (i < outputTensor.dim() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  std::cout << "First few output values: ";
  std::cout << outputTensor.slice(0, 0, 5) << std::endl;

  int outputWidth = outputTensor.size(2);
  int outputHeight = outputTensor.size(1);
  std::vector<Pel> outputData(outputWidth * outputHeight);
  
  loader.tensorToPelArray(outputTensor, outputData.data(), outputWidth, outputHeight, bitDepth);
  
  std::cout << "Output converted to Pel array: " << outputWidth << "x" << outputHeight << std::endl;
  std::cout << "Test completed successfully!" << std::endl;

  return 0;
}
