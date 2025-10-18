#ifndef __SUPERRESOLUTION__
#define __SUPERRESOLUTION__

#include "CommonDef.h"
#include <torch/script.h>
#include <string>

class SuperResolution
{
private:
  torch::jit::script::Module m_model;
  bool m_modelLoaded;

public:
  SuperResolution() : m_modelLoaded(false) {}
  ~SuperResolution() {}

  bool loadModel(const std::string& modelPath);
  bool isModelLoaded() const { return m_modelLoaded; }
  torch::Tensor performInference(const torch::Tensor& input);
  torch::Tensor pelArrayToTensor(const Pel* pelArray, int width, int height, int bitDepth);
  void tensorToPelArray(const torch::Tensor& tensor, Pel* pelArray, int width, int height, int bitDepth);
};

#endif
