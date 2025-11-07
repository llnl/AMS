#pragma once
#include <catch2/catch_tostring.hpp>  // must come before specialization
#include <ostream>
#include <string>
#include <vector>

struct linear_model {
  std::string ModelPath;
  std::string ModelPrecision;
  std::string ModelDevice;
  std::string UQType;
  int numIn, numOut;
  linear_model(std::string ModelPath,
               std::string ModelPrecision,
               std::string ModelDevice,
               std::string UQType,
               int numIn,
               int numOut)
      : ModelPath(ModelPath),
        ModelPrecision(ModelPrecision),
        UQType(UQType),
        numIn(numIn),
        numOut(numOut)
  {
  }
};


template <>
struct Catch::StringMaker<linear_model> {
  static std::string convert(linear_model const& m)
  {
    return "model=" + m.ModelPath + " precision=" + m.ModelPrecision +
           " device=" + m.ModelDevice + " uq=" + m.UQType;
  }
};

inline std::ostream& operator<<(std::ostream& os, linear_model const& m)
{
  return os << "path=" << m.ModelPath << " | precision=" << m.ModelPrecision
            << " | device=" << m.ModelDevice << " | uq=" << m.UQType;
}
