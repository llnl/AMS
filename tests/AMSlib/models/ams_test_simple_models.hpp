#pragma once
#include <catch2/catch_tostring.hpp>  // must come before specialization
#include <ostream>
#include <string>
#include <vector>

struct test_models {
  std::string ModelPath;
  std::string ModelPrecision;
  std::string ModelDevice;
  std::string UQType;
  test_models(std::string ModelPath,
              std::string ModelPrecision,
              std::string ModelDevice,
              std::string UQType)
      : ModelPath(ModelPath),
        ModelPrecision(ModelPrecision),
        ModelDevice(ModelDevice),
        UQType(UQType)
  {
  }
};


template <>
struct Catch::StringMaker<test_models> {
  static std::string convert(test_models const& m)
  {
    return "model=" + m.ModelPath + " precision=" + m.ModelPrecision +
           " device=" + m.ModelDevice + " uq=" + m.UQType;
  }
};

inline std::ostream& operator<<(std::ostream& os, test_models const& m)
{
  return os << "path=" << m.ModelPath << " | precision=" << m.ModelPrecision
            << " | device=" << m.ModelDevice << " | uq=" << m.UQType;
}
