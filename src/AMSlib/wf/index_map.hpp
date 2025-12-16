#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ams
{

/// Field-to-column mapping for layout transformations.
struct IndexMap {
  struct FieldInfo {
    std::string Name;

    enum class Kind { Input, InOut, Output };
    Kind EKind;

    int64_t Offset;  ///< Starting column in the concatenated tensor
    int64_t Cols;    ///< Number of columns this field covers
  };

  std::vector<FieldInfo> Fields;
};

}  // namespace ams
