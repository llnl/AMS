#include "AbstractModel.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "wf/debug.h"

namespace ams
{
namespace ml
{

namespace fs = std::filesystem;
using namespace std;
using namespace nlohmann;

/// Extracts the path from a JSON object.
///
/// If `"model_path"` is present, the path is validated against
/// the filesystem. Returns an empty path if the key is missing.
static inline AbstractModel::Path parsePath(const json& Root)
{
  AbstractModel::Path Path;

  if (Root.contains("model_path")) {
    Path = Root["model_path"].get<string>();

    AMS_CFATAL(AMS,
               (!Path.empty() && !fs::exists(Path)),
               "Path '{}' to model does not exist\n",
               Path.string());
  }

  return Path;
}

static optional<string> parseName(const json& Root)
{
  if (!Root.contains("model_name")) return std::nullopt;

  auto Name = Root["model_name"].get<std::string>();
  return Name;
}

AbstractModel::AbstractModel(const json& Value)
    : ModelPath{parsePath(Value)}, Name{parseName(Value)}, Version{0}
{
}

AbstractModel::AbstractModel(std::string ModelPath,
                             optional<string> Name,
                             int Version)
    : ModelPath{AbstractModel::Path{std::move(ModelPath)}},
      Name(Name),
      Version{Version}
{
  AMS_CWARNING(AbstractModel,
               this->ModelPath.empty(),
               "AbstractModel constructed with empty model path");
}

void AbstractModel::info() const
{
  if (Name) std::cout << "Model Name: " << *Name << " ";
  std::cout << "Model Path: " << ModelPath.string()
            << " with version: " << Version << "\n";
}

}  // namespace ml
}  // namespace ams
