#include "AbstractModel.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace ams
{
namespace ml
{

namespace fs = std::filesystem;
using namespace std;

/// Extracts the path from a JSON object.
///
/// If `"model_path"` is present, the path is validated against
/// the filesystem. Returns an empty path if the key is missing.
static inline AbstractModel::Path parsePath(const Json& root)
{
  AbstractModel::Path path;

  if (root.contains("model_path")) {
    path = root["model_path"].get<string>();

    AMS_CFATAL(AMS,
               (!path.empty() && !fs::exists(path)),
               "Path '%s' to model does not exist\n",
               path.string().c_str());
  }

  return path;
}

static inline optional<string> AbstractModel::Path parseName(const Json& Root)
{
  if (!Root.contains("model_name")) return std::nullopt;

  auto Name = Root["model_name"].get<std::string>();
  return Name;
}

AbstractModel::AbstractModel(const Json& Value)
    : modelPath{parsePath(Value)}, Name(parseName(Value)), Version{0}
{
}

AbstractModel::AbstractModel(std::string modelPath,
                             std::optional<string> Name,
                             int Version)
    : modelPath{Path{std::move(ModelPath)}}, Name(Name), Version{Version}
{
  AMS_CWARNING(AbstractModel,
               this->modelPath.empty(),
               "AbstractModel constructed with empty model path");
}

void AbstractModel::info() const
{
  if (Name) std::cout << "Model Name: " << ModelPath.Name() << " ";
  std::cout << "Model Path: " << ModelPath.string();
  << " with version: " << Version << "\n";
}

}  // namespace ml
}  // namespace ams
