#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace ams
{
namespace ml
{

/// Base class representing a model available to the AMS system.
///
/// An AbstractModel stores the filesystem path to the model
/// and a version number. It does not define loading or execution
/// semantics; those are handled by derived 'concrete' classes or by the
///
/// The model path may come either from a JSON description (containing
/// `"model_path"`) or from an explicit string constructor.
class AbstractModel
{
public:
  using Json = nlohmann::json;
  using Path = std::filesystem::path;

  /// Construct a model from an explicit path and version number.
  ///
  /// The `modelPath` string is converted into a filesystem::path.
  /// If the provided string is empty, the internal path is left empty.
  explicit AbstractModel(std::string modelPath,
                         std::optional<std::string> Name = std::nullopt,
                         int Version = 0);

  /// Construct a model from a JSON object.
  ///
  /// Expects a `"model_path"` field if the model specifies one and optionally a `"model_name"` providing an identifier to the model
  explicit AbstractModel(const Json& value);


  /// Returns the filesystem path associated with the model.
  const Path& getPath() const { return ModelPath; }

  /// Returns the filesystem path associated with the model.
  const std::optional<std::string>& getName() const { return Name; }

  /// Returns the model version identifier.
  int getVersion() const { return Version; }

  /// Print model information to stdout.
  ///
  /// Primarily useful for debugging or logging.
  void info() const;

private:
  /// Filesystem path to the model.
  Path ModelPath;

  /// Optional name of the model.
  std::optional<std::string> Name;

  /// Model version identifier.
  int Version = 0;

  /// Module unique id
  int UUID = 0;
};

}  // namespace ml
}  // namespace ams
