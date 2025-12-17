#pragma once

#include <ostream>
#include <string>
#include <tl/expected.hpp>

#include "wf/debug.h"

namespace ams
{

/// \brief Error category for AMS operations.
enum class AMSErrorType {
  Success,           ///< No erro
  Generic,           ///< Unspecified error.
  FileDoesNotExist,  ///< Path to file or directory does not exist
  TorchInternal,     ///< An internal error that happens to the torch library
  InvalidModel,  ///< A torchscripted model that has not been serialized through AMS
  InvalidShapes,  ///< Some Data shape is not the proper|expected shape
};

/// \brief Strongly-typed error object used across AMS.
///
/// AMSError carries a message, a typed category, and optional file/line
/// information. File/line are stored separately from the message to allow
/// hiding implementation details in certain build modes.
class AMSError
{
public:
  /// \brief Construct an error with type, message, and optional file/line.
  ///
  /// \param type   The error category.
  /// \param message Human-readable description of the error.
  /// \param file   Source file where the error originated (optional).
  /// \param line   Source line where the error originated (optional).
  AMSError(AMSErrorType Type,
           std::string Message,
           std::string File = std::string{},
           int Line = 0)
      : Type(Type),
        Message(std::move(Message)),
        File(std::move(File)),
        Line(Line)
  {
  }

  /// \brief Defaulted copy constructor.
  AMSError(const AMSError&) = default;

  /// \brief Defaulted move constructor.
  AMSError(AMSError&&) noexcept = default;

  /// \brief Defaulted copy assignment.
  AMSError& operator=(const AMSError&) = default;

  /// \brief Defaulted move assignment.
  AMSError& operator=(AMSError&&) noexcept = default;

  /// \brief Return the error message.
  const std::string& getMessage() const { return Message; }

  /// \brief Return the error type.
  AMSErrorType getType() const { return Type; }

  /// \brief Return the source file where the error was created (may be empty).
  const std::string& getFile() const { return File; }

  /// \brief Return the source line where the error was created (may be zero).
  int getLine() const { return Line; }

private:
  AMSErrorType Type;
  std::string Message;
  std::string File;
  int Line = 0;
};

/// \brief Expected type used throughout AMS.
///
/// This is a thin alias over tl::expected with AMSError as the error type.
/// Client code should use AMSExpected<T> instead of tl::expected directly.
template <typename T>
using AMSExpected = tl::expected<T, AMSError>;

using AMSStatus = AMSExpected<void>;

/// \brief Convenience helper to construct an AMSExpected<T> that holds an error.
///
/// This wraps tl::unexpected internally but does not expose it in the API.
///
/// \tparam T The expected value type.
/// \param error Error value to store.
/// \return AMSExpected<T> containing the given error.
template <typename T>
inline AMSExpected<T> MakeError(AMSError Error)
{
  return AMSExpected<T>(tl::unexpected(std::move(Error)));
}

/// \brief Stream insertion operator for AMSError.
///
/// When AMS_DEBUG is defined, this prints the message plus file:line (if
/// available). Otherwise, only the message is printed.
inline std::ostream& operator<<(std::ostream& OS, const AMSError& Error)
{
  if (!amsDebug()) {
    OS << Error.getMessage();
    return OS;
  }

  if (!Error.getFile().empty()) {
    OS << " [" << Error.getFile();
    if (Error.getLine() > 0) OS << ":" << Error.getLine();
    OS << "]";
  }
  OS << Error.getMessage();
  return OS;
}

#ifdef AMS_DEBUG
#define __AMS_FILE__ __FILE__
#define __AMS_LINE__ __LINE__
#else
#define __AMS_FILE__ ""
#define __AMS_LINE__ -1
#endif

/// \brief Helper macro to construct an AMSError with current file/line.
///
/// Example:
/// \code
/// return MakeError<int>(AMS_MAKE_ERROR(AMSErrorType::InvalidArgument,
///                                      "Bad configuration"));
/// \endcode
#define AMS_MAKE_ERROR_OBJ(Type_, Message_) \
  ::ams::AMSError((Type_), (Message_), __AMS_FILE__, __AMS_LINE__)

#define AMS_MAKE_ERROR(Type_, Message_) \
  ::tl::unexpected(AMS_MAKE_ERROR_OBJ((Type_), (Message_)))

}  // namespace ams
