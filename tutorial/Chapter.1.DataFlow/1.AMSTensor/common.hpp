#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

class ExampleArgs
{
public:
  void PrintOptions() const
  {
    std::cout << "Available options:\n";
    for (const auto& opt : registered_) {
      std::cout << "  ";
      for (size_t i = 0; i < opt.keys.size(); ++i) {
        std::cout << opt.keys[i];
        if (i + 1 < opt.keys.size()) std::cout << ", ";
      }
      std::cout << "\n    " << opt.help << "\n";
    }
  }

  void Parse(int argc, char** argv)
  {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg[0] != '-') continue;

      std::string key = arg;
      std::string value;

      // If the next item isn't an option, treat it as a value
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        value = argv[++i];
      } else {
        value = "true";  // Boolean flag
      }

      options_[key] = value;
    }

    // Set parsed values into variables
    for (auto& opt : registered_) {
      const auto& keys = opt.keys;
      for (const auto& key : keys) {
        if (options_.count(key)) {
          opt.setter(options_[key]);
          opt.wasset = true;
          break;
        }
      }
    }
  }

  template <typename T>
  void AddOption(T* out,
                 std::string short_opt,
                 std::string long_opt,
                 std::string help,
                 bool required = true)
  {
    registered_.push_back(
        {{short_opt, long_opt},
         [out](const std::string& val) { parseValue(val, out); },
         [out]() { return toString(*out); },
         help,
         required,
         false});
  }

  bool Good() const
  {
    for (const auto& opt : registered_) {
      if (opt.required && !opt.wasset) {
        return false;
      }
    }
    return true;
  }

  void PrintUsage() const
  {
    std::cout << "Parsed arguments:\n";
    for (const auto& opt : registered_) {
      std::cout << "  ";
      for (size_t i = 0; i < opt.keys.size(); ++i) {
        std::cout << opt.keys[i];
        if (i + 1 < opt.keys.size()) std::cout << ", ";
      }
      std::cout << " = " << opt.getter() << "\n";
    }
  }

private:
  struct RegisteredOption {
    std::vector<std::string> keys;
    std::function<void(const std::string&)> setter;
    std::function<std::string()> getter;
    std::string help;
    bool required;
    bool wasset;
  };

  std::vector<RegisteredOption> registered_;
  std::unordered_map<std::string, std::string> options_;

  // Parser helper
  template <typename T>
  static void parseValue(const std::string& s, T* out);

  static void parseValue(const std::string& s, std::string* out) { *out = s; }

  static void parseValue(const std::string& s, int* out)
  {
    *out = std::stoi(s);
  }

  static void parseValue(const std::string& s, bool* out)
  {
    *out = (s == "true" || s == "1");
  }

  static void parseValue(const std::string& s, double* out)
  {
    *out = std::stod(s);
  }


  static std::string toString(const std::string& val) { return val; }
  static std::string toString(bool val) { return val ? "true" : "false"; }
  static std::string toString(int val) { return std::to_string(val); }
  static std::string toString(double val) { return std::to_string(val); }
};
