#include <chrono>
#include <iostream>
#include <string_view>

template <typename T>
struct ScopedTimer {
  using clock = std::chrono::steady_clock;
  std::string_view name;
  clock::time_point start;
  T elements;

  explicit ScopedTimer(std::string_view n, T elements)
      : name(n), elements(elements), start(clock::now())
  {
  }
  ~ScopedTimer()
  {
    auto end = clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count();
    std::cerr << name << ": " << ns << " ns " << ns / elements << " Elements/ns"
              << "\n";
  }
};
