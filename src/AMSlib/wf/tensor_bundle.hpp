#pragma once

#include <ATen/ATen.h>

#include <string>
#include <utility>
#include <vector>

namespace ams
{

/// A lightweight container that groups named tensors together.
/// This is the primary structure used to represent inputs,
/// in-out parameters, and outputs inside AMS evaluation pipelines.
struct TensorBundle {

  /// A single named tensor.
  struct Item {
    std::string name;
    at::Tensor tensor;

    Item(std::string n, at::Tensor t) : name(std::move(n)), tensor(std::move(t))
    {
    }
  };

  /// Ordered list of items.
  std::vector<Item> items;

  /// Default construction.
  TensorBundle() = default;

  /// Move operations for efficiency.
  TensorBundle(TensorBundle&&) noexcept = default;
  TensorBundle& operator=(TensorBundle&&) noexcept = default;

  /// Copy operations allowed (torch::Tensor has cheap refcounted semantics).
  TensorBundle(const TensorBundle&) = default;
  TensorBundle& operator=(const TensorBundle&) = default;

  /// Add a named tensor to the bundle.
  void add(std::string name, at::Tensor t)
  {
    items.emplace_back(std::move(name), std::move(t));
  }

  /// Number of tensors in the bundle.
  size_t size() const noexcept { return items.size(); }

  /// Random access to items.
  Item& operator[](size_t i) noexcept { return items[i]; }

  const Item& operator[](size_t i) const noexcept { return items[i]; }

  /// Iterators.
  auto begin() noexcept { return items.begin(); }
  auto end() noexcept { return items.end(); }
  auto begin() const noexcept { return items.begin(); }
  auto end() const noexcept { return items.end(); }

  /// Check if empty.
  bool empty() const noexcept { return items.empty(); }

  /// Remove all items.
  void clear() noexcept { items.clear(); }

  /// Find an item by name. Returns pointer to Item if found, nullptr otherwise.
  Item* find(const std::string& name) noexcept {
    for (auto& item : items) {
      if (item.name == name) {
        return &item;
      }
    }
    return nullptr;
  }

  /// Const overload of find.
  const Item* find(const std::string& name) const noexcept {
    for (const auto& item : items) {
      if (item.name == name) {
        return &item;
      }
    }
    return nullptr;
  }
};

}  // namespace ams
