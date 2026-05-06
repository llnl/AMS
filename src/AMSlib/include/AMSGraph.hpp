#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "AMSTensor.hpp"

namespace ams
{

using AMSTensorMap = std::unordered_map<std::string, AMSTensor>;
using AMSHomogeneousGraph = AMSTensorMap;

struct EdgeType {
  std::string src;
  std::string rel;
  std::string dst;

  EdgeType() = default;
  EdgeType(std::string src_, std::string rel_, std::string dst_);

  bool operator==(const EdgeType& other) const noexcept;
  bool operator!=(const EdgeType& other) const noexcept
  {
    return !(*this == other);
  }
};

struct EdgeTypeHash {
  std::size_t operator()(const EdgeType& e) const noexcept;
};

std::string edgeTypeToString(const EdgeType& e);
ams::EdgeType edgeTypeFromString(const std::string& key);

// Generic helpers for any named tensor store.
// These are useful both for AMSHomogeneousGraph and for the stores inside
// AMSHeterogeneousGraph.
bool containsTensor(const AMSTensorMap& store, const std::string& name);

AMSTensor* findTensor(AMSTensorMap& store, const std::string& name) noexcept;
const AMSTensor* findTensor(const AMSTensorMap& store,
                            const std::string& name) noexcept;

// Insert a new tensor. Throws if the name already exists.
void insertTensor(AMSTensorMap& store, std::string name, AMSTensor&& tensor);

// Insert or replace an existing tensor.
void insertOrAssignTensor(AMSTensorMap& store,
                          std::string name,
                          AMSTensor&& tensor);

struct AMSHeterogeneousGraph {
  using NodeStoreMap = std::unordered_map<std::string, AMSTensorMap>;
  using EdgeStoreMap = std::unordered_map<EdgeType, AMSTensorMap, EdgeTypeHash>;

  NodeStoreMap node_stores;
  EdgeStoreMap edge_stores;
  AMSTensorMap global_store;

  AMSHeterogeneousGraph() = default;
  AMSHeterogeneousGraph(const AMSHeterogeneousGraph&) = delete;
  AMSHeterogeneousGraph& operator=(const AMSHeterogeneousGraph&) = delete;
  AMSHeterogeneousGraph(AMSHeterogeneousGraph&&) noexcept = default;
  AMSHeterogeneousGraph& operator=(AMSHeterogeneousGraph&&) noexcept = default;
  ~AMSHeterogeneousGraph() = default;

  bool containsNodeStore(const std::string& name) const;
  bool containsEdgeStore(const EdgeType& edge_type) const;

  AMSTensorMap& getOrCreateNodeStore(std::string name);
  AMSTensorMap& getOrCreateEdgeStore(EdgeType edge_type);

  AMSTensorMap* findNodeStore(const std::string& name) noexcept;
  const AMSTensorMap* findNodeStore(const std::string& name) const noexcept;

  AMSTensorMap* findEdgeStore(const EdgeType& edge_type) noexcept;
  const AMSTensorMap* findEdgeStore(const EdgeType& edge_type) const noexcept;
};

}  // namespace ams
