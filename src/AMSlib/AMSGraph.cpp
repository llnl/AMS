#include "AMSGraph.hpp"

#include <stdexcept>

namespace ams
{
namespace
{

static std::size_t hashCombine(std::size_t seed, std::size_t value) noexcept
{
  seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2);
  return seed;
}

}  // namespace

EdgeType::EdgeType(std::string src_, std::string rel_, std::string dst_)
  : src(std::move(src_)), rel(std::move(rel_)), dst(std::move(dst_))
{
}

bool EdgeType::operator==(const EdgeType& other) const noexcept
{
  return src == other.src && rel == other.rel && dst == other.dst;
}

std::size_t EdgeTypeHash::operator()(const EdgeType& e) const noexcept
{
  std::size_t seed = std::hash<std::string>{}(e.src);
  seed = hashCombine(seed, std::hash<std::string>{}(e.rel));
  seed = hashCombine(seed, std::hash<std::string>{}(e.dst));
  return seed;
}

std::string edgeTypeToString(const EdgeType& e)
{
  return e.src + "__" + e.rel + "__" + e.dst;
}

ams::EdgeType edgeTypeFromString(const std::string& key)
{
  const std::string delim = "__";

  auto p1 = key.find(delim);
  if (p1 == std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: missing first delimiter");
  }

  auto p2 = key.find(delim, p1 + delim.size());
  if (p2 == std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: missing second delimiter");
  }

  if (key.find(delim, p2 + delim.size()) != std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: too many delimiters");
  }

  std::string src = key.substr(0, p1);
  std::string rel = key.substr(p1 + delim.size(), p2 - (p1 + delim.size()));
  std::string dst = key.substr(p2 + delim.size());

  if (src.empty() || rel.empty() || dst.empty()) {
    throw std::runtime_error("edgeTypeFromString: empty src/rel/dst component");
  }

  return ams::EdgeType{std::move(src), std::move(rel), std::move(dst)};
}

bool containsTensor(const AMSTensorMap& store, const std::string& name)
{
  return store.find(name) != store.end();
}

AMSTensor* findTensor(AMSTensorMap& store, const std::string& name) noexcept
{
  auto it = store.find(name);
  if (it == store.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensor* findTensor(const AMSTensorMap& store,
                            const std::string& name) noexcept
{
  auto it = store.find(name);
  if (it == store.end()) {
    return nullptr;
  }
  return &it->second;
}

void insertTensor(AMSTensorMap& store, std::string name, AMSTensor&& tensor)
{
  auto [it, inserted] = store.emplace(std::move(name), std::move(tensor));
  if (!inserted) {
    throw std::runtime_error("insertTensor: tensor name already exists");
  }
}

void insertOrAssignTensor(AMSTensorMap& store,
                          std::string name,
                          AMSTensor&& tensor)
{
  auto it = store.find(name);
  if (it == store.end()) {
    store.emplace(std::move(name), std::move(tensor));
  } else {
    it->second = std::move(tensor);
  }
}

bool AMSHeterogeneousGraph::containsNodeStore(const std::string& name) const
{
  return node_stores.find(name) != node_stores.end();
}

bool AMSHeterogeneousGraph::containsEdgeStore(const EdgeType& edge_type) const
{
  return edge_stores.find(edge_type) != edge_stores.end();
}

AMSTensorMap& AMSHeterogeneousGraph::getOrCreateNodeStore(std::string name)
{
  auto [it, inserted] = node_stores.try_emplace(std::move(name), AMSTensorMap{});
  (void)inserted;
  return it->second;
}

AMSTensorMap& AMSHeterogeneousGraph::getOrCreateEdgeStore(EdgeType edge_type)
{
  auto [it, inserted] =
      edge_stores.try_emplace(std::move(edge_type), AMSTensorMap{});
  (void)inserted;
  return it->second;
}

AMSTensorMap* AMSHeterogeneousGraph::findNodeStore(
    const std::string& name) noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorMap* AMSHeterogeneousGraph::findNodeStore(
    const std::string& name) const noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

AMSTensorMap* AMSHeterogeneousGraph::findEdgeStore(
    const EdgeType& edge_type) noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorMap* AMSHeterogeneousGraph::findEdgeStore(
    const EdgeType& edge_type) const noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace ams
