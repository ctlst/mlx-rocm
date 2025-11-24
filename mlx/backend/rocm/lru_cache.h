// Copyright © 2025 Apple Inc.

#pragma once

#include <list>
#include <optional>
#include <unordered_map>

namespace mlx::core::rocm {

template <typename Key, typename Value>
class LRUCache {
 public:
  LRUCache(size_t capacity = 100) : capacity_(capacity) {}

  void insert(const Key& key, Value&& value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      // Update existing
      list_.erase(it->second.second);
      list_.push_front(key);
      it->second = {std::move(value), list_.begin()};
    } else {
      // Insert new
      if (map_.size() >= capacity_) {
        // Evict oldest
        map_.erase(list_.back());
        list_.pop_back();
      }
      list_.push_front(key);
      map_[key] = {std::move(value), list_.begin()};
    }
  }

  std::optional<Value*> get(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    // Move to front
    list_.erase(it->second.second);
    list_.push_front(key);
    it->second.second = list_.begin();
    return &it->second.first;
  }

  bool contains(const Key& key) const {
    return map_.find(key) != map_.end();
  }

  void clear() {
    map_.clear();
    list_.clear();
  }

  size_t size() const {
    return map_.size();
  }

 private:
  size_t capacity_;
  std::list<Key> list_;
  std::unordered_map<Key, std::pair<Value, typename std::list<Key>::iterator>> map_;
};

} // namespace mlx::core::rocm

