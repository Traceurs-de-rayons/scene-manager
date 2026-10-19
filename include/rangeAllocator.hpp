#pragma once

#include <cstdint>
#include <iterator>
#include <map>
#include <optional>

// Ranges of a buffer, counted in elements (vertices, indices), handed out and given back: the
// first free range large enough is taken, and a range given back merges with its free
// neighbors. What was handed out never moves; the buffer only grows when asked (grow()).
class RangeAllocator {
	private:
		std::map<uint32_t, uint32_t> _free; // first element -> count, never touching each other
		uint32_t					 _capacity = 0;

	public:
		// Everything free again, `capacity` elements.
		void reset(uint32_t capacity) {
			_free.clear();
			_capacity = capacity;
			if (capacity > 0)
				_free[0] = capacity;
		}

		uint32_t capacity() const { return _capacity; }

		// The first element of `count` free ones, or nothing when no free range is that large.
		// A count of 0 takes nothing.
		std::optional<uint32_t> allocate(uint32_t count) {
			if (count == 0)
				return 0;
			for (auto it = _free.begin(); it != _free.end(); ++it) {
				if (it->second < count)
					continue;
				const uint32_t first = it->first, left = it->second - count;
				_free.erase(it);
				if (left > 0)
					_free[first + count] = left;
				return first;
			}
			return std::nullopt;
		}

		void release(uint32_t first, uint32_t count) {
			if (count == 0)
				return;
			auto it = _free.emplace(first, count).first;
			if (const auto next = std::next(it); next != _free.end() && it->first + it->second == next->first) {
				it->second += next->second;
				_free.erase(next);
			}
			if (it != _free.begin()) {
				const auto previous = std::prev(it);
				if (previous->first + previous->second == it->first) {
					previous->second += it->second;
					_free.erase(it);
				}
			}
		}

		// More room at the end; nothing moves.
		void grow(uint32_t capacity) {
			if (capacity <= _capacity)
				return;
			const uint32_t added = capacity - _capacity;
			const uint32_t first = _capacity;
			_capacity			 = capacity;
			release(first, added);
		}
};
