#include "geometryPool.hpp"

#include "renderDevice.hpp"

#include <algorithm>
#include <iostream>

namespace {
	constexpr VkDeviceSize kRefill = 4;
	constexpr VkDeviceSize kLargeCell = 16ull << 20;

}

bool GeometryPool::init(renderApi::device::GPU* gpu) {
	const std::lock_guard<std::mutex> lock(_mutex);
	_gpu = gpu;
	return !_buffers.empty() || addBuffer(kBufferSize, false).has_value();
}

void GeometryPool::destroy() {
	const std::lock_guard<std::mutex> lock(_mutex);
	_buffers.clear();
	_released.clear();
}

std::optional<uint32_t> GeometryPool::addBuffer(VkDeviceSize size, bool dedicated) {
	renderApi::Buffer buffer;
	if (!_gpu || !buffer.create(_gpu, size, renderApi::BufferType::GEOMETRY, renderApi::BufferUsage::STATIC))
		return std::nullopt;

	PoolBuffer added{std::move(buffer), dedicated, {}};
	if (!dedicated)
		added.free[kOrders - 1].insert(0);

	const auto slot	 = std::find_if(_buffers.begin(), _buffers.end(), [](const PoolBuffer& pool) { return !pool.buffer.isValid(); });
	const auto index = static_cast<uint32_t>(slot - _buffers.begin());
	if (slot != _buffers.end())
		*slot = std::move(added);
	else
		_buffers.push_back(std::move(added));
	_changed = true;
	std::cout << "GeometryPool: buffer " << index << " made (" << ((size + (1 << 20) - 1) >> 20) << " MiB" << (dedicated ? ", one piece" : "")
			  << ")" << std::endl;
	return index;
}

std::optional<GeometryPool::Cell> GeometryPool::take(uint32_t buffer, uint8_t order) {
	auto& free = _buffers[buffer].free;
	uint8_t larger = order;
	while (larger < kOrders && free[larger].empty())
		++larger;
	if (larger == kOrders)
		return std::nullopt;

	const VkDeviceSize offset = *free[larger].begin();
	free[larger].erase(free[larger].begin());
	while (larger > order) {
		--larger;
		free[larger].insert(offset + (kMinCell << larger));
	}
	_changed = true;
	return Cell{buffer, order, offset, kMinCell << order};
}

std::optional<GeometryPool::Cell> GeometryPool::allocate(VkDeviceSize bytes) {
	const std::lock_guard<std::mutex> lock(_mutex);
	if (bytes == 0)
		return Cell{};

	if (bytes > kBufferSize) {
		const auto buffer = addBuffer(bytes, true);
		if (!buffer)
			return std::nullopt;
		return Cell{*buffer, kDedicated, 0, bytes};
	}

	uint8_t order = 0;
	while ((kMinCell << order) < bytes)
		++order;

	for (uint32_t buffer = 0; buffer < _buffers.size(); ++buffer)
		if (_buffers[buffer].buffer.isValid() && !_buffers[buffer].dedicated)
			if (const auto cell = take(buffer, order))
				return cell;

	if (statsLocked().free >= (kMinCell << order))
		_needlessBuffer = true;
	const auto buffer = addBuffer(kBufferSize, false);
	if (!buffer)
		return std::nullopt;
	return take(*buffer, order);
}

void GeometryPool::release(const Cell& cell) {
	const std::lock_guard<std::mutex> lock(_mutex);
	if (cell.size > 0)
		_released.push_back(cell);
}

void GeometryPool::releaseNow(const Cell& cell) {
	const std::lock_guard<std::mutex> lock(_mutex);
	if (cell.size > 0)
		free(cell);
}

void GeometryPool::freeReleased() {
	const std::lock_guard<std::mutex> lock(_mutex);
	for (const Cell& cell : _released)
		free(cell);
	_released.clear();
}

void GeometryPool::free(const Cell& cell) {
	PoolBuffer& pool = _buffers[cell.buffer];
	_changed		 = true;
	if (cell.order == kDedicated) {
		pool.buffer.destroy();
		return;
	}

	VkDeviceSize offset = cell.offset;
	uint8_t		 order	= cell.order;
	while (order + 1 < kOrders) {
		const VkDeviceSize buddy = offset ^ (kMinCell << order);
		if (!pool.free[order].erase(buddy))
			break;
		offset = std::min(offset, buddy);
		++order;
	}
	pool.free[order].insert(offset);

	if (order == kOrders - 1 && cell.buffer != 0) {
		pool.buffer.destroy();
		pool.free = {};
		std::cout << "GeometryPool: buffer " << cell.buffer << " given back" << std::endl;
	}
}

GeometryPool::Stats GeometryPool::stats() const {
	const std::lock_guard<std::mutex> lock(_mutex);
	return statsLocked();
}

VkBuffer GeometryPool::handle(uint32_t index) const {
	const std::lock_guard<std::mutex> lock(_mutex);
	return _buffers[index].buffer.getHandle();
}

VkDeviceAddress GeometryPool::address(uint32_t index) const {
	const std::lock_guard<std::mutex> lock(_mutex);
	return _buffers[index].buffer.getDeviceAddress();
}

GeometryPool::Stats GeometryPool::statsLocked() const {
	Stats stats;
	for (const PoolBuffer& pool : _buffers) {
		if (!pool.buffer.isValid() || pool.dedicated)
			continue;
		++stats.buffers;
		for (uint8_t order = 0; order < kOrders; ++order) {
			const VkDeviceSize size = kMinCell << order;
			stats.free += pool.free[order].size() * size;
			if (!pool.free[order].empty())
				stats.largestFree = std::max(stats.largestFree, size);
		}
	}
	stats.used = stats.buffers * kBufferSize - stats.free;
	return stats;
}

bool GeometryPool::wantsCompaction() const {
	const std::lock_guard<std::mutex> lock(_mutex);
	if (!_changed)
		return false;
	const Stats now = statsLocked();
	const bool	emptiable = now.buffers > 1 && now.used * kRefill <= (now.buffers - 1) * kBufferSize * (kRefill - 1);
	const bool	scattered = now.free >= 4 * kLargeCell && now.largestFree < kLargeCell;
	return emptiable || scattered || _needlessBuffer;
}

void GeometryPool::compactionStarted() {
	const std::lock_guard<std::mutex> lock(_mutex);
	const Stats now = statsLocked();
	std::cout << "GeometryPool: compacting (" << now.buffers << " buffers, " << (now.used >> 20) << " MiB used, " << (now.free >> 20)
			  << " MiB free, largest free cell " << (now.largestFree >> 10) << " KiB)" << std::endl;
	_changed		= false;
	_needlessBuffer = false;
}

std::optional<GeometryPool::Cell> GeometryPool::relocate(const Cell& cell) {
	const std::lock_guard<std::mutex> lock(_mutex);
	if (cell.size == 0 || cell.order == kDedicated)
		return std::nullopt;

	for (uint32_t buffer = 0; buffer <= cell.buffer && buffer < _buffers.size(); ++buffer) {
		PoolBuffer& pool = _buffers[buffer];
		if (!pool.buffer.isValid() || pool.dedicated)
			continue;
		uint8_t		 best	= kOrders;
		VkDeviceSize offset = 0;
		for (uint8_t order = cell.order; order < kOrders; ++order) {
			if (pool.free[order].empty())
				continue;
			const VkDeviceSize first = *pool.free[order].begin();
			if ((buffer < cell.buffer || first < cell.offset) && (best == kOrders || first < offset)) {
				best   = order;
				offset = first;
			}
		}
		if (best == kOrders)
			continue;
		pool.free[best].erase(offset);
		while (best > cell.order) {
			--best;
			pool.free[best].insert(offset + (kMinCell << best));
		}
		_changed = true;
		return Cell{buffer, cell.order, offset, cell.size};
	}
	return std::nullopt;
}
