#pragma once

#include "buffer.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

namespace renderApi::device {
	struct GPU;
}

// The GPU memory of the geometry of every scene: buffers of kBufferSize bytes, handed out in
// cells. A piece of geometry (its vertices, its skin, its indices, one after the other) takes
// one cell whole, of the smallest power of two size from kMinCell that holds it: a buffer
// starts as one cell, cut in halves until a cell of the size asked is there, and two free
// halves join again (a buddy allocator). Nothing is bound: the shaders read the vertices and
// the indices by address, so adding or removing geometry never makes a buffer again nor copies
// one (only a compaction moves pieces, on purpose). When no buffer has a
// cell large enough one is added; a piece larger than a buffer gets a buffer of its own.
//
// Every call may come from any thread (a loading thread allocates while the main thread draws).
//
// Cells of fixed counts per size (256 of 256 KiB, 48 of 2 MiB, 6 of 16 MiB) were tried first:
// the game has some 1800 pieces, most under 32 KiB, and they took 6 buffers (1.5 GiB) for 53 MB.
class GeometryPool {
	public:
		static constexpr VkDeviceSize kMinCell	  = 4ull << 10;
		static constexpr VkDeviceSize kBufferSize = 256ull << 20;
		static constexpr uint8_t	  kOrders	  = 17; // cells of kMinCell << 0 .. 16: 4 KiB to 256 MiB
		static constexpr uint8_t	  kDedicated  = 0xff;
		static_assert(kMinCell << (kOrders - 1) == kBufferSize);

		// Where a piece of geometry is. A size of 0 is no cell (nothing to hold).
		struct Cell {
			uint32_t	 buffer = 0;
			uint8_t		 order	= 0; // size kMinCell << order, or kDedicated
			VkDeviceSize offset = 0; // bytes, from the start of the buffer
			VkDeviceSize size	= 0;
		};

		GeometryPool() = default;
		~GeometryPool() { destroy(); }
		GeometryPool(const GeometryPool&)			 = delete;
		GeometryPool& operator=(const GeometryPool&) = delete;

		// Makes the first buffer, so that the first geometry does not wait for it.
		bool init(renderApi::device::GPU* gpu);
		void destroy();

		// A free cell of at least `bytes`. Nothing when the GPU has no memory left.
		std::optional<Cell> allocate(VkDeviceSize bytes);
		// Gives a cell back. The draw list still reads it: it is only free once that list is
		// replaced (freeReleased()).
		void release(const Cell& cell);
		// Gives a cell back at once: the caller waited for the device to be idle.
		void releaseNow(const Cell& cell);
		// The cells released are free: the draw list was just made again under GPU::lockTasks(),
		// and the GPU thread waits for the frames it runs before it lets that lock go, so none of
		// them reads the old list any more (SceneDescriptor::update()).
		void freeReleased();

		// Compaction. A piece of geometry can be moved to a free cell placed before its own (a
		// lower buffer, or lower in its buffer): the pieces pack at the start of the first
		// buffers, the free cells join into large ones, and a buffer left empty is given back.
		struct Stats {
			uint32_t	 buffers = 0; // shared ones (not the buffers of a single piece)
			VkDeviceSize used	 = 0; // bytes of the cells handed out in them
			VkDeviceSize free	 = 0;
			VkDeviceSize largestFree = 0; // the largest free cell
		};
		Stats stats() const;
		// Whether moving pieces would give memory back or make room: a buffer could be emptied,
		// a buffer was added while the others had the room in small cells, or much is free and no
		// large cell. Only looked at again after something was taken or freed.
		bool wantsCompaction() const;
		// A compaction pass starts: wantsCompaction() is false until something changes again.
		void compactionStarted();
		// A free cell of the size of `cell` placed before it, taken; nothing when there is none.
		std::optional<Cell> relocate(const Cell& cell);

		VkBuffer		handle(uint32_t index) const;
		VkDeviceAddress address(uint32_t index) const;

	private:
		struct PoolBuffer {
			renderApi::Buffer buffer; // not valid: a slot given back, taken again by the next buffer
			bool			  dedicated = false;
			std::array<std::set<VkDeviceSize>, kOrders> free; // offsets of the free cells of each order
		};

		mutable std::mutex		_mutex; // over everything below
		renderApi::device::GPU* _gpu = nullptr;
		std::vector<PoolBuffer> _buffers;
		std::vector<Cell>		_released;
		bool					_changed	= false; // something taken or freed since compactionStarted()
		bool					_needlessBuffer = false; // a buffer was added though the others had the room

		std::optional<uint32_t> addBuffer(VkDeviceSize size, bool dedicated);
		std::optional<Cell>		take(uint32_t buffer, uint8_t order);
		void					free(const Cell& cell);
		Stats					statsLocked() const;
};
