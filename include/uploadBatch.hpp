#pragma once

#include "buffer.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

// Copies into GPU buffers, gathered in one staging buffer, and copies from one place of the GPU
// to another, all sent in one submission. It records with a command pool of its own and waits
// for its own fence: it may be sent from any thread, and only waits for its copies.
class UploadBatch {
	private:
		struct Copy {
			VkBuffer	 target;
			VkDeviceSize offset;
			VkDeviceSize from;
			VkDeviceSize size;
			VkBuffer	 source = VK_NULL_HANDLE; // the staging buffer when null
		};
		std::vector<uint8_t> _bytes;
		std::vector<Copy>	 _copies;

	public:
		bool empty() const { return _copies.empty(); }
		void reserve(size_t bytes) { _bytes.reserve(bytes); }

		// `size` bytes of `data` (zeros when it is null) at `offset` of `target`.
		void add(VkBuffer target, VkDeviceSize offset, const void* data, VkDeviceSize size) {
			if (size == 0)
				return;
			const size_t from = _bytes.size();
			_bytes.resize(from + size);
			if (data)
				std::memcpy(_bytes.data() + from, data, size);
			else
				std::memset(_bytes.data() + from, 0, size);
			_copies.push_back({target, offset, from, size});
		}

		// `size` bytes at `from` of `source` to `offset` of `target`, which do not overlap.
		void move(VkBuffer source, VkDeviceSize from, VkBuffer target, VkDeviceSize offset, VkDeviceSize size) {
			if (size > 0)
				_copies.push_back({target, offset, from, size, source});
		}

		// Sends everything and waits for it; the batch is empty again.
		bool submit(renderApi::device::GPU* gpu) {
			if (_copies.empty())
				return true;
			renderApi::Buffer staging;
			if (!_bytes.empty()) {
				staging	 = renderApi::createStagingBuffer(gpu, _bytes.size());
				void* to = staging.isValid() ? staging.map() : nullptr;
				if (!to)
					return false;
				std::memcpy(to, _bytes.data(), _bytes.size());
				staging.unmap();
			}

			VkDevice				device = gpu->device;
			VkCommandPool			pool   = VK_NULL_HANDLE;
			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.flags			  = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
			poolInfo.queueFamilyIndex = gpu->getGraphicsQueueFamilyIndex();
			if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
				return false;

			VkCommandBuffer				cmd = VK_NULL_HANDLE;
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool		 = pool;
			allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = 1;
			VkFence			  fence = VK_NULL_HANDLE;
			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			bool sent		= vkAllocateCommandBuffers(device, &allocInfo, &cmd) == VK_SUCCESS &&
						vkCreateFence(device, &fenceInfo, nullptr, &fence) == VK_SUCCESS;
			if (sent) {
				VkCommandBufferBeginInfo begin{};
				begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
				vkBeginCommandBuffer(cmd, &begin);
				for (const Copy& copy : _copies) {
					const VkBufferCopy region{copy.from, copy.offset, copy.size};
					vkCmdCopyBuffer(cmd, copy.source ? copy.source : staging.getHandle(), copy.target, 1, &region);
				}
				vkEndCommandBuffer(cmd);

				VkSubmitInfo submitInfo{};
				submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers	  = &cmd;
				{
					// Every submission to the graphics queue holds this lock (the frames too).
					const std::lock_guard<std::mutex> lock(gpu->queueMutex);
					sent = vkQueueSubmit(gpu->getGraphicsQueue(), 1, &submitInfo, fence) == VK_SUCCESS;
				}
				if (sent)
					sent = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
			}
			if (fence)
				vkDestroyFence(device, fence, nullptr);
			vkDestroyCommandPool(device, pool, nullptr); // frees the command buffer too
			_bytes.clear();
			_copies.clear();
			return sent;
		}
};
