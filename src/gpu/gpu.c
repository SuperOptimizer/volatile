#include "gpu/gpu.h"
#include "gpu/vk.h"
#include "core/log.h"

#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// gpu_device
// ---------------------------------------------------------------------------

struct gpu_device {
  gpu_backend_t  backend;
  vk_context    *vk;
};

// ---------------------------------------------------------------------------
// gpu_buffer
// ---------------------------------------------------------------------------

struct gpu_buffer {
  gpu_device    *dev;
  VkBuffer       buf;
  VkDeviceMemory mem;
  size_t         size;
  bool           host_visible;
  void          *mapped;       // non-NULL while mapped
};

// ---------------------------------------------------------------------------
// gpu_pipeline
// ---------------------------------------------------------------------------

struct gpu_pipeline {
  gpu_device       *dev;
  VkPipeline        pipeline;       // uses layout (16 storage buffer bindings)
  VkPipeline        pipeline_empty; // uses layout_empty (0 bindings, for 0-buffer dispatch)
  VkPipelineLayout  layout;         // layout with 16 storage buffer bindings
  VkPipelineLayout  layout_empty;   // layout with 0 bindings
  VkDescriptorSetLayout ds_layout;
  VkDescriptorPool  ds_pool;
};

// ---------------------------------------------------------------------------
// Vulkan helpers
// ---------------------------------------------------------------------------

// Find a memory type index matching required_bits and desired properties.
static int find_memory_type(VkPhysicalDevice pd, uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
      return (int)i;
  }
  return -1;
}

static VkCommandPool make_cmd_pool(VkDevice device, uint32_t qf) {
  VkCommandPoolCreateInfo ci = {
    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = qf,
    .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
  };
  VkCommandPool pool = VK_NULL_HANDLE;
  vkCreateCommandPool(device, &ci, NULL, &pool);
  return pool;
}

static VkCommandBuffer begin_one_shot(VkDevice device, VkCommandPool pool) {
  VkCommandBufferAllocateInfo ai = {
    .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool        = pool,
    .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  VkCommandBuffer cb = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(device, &ai, &cb);
  VkCommandBufferBeginInfo bi = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(cb, &bi);
  return cb;
}

static void end_one_shot(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cb) {
  vkEndCommandBuffer(cb);
  VkSubmitInfo si = {
    .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers    = &cb,
  };
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &cb);
}

// ---------------------------------------------------------------------------
// gpu_init / gpu_shutdown
// ---------------------------------------------------------------------------

gpu_device *gpu_init(gpu_backend_t preferred) {
  if (preferred == GPU_BACKEND_METAL) {
    LOG_WARN("gpu_init: Metal backend not yet implemented");
    return NULL;
  }
  if (preferred == GPU_BACKEND_DX12) {
    LOG_WARN("gpu_init: DX12 backend not yet implemented");
    return NULL;
  }

  // GPU_BACKEND_NONE or GPU_BACKEND_VULKAN: try Vulkan
  vk_config cfg = { .validation = false, .headless = true };
  vk_context *vk = vk_init(cfg);
  if (!vk) {
    LOG_WARN("gpu_init: Vulkan initialisation failed");
    return NULL;
  }

  gpu_device *dev = calloc(1, sizeof(*dev));
  if (!dev) { vk_shutdown(vk); return NULL; }
  dev->backend = GPU_BACKEND_VULKAN;
  dev->vk      = vk;
  LOG_INFO("gpu_init: Vulkan backend ready (%s)", vk_device_name(vk));
  return dev;
}

void gpu_shutdown(gpu_device *dev) {
  if (!dev) return;
  vk_shutdown(dev->vk);
  free(dev);
}

gpu_backend_t gpu_active_backend(const gpu_device *dev) {
  assert(dev);
  return dev->backend;
}

const char *gpu_device_name(const gpu_device *dev) {
  assert(dev);
  return vk_device_name(dev->vk);
}

// ---------------------------------------------------------------------------
// gpu_buffer
// ---------------------------------------------------------------------------

gpu_buffer *gpu_buffer_create(gpu_device *dev, size_t size, bool host_visible) {
  assert(dev && size > 0);
  VkDevice vkdev   = vk_device(dev->vk);
  VkPhysicalDevice pd = vk_physical_device(dev->vk);

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                           | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                           | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VkBufferCreateInfo bci = {
    .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size        = size,
    .usage       = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VkBuffer vkbuf = VK_NULL_HANDLE;
  if (vkCreateBuffer(vkdev, &bci, NULL, &vkbuf) != VK_SUCCESS) {
    LOG_WARN("gpu_buffer_create: vkCreateBuffer failed");
    return NULL;
  }

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(vkdev, vkbuf, &mr);

  VkMemoryPropertyFlags mem_props = host_visible
    ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  int mt = find_memory_type(pd, mr.memoryTypeBits, mem_props);
  if (mt < 0 && !host_visible) {
    // fall back to host-visible if no device-local available (e.g. integrated GPU)
    mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mt = find_memory_type(pd, mr.memoryTypeBits, mem_props);
    if (mt >= 0) host_visible = true;
  }
  if (mt < 0) {
    LOG_WARN("gpu_buffer_create: no suitable memory type");
    vkDestroyBuffer(vkdev, vkbuf, NULL);
    return NULL;
  }

  VkMemoryAllocateInfo mai = {
    .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize  = mr.size,
    .memoryTypeIndex = (uint32_t)mt,
  };
  VkDeviceMemory mem = VK_NULL_HANDLE;
  if (vkAllocateMemory(vkdev, &mai, NULL, &mem) != VK_SUCCESS) {
    LOG_WARN("gpu_buffer_create: vkAllocateMemory failed");
    vkDestroyBuffer(vkdev, vkbuf, NULL);
    return NULL;
  }
  vkBindBufferMemory(vkdev, vkbuf, mem, 0);

  gpu_buffer *b = calloc(1, sizeof(*b));
  if (!b) { vkFreeMemory(vkdev, mem, NULL); vkDestroyBuffer(vkdev, vkbuf, NULL); return NULL; }
  b->dev          = dev;
  b->buf          = vkbuf;
  b->mem          = mem;
  b->size         = size;
  b->host_visible = host_visible;
  return b;
}

void gpu_buffer_destroy(gpu_buffer *buf) {
  if (!buf) return;
  VkDevice vkdev = vk_device(buf->dev->vk);
  if (buf->mapped) vkUnmapMemory(vkdev, buf->mem);
  vkDestroyBuffer(vkdev, buf->buf, NULL);
  vkFreeMemory(vkdev, buf->mem, NULL);
  free(buf);
}

void *gpu_buffer_map(gpu_buffer *buf) {
  assert(buf && buf->host_visible);
  if (buf->mapped) return buf->mapped;
  VkDevice vkdev = vk_device(buf->dev->vk);
  if (vkMapMemory(vkdev, buf->mem, 0, buf->size, 0, &buf->mapped) != VK_SUCCESS) {
    LOG_WARN("gpu_buffer_map: vkMapMemory failed");
    return NULL;
  }
  return buf->mapped;
}

void gpu_buffer_unmap(gpu_buffer *buf) {
  assert(buf);
  if (!buf->mapped) return;
  vkUnmapMemory(vk_device(buf->dev->vk), buf->mem);
  buf->mapped = NULL;
}

void gpu_buffer_upload(gpu_buffer *buf, const void *data, size_t size) {
  assert(buf && data && size <= buf->size);
  if (buf->host_visible) {
    void *ptr = gpu_buffer_map(buf);
    if (ptr) { memcpy(ptr, data, size); gpu_buffer_unmap(buf); }
    return;
  }
  // Device-local: use a staging buffer
  gpu_buffer *staging = gpu_buffer_create(buf->dev, size, true);
  if (!staging) { LOG_WARN("gpu_buffer_upload: staging alloc failed"); return; }
  void *ptr = gpu_buffer_map(staging);
  if (!ptr) { gpu_buffer_destroy(staging); return; }
  memcpy(ptr, data, size);
  gpu_buffer_unmap(staging);

  VkDevice vkdev = vk_device(buf->dev->vk);
  VkCommandPool pool = make_cmd_pool(vkdev, vk_compute_queue_family(buf->dev->vk));
  VkCommandBuffer cb = begin_one_shot(vkdev, pool);
  VkBufferCopy region = { .size = size };
  vkCmdCopyBuffer(cb, staging->buf, buf->buf, 1, &region);
  end_one_shot(vkdev, pool, vk_compute_queue(buf->dev->vk), cb);
  vkDestroyCommandPool(vkdev, pool, NULL);
  gpu_buffer_destroy(staging);
}

void gpu_buffer_download(gpu_buffer *buf, void *data, size_t size) {
  assert(buf && data && size <= buf->size);
  if (buf->host_visible) {
    void *ptr = gpu_buffer_map(buf);
    if (ptr) { memcpy(data, ptr, size); gpu_buffer_unmap(buf); }
    return;
  }
  gpu_buffer *staging = gpu_buffer_create(buf->dev, size, true);
  if (!staging) { LOG_WARN("gpu_buffer_download: staging alloc failed"); return; }

  VkDevice vkdev = vk_device(buf->dev->vk);
  VkCommandPool pool = make_cmd_pool(vkdev, vk_compute_queue_family(buf->dev->vk));
  VkCommandBuffer cb = begin_one_shot(vkdev, pool);
  VkBufferCopy region = { .size = size };
  vkCmdCopyBuffer(cb, buf->buf, staging->buf, 1, &region);
  end_one_shot(vkdev, pool, vk_compute_queue(buf->dev->vk), cb);
  vkDestroyCommandPool(vkdev, pool, NULL);

  void *ptr = gpu_buffer_map(staging);
  if (ptr) { memcpy(data, ptr, size); gpu_buffer_unmap(staging); }
  gpu_buffer_destroy(staging);
}

// ---------------------------------------------------------------------------
// gpu_pipeline
// ---------------------------------------------------------------------------

gpu_pipeline *gpu_pipeline_create(gpu_device *dev, const uint8_t *spirv, size_t spirv_size) {
  assert(dev && spirv && spirv_size > 0);
  VkDevice vkdev = vk_device(dev->vk);

  VkShaderModuleCreateInfo smci = {
    .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spirv_size,
    .pCode    = (const uint32_t *)spirv,
  };
  VkShaderModule shader = VK_NULL_HANDLE;
  if (vkCreateShaderModule(vkdev, &smci, NULL, &shader) != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreateShaderModule failed");
    return NULL;
  }

  // Descriptor set layout: N storage buffers bound at binding 0..N-1.
  // We use a variable-count approach: create one binding per buffer at
  // creation time. For simplicity cap at 16 bindings.
  VkDescriptorSetLayoutBinding bindings[16];
  for (int i = 0; i < 16; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
      .binding         = (uint32_t)i,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
    };
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
    .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 16,
    .pBindings    = bindings,
  };
  VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
  if (vkCreateDescriptorSetLayout(vkdev, &dslci, NULL, &ds_layout) != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreateDescriptorSetLayout failed");
    vkDestroyShaderModule(vkdev, shader, NULL);
    return NULL;
  }

  VkPipelineLayoutCreateInfo plci = {
    .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts    = &ds_layout,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(vkdev, &plci, NULL, &layout) != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreatePipelineLayout failed");
    vkDestroyDescriptorSetLayout(vkdev, ds_layout, NULL);
    vkDestroyShaderModule(vkdev, shader, NULL);
    return NULL;
  }

  // Also create an empty layout (0 set layouts) for dispatches with no buffers.
  VkPipelineLayoutCreateInfo plci_empty = {
    .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 0,
    .pSetLayouts    = NULL,
  };
  VkPipelineLayout layout_empty = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(vkdev, &plci_empty, NULL, &layout_empty) != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreatePipelineLayout (empty) failed");
    vkDestroyPipelineLayout(vkdev, layout, NULL);
    vkDestroyDescriptorSetLayout(vkdev, ds_layout, NULL);
    vkDestroyShaderModule(vkdev, shader, NULL);
    return NULL;
  }

  // Create two pipelines: one with the full buffer layout, one with the empty
  // layout (0 set layouts) for dispatches with no buffers. This ensures
  // vkCmdBindDescriptorSets is never called with uninitialized descriptor sets
  // on drivers (e.g. Adreno) that crash on such operations.
  VkComputePipelineCreateInfo cpcis[2] = {
    {
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName  = "main",
      },
      .layout = layout,
    },
    {
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName  = "main",
      },
      .layout = layout_empty,
    },
  };
  VkPipeline pipelines[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
  VkResult r = vkCreateComputePipelines(vkdev, VK_NULL_HANDLE, 2, cpcis, NULL, pipelines);
  vkDestroyShaderModule(vkdev, shader, NULL);

  if (r != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreateComputePipelines failed (%d)", r);
    vkDestroyPipeline(vkdev, pipelines[0], NULL);
    vkDestroyPipeline(vkdev, pipelines[1], NULL);
    vkDestroyPipelineLayout(vkdev, layout_empty, NULL);
    vkDestroyPipelineLayout(vkdev, layout, NULL);
    vkDestroyDescriptorSetLayout(vkdev, ds_layout, NULL);
    return NULL;
  }

  // Descriptor pool: one set with up to 16 storage buffer descriptors.
  VkDescriptorPoolSize pool_size = {
    .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 16,
  };
  VkDescriptorPoolCreateInfo dpci = {
    .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets       = 1,
    .poolSizeCount = 1,
    .pPoolSizes    = &pool_size,
  };
  VkDescriptorPool ds_pool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(vkdev, &dpci, NULL, &ds_pool) != VK_SUCCESS) {
    LOG_WARN("gpu_pipeline_create: vkCreateDescriptorPool failed");
    vkDestroyPipeline(vkdev, pipelines[1], NULL);
    vkDestroyPipeline(vkdev, pipelines[0], NULL);
    vkDestroyPipelineLayout(vkdev, layout_empty, NULL);
    vkDestroyPipelineLayout(vkdev, layout, NULL);
    vkDestroyDescriptorSetLayout(vkdev, ds_layout, NULL);
    return NULL;
  }

  gpu_pipeline *p = calloc(1, sizeof(*p));
  if (!p) {
    vkDestroyDescriptorPool(vkdev, ds_pool, NULL);
    vkDestroyPipeline(vkdev, pipelines[1], NULL);
    vkDestroyPipeline(vkdev, pipelines[0], NULL);
    vkDestroyPipelineLayout(vkdev, layout_empty, NULL);
    vkDestroyPipelineLayout(vkdev, layout, NULL);
    vkDestroyDescriptorSetLayout(vkdev, ds_layout, NULL);
    return NULL;
  }
  p->dev            = dev;
  p->pipeline       = pipelines[0];
  p->pipeline_empty = pipelines[1];
  p->layout         = layout;
  p->layout_empty   = layout_empty;
  p->ds_layout      = ds_layout;
  p->ds_pool        = ds_pool;
  return p;
}

void gpu_pipeline_destroy(gpu_pipeline *p) {
  if (!p) return;
  VkDevice vkdev = vk_device(p->dev->vk);
  vkDestroyDescriptorPool(vkdev, p->ds_pool, NULL);
  vkDestroyPipeline(vkdev, p->pipeline_empty, NULL);
  vkDestroyPipeline(vkdev, p->pipeline, NULL);
  vkDestroyPipelineLayout(vkdev, p->layout_empty, NULL);
  vkDestroyPipelineLayout(vkdev, p->layout, NULL);
  vkDestroyDescriptorSetLayout(vkdev, p->ds_layout, NULL);
  free(p);
}

// ---------------------------------------------------------------------------
// gpu_dispatch / gpu_wait
// ---------------------------------------------------------------------------

void gpu_dispatch(gpu_device *dev, gpu_pipeline *p,
                  gpu_buffer **buffers, int num_buffers,
                  int groups_x, int groups_y, int groups_z) {
  assert(dev && p && num_buffers >= 0 && groups_x > 0 && groups_y > 0 && groups_z > 0);
  VkDevice vkdev = vk_device(dev->vk);

  int nb = num_buffers < 16 ? num_buffers : 16;

  VkCommandPool pool = make_cmd_pool(vkdev, vk_compute_queue_family(dev->vk));
  VkCommandBuffer cb = begin_one_shot(vkdev, pool);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);

  if (nb > 0) {
    // Allocate and populate a descriptor set for the storage buffers.
    VkDescriptorSetAllocateInfo dsai = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = p->ds_pool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &p->ds_layout,
    };
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkResetDescriptorPool(vkdev, p->ds_pool, 0);
    vkAllocateDescriptorSets(vkdev, &dsai, &ds);

    VkDescriptorBufferInfo buf_infos[16];
    VkWriteDescriptorSet   writes[16];
    for (int i = 0; i < nb; i++) {
      buf_infos[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i]->buf, .offset = 0,
                                               .range  = buffers[i]->size };
      writes[i] = (VkWriteDescriptorSet){
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = ds,
        .dstBinding      = (uint32_t)i,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo     = &buf_infos[i],
      };
    }
    vkUpdateDescriptorSets(vkdev, (uint32_t)nb, writes, 0, NULL);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout,
                            0, 1, &ds, 0, NULL);
  }
  // When nb == 0: use layout_empty (0 set layouts) — no descriptor sets to bind.

  vkCmdDispatch(cb, (uint32_t)groups_x, (uint32_t)groups_y, (uint32_t)groups_z);
  end_one_shot(vkdev, pool, vk_compute_queue(dev->vk), cb);
  vkDestroyCommandPool(vkdev, pool, NULL);
}

void gpu_wait(gpu_device *dev) {
  assert(dev);
  vkQueueWaitIdle(vk_compute_queue(dev->vk));
}
