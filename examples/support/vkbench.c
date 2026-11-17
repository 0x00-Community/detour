// Times one case on the local Vulkan (Mesa's ANV unless VK_DRIVER_FILES says otherwise): the SPIR-V shaders made a
// pipeline, then FRAMES frames of clear + one triangle into a W×H RGBA32F image, each waited for (a fence). Prints, in
// nanoseconds: `COMPILE FIRST MEAN MIN` (COMPILE: shader modules + pipeline; FIRST: the first frame; MEAN / MIN: the
// frames after it).
//
//   cc -O2 -o vkbench vkbench.c -lvulkan
//   vkbench VERT.spv FRAG.spv ATTRS W H FRAMES
//
// ATTRS: one vertex input a line, `LOCATION N V...` (N floats a vertex, three vertices).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define VERTICES 3
#define MAX_ATTRS 16
#define MAX_VALUES 256
#define NS 1000000000ull
#define FOREVER UINT64_MAX

#define CHECK(x)                                                    \
    do {                                                            \
        VkResult r_ = (x);                                          \
        if (r_ != VK_SUCCESS) {                                     \
            fprintf(stderr, "vkbench: %s: %d\n", #x, r_);           \
            exit(1);                                                \
        }                                                           \
    } while (0)

static void die(const char *what) {
    fprintf(stderr, "vkbench: %s\n", what);
    exit(1);
}

static uint64_t now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * NS + (uint64_t)t.tv_nsec;
}

static void *slurp(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) die(path);
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(*size);
    if (fread(buf, 1, *size, f) != *size) die(path);
    fclose(f);
    return buf;
}

struct attr {
    uint32_t location, size;
    float values[MAX_VALUES];
};

static int attrs(const char *path, struct attr *out) {
    FILE *f = fopen(path, "r");
    if (!f) die(path);
    int n = 0;
    while (n < MAX_ATTRS && fscanf(f, "%u %u", &out[n].location, &out[n].size) == 2) {
        if (out[n].size < 1 || out[n].size > 4) die("1 to 4 floats a vertex");
        for (uint32_t i = 0; i < out[n].size * VERTICES; i++)
            if (fscanf(f, "%f", &out[n].values[i]) != 1) die("an attribute value");
        n++;
    }
    fclose(f);
    return n;
}

static uint32_t memory_type(VkPhysicalDevice gpu, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(gpu, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & want) == want) return i;
    die("no memory type");
    return 0;
}

static VkShaderModule module(VkDevice dev, const char *path) {
    size_t size;
    void *code = slurp(path, &size);
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m;
    CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
    free(code);
    return m;
}

int main(int argc, char **argv) {
    if (argc != 7) die("usage: vkbench VERT.spv FRAG.spv ATTRS W H FRAMES");
    uint32_t w = (uint32_t)atoi(argv[4]), h = (uint32_t)atoi(argv[5]);
    int frames = atoi(argv[6]);
    if (frames < 2) die("at least two frames");
    struct attr *as = calloc(MAX_ATTRS, sizeof *as);
    int nattrs = attrs(argv[3], as);

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t ngpu = 1;
    VkPhysicalDevice gpu;
    VkResult r = vkEnumeratePhysicalDevices(inst, &ngpu, &gpu);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || ngpu == 0) die("no Vulkan device");
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, qp);
    uint32_t family = 0;
    while (family < nq && !(qp[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) family++;
    if (family == nq) die("no graphics queue");
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    VkDevice dev;
    CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);

    VkImageCreateInfo imci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage image;
    CHECK(vkCreateImage(dev, &imci, NULL, &image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, image, &req);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = memory_type(gpu, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory imem;
    CHECK(vkAllocateMemory(dev, &mai, NULL, &imem));
    CHECK(vkBindImageMemory(dev, image, imem, 0));
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = imci.format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view;
    CHECK(vkCreateImageView(dev, &vci, NULL, &view));

    VkAttachmentDescription att = {.format = imci.format, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
    VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &ref};
    VkRenderPassCreateInfo rpci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1, .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub};
    VkRenderPass pass;
    CHECK(vkCreateRenderPass(dev, &rpci, NULL, &pass));
    VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = 1, .pAttachments = &view, .width = w, .height = h, .layers = 1};
    VkFramebuffer fb;
    CHECK(vkCreateFramebuffer(dev, &fci, NULL, &fb));

    VkBuffer bufs[MAX_ATTRS];
    VkDeviceMemory bmem[MAX_ATTRS];
    VkVertexInputBindingDescription bind[MAX_ATTRS];
    VkVertexInputAttributeDescription in[MAX_ATTRS];
    const VkFormat formats[] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
    for (int i = 0; i < nattrs; i++) {
        VkDeviceSize bytes = as[i].size * VERTICES * sizeof(float);
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
        CHECK(vkCreateBuffer(dev, &bci, NULL, &bufs[i]));
        vkGetBufferMemoryRequirements(dev, bufs[i], &req);
        VkMemoryAllocateInfo bai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = memory_type(gpu, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        CHECK(vkAllocateMemory(dev, &bai, NULL, &bmem[i]));
        CHECK(vkBindBufferMemory(dev, bufs[i], bmem[i], 0));
        void *p;
        CHECK(vkMapMemory(dev, bmem[i], 0, bytes, 0, &p));
        memcpy(p, as[i].values, bytes);
        vkUnmapMemory(dev, bmem[i]);
        bind[i] = (VkVertexInputBindingDescription){(uint32_t)i, as[i].size * (uint32_t)sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
        in[i] = (VkVertexInputAttributeDescription){as[i].location, (uint32_t)i, formats[as[i].size - 1], 0};
    }

    uint64_t t0 = now();
    VkShaderModule vs = module(dev, argv[1]), fs = module(dev, argv[2]);
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
    };
    VkPipelineVertexInputStateCreateInfo vin = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = (uint32_t)nattrs, .pVertexBindingDescriptions = bind, .vertexAttributeDescriptionCount = (uint32_t)nattrs, .pVertexAttributeDescriptions = in};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &blend};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
    VkGraphicsPipelineCreateInfo gpci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages, .pVertexInputState = &vin, .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = layout, .renderPass = pass};
    VkPipeline pipe;
    CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
    uint64_t compile = now() - t0;

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
    VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    VkClearValue clear = {.color = {{0, 0, 0, 0}}};
    VkRenderPassBeginInfo rpbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = fb, .renderArea = sc, .clearValueCount = 1, .pClearValues = &clear};
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkDeviceSize offsets[MAX_ATTRS] = {0};
    if (nattrs) vkCmdBindVertexBuffers(cmd, 0, (uint32_t)nattrs, bufs, offsets);
    vkCmdDraw(cmd, VERTICES, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fnci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(dev, &fnci, NULL, &fence));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    uint64_t first = 0, sum = 0, min = UINT64_MAX;
    for (int i = 0; i < frames; i++) {
        uint64_t t = now();
        CHECK(vkQueueSubmit(queue, 1, &si, fence));
        CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, FOREVER));
        CHECK(vkResetFences(dev, 1, &fence));
        uint64_t d = now() - t;
        if (i == 0) {
            first = d;
            continue;
        }
        sum += d;
        if (d < min) min = d;
    }
    printf("%llu %llu %llu %llu\n", (unsigned long long)compile, (unsigned long long)first, (unsigned long long)(sum / (frames - 1)), (unsigned long long)min);
    vkDeviceWaitIdle(dev);
    return 0;
}
