//
// Created by berry on 2023/3/12.
//

#ifndef VKLEARN_RENDER_THREAD_HPP
#define VKLEARN_RENDER_THREAD_HPP

#include "shader_modules.hpp"
#include "graphics_pipeline.hpp"
#include "bindings_management.hpp"
#include "resource_management.hpp"
#include "renderpass.hpp"
void render_work_thread(
        size_t inflightIndex,
        vk::Instance inst, vk::PhysicalDevice physDev, PhysicalDeviceInfo devProps, vk::Device renderDev,
        uint32_t queueFamilyIndex,
        VulkanResourceManager &&resMgrHdl,
        vk::Extent2D renderExtent, vk::Format renderSwapchainFmt, vk::ImageUsageFlags renderSwapchainUsg, vk::Format renderDepthFmt,
        vk::Semaphore imageAvailableSemaphore, vk::Semaphore renderCompleteSemaphore) {
    // Queue 0 is used by the main thread.
    auto queueIndex = inflightIndex+1;
    auto renderQueue = utils::QueueStruct{.queue = renderDev.getQueue(queueFamilyIndex,
                                                                      queueIndex), .queueFamilyIdx = queueFamilyIndex};
    auto resMgr = VulkanResourceManager{std::forward<VulkanResourceManager>(resMgrHdl)};resMgr.setupManagerHandle(renderQueue);
    auto viewport = get_viewport_info(renderExtent.width, renderExtent.height);
    /*The order of preparing stuffs:
     * Descriptor
     * Renderpass and Image-less FrameBuffer
     * Attachment requested resources
     * Descriptor requested resources
     * */
    auto startTime = std::chrono::high_resolution_clock::now();
    auto descMgr = DescriptorManager{renderDev};
    auto cmdMgr = CommandBufferManager{renderDev, renderQueue};

    /* Build a renderpass:
     *   For each subpass inside a renderpass:
     *     Get its pipeline.
     *     Scan its descriptors and
     *     Register its descriptor layouts.
     *     Get its attachment requirements:
     *       Input attachments can be found at fragment shader's spirv_cross::ShaderResources.subpass_inputs,
     *        with its type being Image
     *       Color attachments can be found at fragment shader's spirv_cross::ShaderResources.stage_outputs
     *       Resolve attachments can be found at fragment shader's spirv_cross::ShaderResources.subpass_inputs,
     *        with its type being SampledImage and SPIRType::image::ms being true
     *       Depth attachment is determined by vk::PipelineDepthStencilStateCreateInfo of the pipeline
     *     Create its pipeline layout object.
     *     Register subpass' attachments to the renderpass.
     *
     *   Then for each subpass:
     *     Register subpass' SubpassInfo object.
     *   Now we can create the renderpass object.
     *   Create an image-less frameBuffer with all registered attachments included.
     * */

    // First, get the pipeline of a subpass
    auto graphPipe = GraphicsPipeline<VertexShader, FragShader>{renderDev, viewport};
    // Loop over each descriptor set
    for (auto &setBind: graphPipe.bindings_) {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBinds{};
    // Loop over each descriptor in one set
        for (auto &bind: setBind.second) {
            layoutBinds.push_back(bind);
        }
    // We don't need to unregister later, don't record index for now.
        descMgr.registerDescriptorBindings(layoutBinds);
    }
    auto graphPipeLayout = graphPipe.createPipelineLayout();

    auto renderpassMgr = Renderpass{renderDev};

    for (auto &attach: graphPipe.fragShader_.attachmentResourceInfos_) {
        // TODO: Add proper mechanism for filling those info at runtime
        if (attach.second.resId == "swapchainIMG") {
            attach.second.format = renderSwapchainFmt;
            attach.second.description.format = renderSwapchainFmt;
            attach.second.usage = renderSwapchainUsg;
        }
        if (attach.second.resId == "depthIMG") {
            attach.second.format = renderDepthFmt;
            attach.second.description.format = renderDepthFmt;
        }
        renderpassMgr.registerSubpassAttachment(attach.second, attach.first);
    }

    renderpassMgr.registerSubpassInfo(graphPipe.subpassInfo_);
    renderpassMgr.createRenderpass();
    // Must create renderpass object before creating framebuffer
    auto viewlessFrameBuffer = renderpassMgr.createViewlessFramebuffer(renderExtent);

    graphPipe.createPipeline(renderpassMgr.renderpass_);

    // here we just hard-code everything
    /*TODO: Create attachment requested resources
     * Loop over attachments:
     *   Inspect info to determine whether the vk::ImageView object needs to be created by us
     *   If we need to create vk::ImageView, check whether we need to create vk::Image as well ()
     * */

    auto depthImgRes = resMgr.createImagenMemory(
            renderExtent, renderDepthFmt, vk::ImageTiling::eOptimal, vk::ImageLayout::eUndefined,
            vk::ImageUsageFlagBits::eDepthStencilAttachment);
    depthImgRes.createView(vk::ImageAspectFlagBits::eDepth);

    /* Hard-code everything, also duplicate global static resources
     * TODO: Create descriptor requested resources
     *  Loop over descriptor sets
     *   Loop over each descriptor
     * */
    // Right now there are three resources:
    //  A texture: Load data from disk, create memory, image, transfer data to device, create image view, sampler
    //  A uniform buffer for sending in model matrix: Create memory, buffer
    //  An accelerate structure: Get handles of scene geometry, create memory interactively for BLAS, build BLAS,
    //   create memory interactively for TLAS, build TLAS. A testament of the flexibility of our resource management scheme

    // First create descriptor pool, as all required descriptor sets are registered now

    descMgr.createDescriptorPool(2);

    // Allocate resources, TODO: Add AccelStruct
    auto testTexture = TextureObject{"textures/test_512.png", resMgr};

    auto modelUBO = resMgr.createBuffernMemory(
            sizeof(ModelUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped);
    // Create descriptor sets for each subpass's pipeline inside a renderpass

    auto descriptorSet = descMgr.createDescriptorSet(graphPipe.getDescriptorLayout(0));
    auto testTextureBind = graphPipe.queryDescriptorSetLayoutBinding(0, "texSampler");
    descMgr.updateDescriptorSet(descriptorSet.get(), testTexture.sampler_.get(), testTexture.view_,
                                testTexture.imageLayout_, testTextureBind, 0, 1);
    auto modelUBOBind = graphPipe.queryDescriptorSetLayoutBinding(0, "modelUBO");
    descMgr.updateDescriptorSet(descriptorSet.get(), modelUBO.resource.get(), 0, modelUBO.resInfo.size, modelUBOBind, 0,
                                1);

    // Lastly, create other resources required by the renderpass
    auto modelVertInputHost = model_info::gVertices;
    auto modelVertInputDevice = resMgr.createBuffernMemoryFromHostData<model_info::PCTVertex>(modelVertInputHost,
                                                                                              vk::BufferUsageFlagBits::eVertexBuffer);
    auto modelVertIdxHost = model_info::gVertexIdx;
    auto modelVertIdxDevice = resMgr.createBuffernMemoryFromHostData<model_info::VertIdxType>(modelVertIdxHost,
                                                                                              vk::BufferUsageFlagBits::eIndexBuffer);
    auto sceneVP = ScenePushConstants{};

    // Now should do the loop

    bool exitSignal = false;
    auto cmdBufs = cmdMgr.createCommandBuffers(1);// We only have 1 renderpass, should be enough
    vk::FenceCreateInfo cmdPoolRstFenceInfo{};
    cmdPoolRstFenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
    auto [resultFence, commandPoolResetFence] = renderDev.createFenceUnique(cmdPoolRstFenceInfo);
    utils::vk_ensure(resultFence);
    while (!exitSignal) {
        while (!mainRendererComms[inflightIndex].mainLoopReady.load()){
            utils::log_and_pause(std::format("Renderer {}: waiting for the main thread...", inflightIndex), 10);
        }
        // Begin of pacing-insensitive tasks
        auto &commandBuffer = cmdBufs[0];
        vk::CommandBufferBeginInfo beginInfo = {};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        // Wait for reset
        bool resumeRender = false;
        do {
            auto resultResetPool = renderDev.waitForFences(commandPoolResetFence.get(), VK_TRUE, 1'000'000);
            switch (resultResetPool) {
                case vk::Result::eSuccess:
                    resumeRender = true;
                    break;
                case vk::Result::eTimeout:
                    if (mainRendererComms[inflightIndex].swapchainInvalid.load()) {
                        exitSignal = true;
                    }
                    break;
                default:
                    utils::vk_ensure(resultResetPool);
            }
        } while (!exitSignal & !resumeRender);
        if (exitSignal) {
            break;
        }
        cmdMgr.resetPool();
        utils::vk_ensure(renderDev.resetFences(commandPoolResetFence.get()));
        auto beginResult = commandBuffer.begin(beginInfo);
        utils::vk_ensure(beginResult);

        vk::RenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.renderPass = renderpassMgr.renderpass_;
        renderPassInfo.framebuffer = viewlessFrameBuffer.get();
        renderPassInfo.renderArea = vk::Offset2D{0, 0};
        renderPassInfo.renderArea.extent = renderExtent;

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color = {std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}};
        clearValues[1].depthStencil.depth = 0.0f;
        clearValues[1].depthStencil.stencil = 0;
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();

        // Prepare submit info in advance
        vk::SubmitInfo2 submitInfo2 = {};
        submitInfo2.flags = vk::SubmitFlags{};
        std::vector<vk::SemaphoreSubmitInfo> waitSemaphores = {};
        // The pipeline will wait imageAvailableSemaphore at the ColorAttachmentOutput stage.
        auto imageAvailSemaInfo = vk::SemaphoreSubmitInfo{
                imageAvailableSemaphore,
                0,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                0
        };
        waitSemaphores.push_back(imageAvailSemaInfo);
        std::vector<vk::SemaphoreSubmitInfo> signalSemaphores = {};
        // When all commands in this submission are completed, renderCompleteSemaphore will be signaled.
        auto renderCompleteSemaInfo = vk::SemaphoreSubmitInfo{
                renderCompleteSemaphore,
                0,
                vk::PipelineStageFlagBits2::eAllCommands,
                0
        };
        signalSemaphores.push_back(renderCompleteSemaInfo);

        std::vector<vk::ImageView> attaches{}; attaches.resize(2);
        attaches[1] = depthImgRes.view.get();
        vk::RenderPassAttachmentBeginInfo attachInfo{};
        attachInfo.setAttachments(attaches);
        vk::StructureChain<vk::RenderPassBeginInfo, vk::RenderPassAttachmentBeginInfo> renderPassInfoCombined{
            renderPassInfo, attachInfo
        };
        vk::CommandBufferSubmitInfo cmdBufInfo{};
        cmdBufInfo.commandBuffer = commandBuffer;
        // Assemble VkQueueSubmitInfo2.
        submitInfo2.waitSemaphoreInfoCount = waitSemaphores.size();
        submitInfo2.pWaitSemaphoreInfos = waitSemaphores.data();
        submitInfo2.commandBufferInfoCount = 1;
        submitInfo2.pCommandBufferInfos = &cmdBufInfo;
        submitInfo2.signalSemaphoreInfoCount = signalSemaphores.size();
        submitInfo2.pSignalSemaphoreInfos = signalSemaphores.data();
        // This is the furthest we can go in terms of submission before waiting for the handle,
        //  but we can first wait for frame-pacing related fences.

        // TODO: Add support for vkWaitForPresentKHR
        // (Host access to swapchain must be externally synchronized)
        if (mainRendererComms[inflightIndex].swapchainInvalid.load()) {
            exitSignal = true;
        }
        if (exitSignal) {
            break;
        }
        // Do pacing-sensitive work
        auto currentTime = std::chrono::high_resolution_clock::now();
        float runTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        {
            ModelUBO ubo{};
            ubo.model = glm::rotate(glm::mat4(1.0f), runTime * glm::radians(37.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            std::memcpy(modelUBO.auxInfo.pMappedData, &ubo, sizeof(ubo));
            sceneVP.view = utils::vkuLookAtRH(glm::vec3(0.65f, 0.65f, 0.65f), glm::vec3(0.0f, 0.0f, 0.0f),
                                              glm::vec3(0.0f, 0.0f, 1.0f));
            sceneVP.proj = utils::vkuPerspectiveRHReverse_ZO(glm::radians(60.0f),
                                                             renderExtent.width / (float) renderExtent.height, 0.1f,
                                                             10.0f);
        }
        // Now we must wait main thread for imageview handle
        while (!mainRendererComms[inflightIndex].imageViewReadyToRender.try_acquire_for(
                std::chrono::milliseconds(100))) {
            if (mainRendererComms[inflightIndex].swapchainInvalid.load()) {
                exitSignal = true;
                break;
            } else {
                utils::log_and_pause("Main thread is lagging!", 0);
            }
        }
        if (exitSignal) {
            break;
        }
        auto swapchainImageViewHandle = mainRendererComms[inflightIndex].imageViewHandle.load();
        attaches[0] = swapchainImageViewHandle;

        commandBuffer.beginRenderPass(renderPassInfoCombined.get<vk::RenderPassBeginInfo>(), vk::SubpassContents::eInline);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphPipe.pipeline_);

        // Omit some code since here we only have one vertex buffer
        auto vertBufOffset = vk::DeviceSize{0};
        commandBuffer.bindVertexBuffers(
                0, //offset
                1, //count
                &modelVertInputDevice.resource.get(),
                &vertBufOffset
        );
        commandBuffer.bindIndexBuffer(
                modelVertIdxDevice.resource.get(),
                0, //offset
                vk::IndexTypeValue<decltype(modelVertIdxHost)::value_type>::value //vertex index buffer elements' type
        );
        commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, graphPipeLayout,
                0, //offset of descriptor set
                1, //count
                &descriptorSet.get(),
                0, nullptr // dynamic offset related info
        );

        commandBuffer.pushConstants(
                graphPipeLayout,
                vk::ShaderStageFlagBits::eVertex,
                0,//byte offset
                sizeof(ScenePushConstants), &sceneVP);

        commandBuffer.drawIndexed(modelVertIdxHost.size(), 1, 0, 0, 0);

        commandBuffer.endRenderPass();

        auto endResult = commandBuffer.end();
        utils::vk_ensure(endResult);
        auto resultSubmit = renderQueue.queue.submit2(submitInfo2, commandPoolResetFence.get());
        utils::vk_ensure(resultSubmit);
        // Notify the main thread that submission process is complete
        mainRendererComms[inflightIndex].imageViewRendered.release();
        // After submit
    }
    // Workaround for the lack of VK_EXT_swapchain_maintenance1 support,
    //   use semaphore to avoid validation layer emitting `337425955: UNASSIGNED-Threading-MultipleThreads`,
    //   or just mind our own business instead
    utils::vk_ensure(renderQueue.queue.waitIdle());
}

#endif //VKLEARN_RENDER_THREAD_HPP
