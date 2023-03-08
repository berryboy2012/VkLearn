//
// Created by berry on 2023/3/5.
//

#ifndef VKLEARN_BINDINGS_MANAGEMENT_HPP
#define VKLEARN_BINDINGS_MANAGEMENT_HPP
class DescriptorManager{
public:
    vk::DescriptorPool descriptorPool_{};
    typedef size_t LayoutIdx;
    DescriptorManager() = default;
    DescriptorManager(const DescriptorManager &) = delete;
    DescriptorManager& operator= (const DescriptorManager &) = delete;
    explicit DescriptorManager(vk::Device device){
        device_ = device;
    }
    DescriptorManager& operator= (DescriptorManager &&other) noexcept{
        if (this != &other){
            device_ = other.device_;
            layoutSizes_ = other.layoutSizes_;
            numDescLayouts_ = other.numDescLayouts_;
            pool_ = std::move(other.pool_);
            other.descriptorPool_ = VK_NULL_HANDLE;
            descriptorPool_ = pool_.get();
        }
        return *this;
    }
    DescriptorManager(DescriptorManager &&other) noexcept{
        *this = std::move(other);
    }
    // Vulkan does not provide a way to calculate the descriptor pool usage of a descriptor set layout, we have to go
    //  one step lower.
    // Each call should correspond to a single descriptor set layout.
    // The returned LayoutIdx object is used to unregister the bindings later.
    LayoutIdx registerDescriptorBindings(const std::span<const vk::DescriptorSetLayoutBinding> bindings){
        std::vector<vk::DescriptorPoolSize> layoutSize{};
        for (auto & bind: bindings){
            vk::DescriptorPoolSize bindSize = {};
            bindSize.type = bind.descriptorType;
            bindSize.descriptorCount = bind.descriptorCount;
            layoutSize.push_back(bindSize);
        }
        LayoutIdx index = layoutSizes_.empty() ? 0 : layoutSizes_.rbegin()->first+1;
        layoutSizes_[index] = layoutSize;
        numDescLayouts_ += 1;
        return index;
    }
    // Supply the object returned by registerDescriptorBindings() here.
    void unregisterDescriptorBindings(const LayoutIdx layoutIndex){
        if (layoutSizes_.contains(layoutIndex)){
            layoutSizes_.erase(layoutIndex);
            numDescLayouts_ -= 1;
        }
    }
    // descriptorSetDuplicates controls the multiplier of vk::DescriptorPoolCreateInfo::maxSets
    void createDescriptorPool(const size_t descriptorSetDuplicates){
        std::vector<vk::DescriptorPoolSize> allPoolSizes{};
        for (auto const& sizes: layoutSizes_){
            allPoolSizes.insert(allPoolSizes.end(), sizes.second.begin(), sizes.second.end());
        }
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.setPoolSizes(allPoolSizes);
        poolInfo.maxSets = static_cast<uint32_t>(descriptorSetDuplicates*numDescLayouts_);
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;

        auto [result, descriptorPool] = device_.createDescriptorPoolUnique(poolInfo);
        utils::vkEnsure(result);
        pool_ = std::move(descriptorPool);
        descriptorPool_ = pool_.get();
    }
    // TODO: give user the choice to manage lifetimes of descriptor set by DescriptorManager.
    std::vector<vk::UniqueDescriptorSet> createDescriptorSets(const std::span<const vk::DescriptorSetLayout> layouts){
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = pool_.get();
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        auto [result, descriptorSets] = device_.allocateDescriptorSetsUnique(allocInfo);
        utils::vkEnsure(result);
        return std::move(descriptorSets);
    }

    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             vk::Sampler sampler, vk::ImageView view, vk::ImageLayout layout,
                             const vk::DescriptorSetLayoutBinding &bindInfo, uint32_t bindOffset, uint32_t numElements){
        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = layout;
        imageInfo.imageView = view;
        imageInfo.sampler = sampler;

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = bindInfo.binding;
        descriptorWrite.dstArrayElement = bindOffset;
        descriptorWrite.descriptorType = bindInfo.descriptorType;
        descriptorWrite.descriptorCount = numElements;
        descriptorWrite.pImageInfo = &imageInfo;

        updateDescriptorSet(descriptorWrite);
    }

    void updateDescriptorSet(vk::DescriptorSet descriptorSet,
                             vk::Buffer buffer, vk::DeviceSize bufferOffset, vk::DeviceSize bufferRange,
                             const vk::DescriptorSetLayoutBinding &bindInfo, uint32_t bindOffset, uint32_t numElements){
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer;
        bufferInfo.offset = bufferOffset;
        bufferInfo.range = bufferRange;

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = bindInfo.binding;
        descriptorWrite.dstArrayElement = bindOffset;
        descriptorWrite.descriptorType = bindInfo.descriptorType;
        descriptorWrite.descriptorCount = numElements;
        descriptorWrite.pBufferInfo = &bufferInfo;

        updateDescriptorSet(descriptorWrite);
    }

    void updateDescriptorSet(const vk::WriteDescriptorSet &descriptorWriteInfo){
        device_.updateDescriptorSets(descriptorWriteInfo,{});
    }
private:
    vk::Device device_{};
    std::map<LayoutIdx, std::vector<vk::DescriptorPoolSize>> layoutSizes_{};
    size_t numDescLayouts_{};
    vk::UniqueDescriptorPool pool_{};
};
#endif //VKLEARN_BINDINGS_MANAGEMENT_HPP
