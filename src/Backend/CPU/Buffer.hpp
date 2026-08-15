#pragma once

#include "Core/IBuffer.hpp"

#include <vector>

class CpuBuffer final : public IBuffer
{
public:
    explicit CpuBuffer(std::size_t sizeBytes)
        : m_Storage(sizeBytes)
    {
    }

    [[nodiscard]]
    DeviceType Device() const noexcept override
    {
        return DeviceType::CPU;
    }

    [[nodiscard]]
    std::size_t SizeBytes() const noexcept override
    {
        return m_Storage.size();
    }

    void* Data() noexcept override
    {
        return m_Storage.data();
    }

    [[nodiscard]]
    const void* Data() const noexcept override
    {
        return m_Storage.data();
    }

private:
    std::vector<std::byte> m_Storage;
};
