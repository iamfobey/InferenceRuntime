#pragma once

#include "Core/IBuffer.hpp"

#include <vector>

class CpuBuffer final : public IBuffer
{
public:
    explicit CpuBuffer(const std::size_t sizeBytes)
        : m_Storage(sizeBytes)
    {
    }

    DeviceType Device() const noexcept override
    {
        return DeviceType::CPU;
    }

    std::size_t SizeBytes() const noexcept override
    {
        return m_Storage.size();
    }

    void* Data() noexcept override
    {
        return m_Storage.data();
    }

    const void* Data() const noexcept override
    {
        return m_Storage.data();
    }

private:
    std::vector<std::byte> m_Storage;
};
