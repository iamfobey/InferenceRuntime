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
    DeviceType GetDeviceType() const noexcept override
    {
        return DeviceType::CPU;
    }

    [[nodiscard]]
    std::size_t GetSize() const noexcept override
    {
        return m_Storage.size();
    }

    void* GetData() noexcept override
    {
        return m_Storage.data();
    }

    [[nodiscard]]
    const void* GetData() const noexcept override
    {
        return m_Storage.data();
    }

private:
    std::vector<std::byte> m_Storage;
};
