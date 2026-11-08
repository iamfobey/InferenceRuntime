#pragma once

#if RUNTIME_ENABLE_CUDA
#include "Core/IBuffer.hpp"
#include "Utils/Utils.hpp"

class CudaBuffer final : public IBuffer
{
public:
    explicit CudaBuffer(std::size_t sizeBytes) :
        m_SizeBytes(sizeBytes)
    {
        if (m_SizeBytes != 0)
            CUDA_CHECK(cudaMalloc(&m_pData, m_SizeBytes));
    }

    ~CudaBuffer() override
    {
        if (m_pData != nullptr)
            cudaFree(m_pData);
    }

    [[nodiscard]]
    DeviceType Device() const noexcept override
    {
        return DeviceType::CUDA;
    }

    [[nodiscard]]
    std::size_t SizeBytes() const noexcept override
    {
        return m_SizeBytes;
    }

    void* Data() noexcept override
    {
        return m_pData;
    }

    [[nodiscard]] const void* Data() const noexcept override
    {
        return m_pData;
    }

private:
    void* m_pData{};
    std::size_t m_SizeBytes{};
};
#endif
