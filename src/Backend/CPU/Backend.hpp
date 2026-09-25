#pragma once

#include "Backend/IBackend.hpp"

#include <span>

#include "Core/ThreadPool.hpp"

struct CpuBackendOptions
{
    int threadCount{1};
};

class CpuBackend final : public IBackend
{
public:
    explicit CpuBackend(CpuBackendOptions options);

    [[nodiscard]]
    DeviceType Device() const noexcept override;

    [[nodiscard]]
    Tensor CreateTensor(TensorDimVec shape, DataType dataType) override;

    void Upload(Tensor& destination, std::span<const float> source) override;

    void Download(const Tensor& source, std::span<float> destination) override;

    void Embedding(const Tensor& embeddingTable, std::span<const std::int32_t> tokenIds, Tensor& output) override;

    void Linear(const Tensor& weights, const Tensor& input, Tensor& output) override;

    void RMSNorm(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output) override;

    void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output) override;

    void Multiply(const Tensor& inputA, const Tensor& inputB, Tensor& output) override;

    void SiLU(const Tensor& input, Tensor& output) override;

    void ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                    float theta) override;

    void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
              size_t position, std::size_t headDimension) override;

    void Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                   size_t validTokenCount, size_t attentionHeadCount, size_t keyValueHeadCount,
                   Tensor& output) override;

    void CopyToCache(const Tensor& source, Tensor& cache, size_t position) override;

    void Synchronize() override;

private:
    CpuBackendOptions m_Options;
    ThreadPool m_ThreadPool;
};
