#pragma once

#include "Backend/IBackend.hpp"

#include <span>
#include <vector>

struct CpuBackendOptions
{
    std::uint8_t threadCount{1};
};

class CpuBackend final : public IBackend
{
public:
    explicit CpuBackend(CpuBackendOptions options);

    [[nodiscard]]
    DeviceType Device() const noexcept override;

    [[nodiscard]]
    Tensor CreateTensor(std::vector<std::size_t> shape, DataType dataType) override;

    void Upload(Tensor& destination, std::span<const float> source) override;

    void Download(const Tensor& source, std::span<float> destination) override;

    void Embedding(const Tensor& embeddingTable, std::span<const std::int32_t> tokenIds, Tensor& output) override;

    void Linear(const Tensor& W, const Tensor& x, const Tensor* b, Tensor& y) override;

    void RMSNorm(const Tensor& x, const Tensor& weight, float epsilon, Tensor& y) override;

    void Add(const Tensor& a, const Tensor& b, Tensor& output) override;

    void Multiply(const Tensor& a, const Tensor& b, Tensor& output) override;

    void SiLU(const Tensor& x, Tensor& output) override;

    void PreCalc(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                 float theta) override;

    void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
              std::size_t headDimension) override;

    void Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache, std::size_t validTokenCount,
                   std::size_t attentionHeadCount, std::size_t keyValueHeadCount, Tensor& output) override;

    void CopyToCache(const Tensor& source, Tensor& cache, std::size_t position) override;

    void Synchronize() override;

private:
    CpuBackendOptions m_Options;
};
