#pragma once

#include "Core/Tensor.hpp"

#include <span>
#include <vector>

class IBackend
{
public:
    virtual ~IBackend() = default;

    [[nodiscard]]
    virtual DeviceType Device() const noexcept = 0;

    [[nodiscard]]
    virtual Tensor CreateTensor(std::vector<std::size_t> shape, DataType dataType) = 0;

    virtual void Upload(Tensor& destination, std::span<const float> source) = 0;

    virtual void Download(const Tensor& source, std::span<float> destination) = 0;

    virtual void Embedding(const Tensor& embeddingTable, std::span<const std::int32_t> tokenIds, Tensor& output) = 0;

    virtual void Linear(const Tensor& W, const Tensor& x, Tensor& y) = 0;

    virtual void RMSNorm(const Tensor& x, const Tensor& weight, float epsilon, Tensor& y) = 0;

    virtual void Add(const Tensor& a, const Tensor& b, Tensor& output) = 0;

    virtual void Multiply(const Tensor& a, const Tensor& b, Tensor& output) = 0;

    virtual void SiLU(const Tensor& x, Tensor& output) = 0;

    virtual void PreCalc(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                         float theta) = 0;

    virtual void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
                      size_t position, std::size_t headDimension) = 0;

    virtual void Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache, Tensor& scores,
                           std::size_t validTokenCount, std::size_t attentionHeadCount, std::size_t keyValueHeadCount, Tensor& output) = 0;

    virtual void CopyToCache(const Tensor& source, Tensor& cache, std::size_t position) = 0;

    virtual void Synchronize() = 0;
};
