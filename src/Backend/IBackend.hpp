#pragma once

#include "Core/Tensor.hpp"

#include <span>

class IBackend
{
public:
    virtual ~IBackend() = default;

    [[nodiscard]]
    virtual DeviceType Device() const noexcept = 0;

    [[nodiscard]]
    virtual Tensor CreateTensor(TensorDimVec, DataType dataType) = 0;

    virtual void Upload(Tensor& destination, std::span<const float> source) = 0;

    virtual void Download(const Tensor& source, std::span<float> destination) = 0;

    virtual void Embedding(const Tensor& embeddingTable, std::span<const std::int32_t> tokenIds, Tensor& output) = 0;

    virtual void Linear(const Tensor& weights, const Tensor& input, Tensor& output) = 0;

    virtual void RMSNorm(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output) = 0;

    virtual void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output) = 0;

    virtual void Multiply(const Tensor& inputA, const Tensor& inputB, Tensor& output) = 0;

    virtual void SiLU(const Tensor& input, Tensor& output) = 0;

    virtual void ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                                   float theta) = 0;

    virtual void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
                      size_t position, std::size_t headDimension) = 0;

    virtual void Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                           size_t validTokenCount, size_t attentionHeadCount, size_t keyValueHeadCount,
                           Tensor& output) = 0;

    virtual void CopyToCache(const Tensor& source, Tensor& cache, size_t position) = 0;

    virtual void Synchronize() = 0;
};
