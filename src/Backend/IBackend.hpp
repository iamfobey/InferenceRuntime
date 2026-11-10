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
    virtual Tensor CreateTensor(TensorDimVec shape, DataType dataType) = 0;

    virtual void Upload(Tensor& dest, std::span<const float> src) = 0;

    virtual void Download(const Tensor& src, std::span<float> dest) = 0;

    virtual void Embedding(const Tensor& embeddingTable, std::span<const std::int32_t> tokenIds, Tensor& output) = 0;

    virtual void Linear(const Tensor& weights, const Tensor& input, Tensor& output) = 0;

    virtual void RMSNorm(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output) = 0;

    virtual void Add(const Tensor& inputA, const Tensor& inputB, Tensor& output) = 0;

    virtual void Multiply(const Tensor& inputA, const Tensor& inputB, Tensor& output) = 0;

    virtual void SwiGLU(const Tensor& input, const Tensor&, Tensor& output) = 0;

    virtual void ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, std::size_t position,
                                   std::size_t headDimension,
                                   float theta) = 0;

    virtual void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
                      std::size_t position, std::size_t headDimension) = 0;

    virtual void Attention(const Tensor& query, const Tensor& kCache, const Tensor& vCache,
                           std::size_t validTokenCount, std::size_t attentionHeadCount, std::size_t keyValueHeadCount,
                           Tensor& output) = 0;

    virtual void CopyToCache(const Tensor& src, Tensor& dest, std::size_t position) = 0;

    virtual void Synchronize() = 0;
};
