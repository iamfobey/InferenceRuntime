#pragma once

#if RUNTIME_ENABLE_CUDA
#include "Backend/IBackend.hpp"

class CudaBackend final : public IBackend
{
public:
    CudaBackend();

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

    void SwiGLU(const Tensor& input, const Tensor& gateUp, Tensor& output) override;

    void ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                           float theta) override;

    void RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
              std::size_t position, std::size_t headDimension) override;

    void Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                   std::size_t validTokenCount, std::size_t attentionHeadCount, std::size_t keyValueHeadCount,
                   Tensor& output) override;

    void CopyToCache(const Tensor& source, Tensor& cache, std::size_t position) override;

    void Synchronize() override;
};
#endif
