#pragma once

#include <cstddef>
#include <cstdint>

namespace Math
{
    [[nodiscard]]
    std::uint16_t Float32ToFloat16(float value) noexcept;

    [[nodiscard]]
    float Float16ToFloat32(std::uint16_t value) noexcept;

    void ConvertFloat32ToFloat16(const float* source, std::uint16_t* destination, std::size_t elementCount);

    void ConvertFloat16ToFloat32(const std::uint16_t* source, float* destination, std::size_t elementCount);

    void Embedding(const std::uint16_t* embeddingTable, const std::int32_t* tokenIds, float* output,
                   std::size_t tokenCount, std::size_t vocabularySize, std::size_t hiddenSize);

    void Linear(const std::uint16_t* matrix, const float* input, const float* bias, float* output, std::size_t rows,
                std::size_t columns);

    void RMSNorm(const float* x, const std::uint16_t* weight, float epsilon, float* y, std::size_t elementCount);

    void Add(const float* a, const float* b, float* output, std::size_t elementCount);

    void Multiply(const float* a, const float* b, float* output, std::size_t elementCount);

    [[nodiscard]]
    std::size_t CheckedMultiply(std::size_t a, std::size_t b);

    void SiLU(const float* x, float* output, std::size_t elementCount);

    void RoPE(float* source, std::size_t position, std::size_t headCount, std::size_t headDimension, float theta);

    void Attention(const float* q, const float* kCache, const float* vCache, float* output, std::size_t validTokenCount,
                   std::size_t attentionHeadCount, std::size_t keyValueHeadCount, std::size_t headDimension);

    void CopyToCache(const float* source, float* cache, std::size_t position, std::size_t elementCount);
}
