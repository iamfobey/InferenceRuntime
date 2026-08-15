#pragma once

#include <cstddef>
#include <cstdint>

namespace Math
{
    void Embedding(const std::uint16_t* pEmbeddingTable, const std::int32_t* pTokenIds, float* pOutput,
                   size_t tokenCount, size_t vocabularySize, size_t hiddenSize);

    void Linear(const std::uint16_t* pMatrix, const float* pInput, float* pOutput, std::size_t rows,
                std::size_t columns);

    void RMSNorm(const float* pX, const std::uint16_t* pWeight, float epsilon, float* pY, size_t elementCount);

    void Add(const float* pA, const float* pB, float* pOutput, size_t elementCount);

    void Multiply(const float* a, const float* b, float* output, size_t elementCount);

    [[nodiscard]]
    std::size_t CheckedMultiply(size_t a, size_t b);

    void SiLU(const float* pX, float* pOutput, size_t elementCount);

    void SinCosRoPE(float* pSourceCos, float* pSourceSin, std::size_t position, std::size_t headDimension, float theta);

    void RoPE(float* pSource, const float* pInputCos, const float* pInputSin, size_t headCount,
              size_t headDimension);

    void Attention(const float* pQ, const float* pKCache, const float* pVCache, float* pOutput, float* scores,
                   size_t validTokenCount, size_t attentionHeadCount, size_t keyValueHeadCount, size_t headDimension);

    void CopyToCache(const float* pSource, float* pCache, size_t position, size_t elementCount);
}
