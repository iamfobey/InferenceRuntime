#pragma once

#include <cstddef>
#include <cstdint>

namespace Math
{
    template <class EmbeddingType, class OutputType>
    void Embedding(const EmbeddingType* pEmbeddingTable, const std::int32_t* pTokenIds, OutputType* pOutput,
                   size_t tokenCount, size_t vocabularySize, size_t hiddenSize);

    template <class WeightType, class InputType, class OutputType>
    void LinearRange(const WeightType* pMatrix, const InputType* pInput, OutputType* pOutput,
                     std::size_t beginRow, std::size_t endRow, std::size_t columns);

    template <class InputType, class WeightType, class OutputType>
    void RMSNorm(const InputType* pInput, const WeightType* pWeight, float epsilon, OutputType* pOutput, size_t elementCount);

    template <class InputAType, class InputBType, class OutputType>
    void Add(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, size_t elementCount);

    template <class InputAType, class InputBType, class OutputType>
    void Multiply(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, size_t elementCount);

    [[nodiscard]]
    std::size_t CheckedMultiply(size_t inputA, size_t inputB);

    template <class InputType, class OutputType>
    void SiLU(const InputType* pInput, OutputType* pOutput, size_t elementCount);

    template <class CosType, class SinType>
    void ComputeRoPECosSin(CosType* pSourceCos, SinType* pSourceSin, std::size_t position, std::size_t headDimension, float theta);

    template <class SourceType, class CosType, class SinType>
    void RoPE(SourceType* pSource, const CosType* pInputCos, const SinType* pInputSin, size_t headCount, size_t headDimension);

    template <class QType, class KType, class VType, class OutputType>
    void Attention(const QType* pQ, const KType* pKCache, const VType* pVCache, OutputType* pOutput,
                   size_t validTokenCount, size_t attentionHeadCount, size_t keyValueHeadCount, size_t headDimension);

    template <class SourceType, class CacheType>
    void CopyToCache(const SourceType* pSource, CacheType* pCache, size_t position, size_t elementCount);
}
