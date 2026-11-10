#pragma once

#include <cstddef>
#include <cstdint>

namespace Math::CPU
{
    template <class EmbeddingType, class OutputType>
    void Embedding(const EmbeddingType* pTable, const std::int32_t* pTokenIds, OutputType* pOutput,
                   std::size_t tokenCount, std::size_t vocabularySize, std::size_t hiddenSize);

    template <class WeightType, class InputType, class OutputType>
    void LinearRange(const WeightType* pWeights, const InputType* pInput, OutputType* pOutput,
                     std::size_t beginRow, std::size_t endRow, std::size_t columns);

    template <class InputType, class WeightType, class OutputType>
    void RMSNorm(const InputType* pInput, const WeightType* pWeight, float epsilon, OutputType* pOutput,
                 std::size_t elementCount);

    template <class InputAType, class InputBType, class OutputType>
    void Add(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, std::size_t elementCount);

    template <class InputAType, class InputBType, class OutputType>
    void Multiply(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, std::size_t elementCount);

    [[nodiscard]]
    std::size_t CheckedMultiply(std::size_t inputA, std::size_t inputB);

    template <class InputType, class GateUpType, class OutputType>
    void SwiGLU(const InputType* pInput, const GateUpType* pGateUp, OutputType* pOutput, std::size_t elementCount);

    template <class OutputCosType, class OutputSinType>
    void ComputeRoPECosSin(OutputCosType* pOutputCos, OutputSinType* pOutputSin, std::size_t position, std::size_t headDimension,
                           float theta);

    template <class ValueType, class CosType, class SinType>
    void RoPE(ValueType* pData, const CosType* pInputCos, const SinType* pInputSin, std::size_t headCount,
              std::size_t headDimension);

    template <class QType, class KType, class VType, class OutputType>
    void Attention(const QType* pQuery, const KType* pKCache, const VType* pVCache, OutputType* pOutput,
                   std::size_t validTokenCount, std::size_t attentionHeadCount, std::size_t keyValueHeadCount,
                   std::size_t headDimension);

    template <class SourceType, class CacheType>
    void CopyToCache(const SourceType* pSrc, CacheType* pDest, std::size_t position, std::size_t elementCount);
}
