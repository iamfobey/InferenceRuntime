#pragma once

#if RUNTIME_ENABLE_CUDA
#include "Core/Core.hpp"

#include <cstddef>
#include <cstdint>

namespace Math::CUDA
{
    void Embedding(const void* pTable, DataType tableType, const std::int32_t* pTokenIds, void* pOutput,
                   DataType outputType, std::size_t tokenCount, std::size_t vocabularySize, std::size_t hiddenSize);

    void Linear(const void* pWeights, DataType weightsType, const void* pInput, DataType inputType,
                void* pOutput, DataType outputType, std::size_t rows, std::size_t columns);

    void RMSNorm(const void* pInput, DataType inputType, const void* pWeight, DataType weightType,
                 float epsilon, void* pOutput, DataType outputType, std::size_t elementCount);

    void Add(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType,
             void* pOutput, DataType outputType, std::size_t elementCount);

    void Multiply(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType,
                  void* pOutput, DataType outputType, std::size_t elementCount);

    void SwiGLU(const void* pInput, DataType inputType, const void* pGateUp, DataType gateUpType,
                void* pOutput, DataType outputType, std::size_t elementCount);

    void ComputeRoPECosSin(void* pOutputCos, DataType outputCosType, void* pOutputSin, DataType outputSinType,
                           std::size_t position, std::size_t headDimension, float theta);

    void RoPE(void* pData, DataType dataType, const void* pInputCos, DataType inputCosType,
              const void* pInputSin, DataType inputSinType, std::size_t headCount, std::size_t headDimension);

    void Attention(const void* pQuery, DataType queryType, const void* pKCache, DataType kCacheType,
                   const void* pVCache, DataType vCacheType, void* pOutput, DataType outputType,
                   std::size_t validTokenCount, std::size_t attentionHeadCount,
                   std::size_t keyValueHeadCount, std::size_t headDimension);

    void CopyToCache(const void* pSrc, DataType srcType, void* pCache, DataType cacheType,
                     std::size_t position, std::size_t elementCount);
}
#endif
