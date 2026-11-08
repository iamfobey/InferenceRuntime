#pragma once

#if RUNTIME_ENABLE_CUDA
#include "Core/Core.hpp"

#include <cstddef>
#include <cstdint>

namespace Math::CUDA
{
    void Embedding(const void* table, DataType tableType, const std::int32_t* tokenIds, void* output,
                   DataType outputType, std::size_t tokenCount, std::size_t vocabularySize, std::size_t hiddenSize);

    void Linear(const void* weights, DataType weightsType, const void* input, DataType inputType,
                void* output, DataType outputType, std::size_t rows, std::size_t columns);

    void RMSNorm(const void* input, DataType inputType, const void* weight, DataType weightType,
                 float epsilon, void* output, DataType outputType, std::size_t elementCount);

    void Add(const void* inputA, DataType inputAType, const void* inputB, DataType inputBType,
             void* output, DataType outputType, std::size_t elementCount);

    void Multiply(const void* inputA, DataType inputAType, const void* inputB, DataType inputBType,
                  void* output, DataType outputType, std::size_t elementCount);

    void SwiGLU(const void* input, DataType inputType, const void* gateUp, DataType gateUpType,
                void* output, DataType outputType, std::size_t elementCount);

    void ComputeRoPECosSin(void* sourceCos, DataType cosType, void* sourceSin, DataType sinType,
                           std::size_t position, std::size_t headDimension, float theta);

    void RoPE(void* source, DataType sourceType, const void* inputCos, DataType cosType,
              const void* inputSin, DataType sinType, std::size_t headCount, std::size_t headDimension);

    void Attention(const void* q, DataType qType, const void* kCache, DataType kType,
                   const void* vCache, DataType vType, void* output, DataType outputType,
                   std::size_t validTokenCount, std::size_t attentionHeadCount,
                   std::size_t keyValueHeadCount, std::size_t headDimension);

    void CopyToCache(const void* source, DataType sourceType, void* cache, DataType cacheType,
                     std::size_t position, std::size_t elementCount);
}
#endif
