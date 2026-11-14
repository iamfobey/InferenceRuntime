#include "Math/MathCUDA.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "Utils/Utils.hpp"

namespace
{
    constexpr unsigned int ThreadsPerBlock = 256;

    __device__ float Read(const void* pSrc, DataType type, std::size_t index)
    {
        if (type == DataType::Float16)
            return __half2float(static_cast<const __half*>(pSrc)[index]);

        return static_cast<const float*>(pSrc)[index];
    }

    __device__ void Write(void* pDest, DataType type, std::size_t index, float value)
    {
        if (type == DataType::Float16)
            static_cast<__half*>(pDest)[index] = __float2half_rn(value);
        else
            static_cast<float*>(pDest)[index] = value;
    }

    __global__ void EmbeddingKernel(const void* pTable, DataType tableType, const std::int32_t* pTokenIds, void* pOutput, DataType outputType,
                                    std::size_t tokenCount, std::size_t vocabularySize, std::size_t hiddenSize)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i >= tokenCount * hiddenSize)
            return;

        const auto token = pTokenIds[i / hiddenSize];

        if (token < 0 || static_cast<std::size_t>(token) >= vocabularySize)
            return;

        Write(pOutput, outputType, i, Read(pTable, tableType, static_cast<std::size_t>(token) * hiddenSize + i % hiddenSize));
    }

    __global__ void LinearKernel(const void* pWeights, DataType weightsType, const void* pInput, DataType inputType, void* pOutput,
                                 DataType outputType, std::size_t rows, std::size_t columns)
    {
        const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (row >= rows)
            return;

        float sum{};

        for (std::size_t column{}; column < columns; ++column)
            sum += Read(pWeights, weightsType, row * columns + column) * Read(pInput, inputType, column);

        Write(pOutput, outputType, row, sum);
    }

    __global__ void RMSNormKernel(const void* pInput, DataType inputType, const void* pWeight, DataType weightType, float epsilon, void* pOutput,
                                  DataType outputType, std::size_t elementCount)
    {
        float meanSquare{};

        for (std::size_t i{}; i < elementCount; ++i)
        {
            const auto value = Read(pInput, inputType, i);
            meanSquare += value * value;
        }

        meanSquare /= static_cast<float>(elementCount);

        const auto inverseRms = 1.0f / sqrtf(meanSquare + epsilon);

        for (std::size_t i{}; i < elementCount; ++i)
            Write(pOutput, outputType, i, Read(pWeight, weightType, i) * (Read(pInput, inputType, i) * inverseRms));
    }

    __global__ void AddKernel(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType, void* pOutput, DataType outputType,
                              std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(pOutput, outputType, i, Read(pInputA, inputAType, i) + Read(pInputB, inputBType, i));
    }

    __global__ void MultiplyKernel(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType, void* pOutput,
                                   DataType outputType, std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(pOutput, outputType, i, Read(pInputA, inputAType, i) * Read(pInputB, inputBType, i));
    }

    __global__ void SwiGLUKernel(const void* pInput, DataType inputType, const void* pGateUp, DataType gateUpType, void* pOutput, DataType outputType,
                                 std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
        {
            const auto value = Read(pInput, inputType, i);
            Write(pOutput, outputType, i, value / (1.0f + expf(-value)) * Read(pGateUp, gateUpType, i));
        }
    }

    __global__ void ComputeRoPECosSinKernel(void* pOutputCos, DataType outputCosType, void* pOutputSin, DataType outputSinType, std::size_t position,
                                            std::size_t headDimension, float theta)
    {
        const auto p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (p >= headDimension / 2)
            return;

        const auto exponent = static_cast<float>(2 * p) / static_cast<float>(headDimension);
        const auto angle = static_cast<float>(position) / powf(theta, exponent);

        Write(pOutputCos, outputCosType, p, cosf(angle));
        Write(pOutputSin, outputSinType, p, sinf(angle));
    }

    __global__ void RoPEKernel(void* pData, DataType dataType, const void* pInputCos, DataType inputCosType, const void* pInputSin,
                               DataType inputSinType, std::size_t headCount, std::size_t headDimension)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const auto halfDimension = headDimension / 2;

        if (i >= headCount * halfDimension)
            return;

        const auto headOffset = i / halfDimension * headDimension;
        const auto p = i % halfDimension;
        const auto firstIndex = headOffset + p;
        const auto secondIndex = headOffset + halfDimension + p;
        const auto first = Read(pData, dataType, firstIndex);
        const auto second = Read(pData, dataType, secondIndex);
        const auto cosAngle = Read(pInputCos, inputCosType, p);
        const auto sinAngle = Read(pInputSin, inputSinType, p);

        Write(pData, dataType, firstIndex, first * cosAngle - second * sinAngle);
        Write(pData, dataType, secondIndex, first * sinAngle + second * cosAngle);
    }

    __global__ void AttentionKernel(const void* pQuery, DataType queryType, const void* pKCache, DataType kCacheType,
                                    const void* pVCache, DataType vCacheType, void* pOutput, DataType outputType,
                                    std::size_t validTokenCount,
                                    std::size_t attentionHeadCount, std::size_t keyValueHeadCount, std::size_t headDimension)
    {
        const auto head = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (head >= attentionHeadCount)
            return;

        const auto groupSize = attentionHeadCount / keyValueHeadCount;
        const auto kvHead = head / groupSize;
        const auto qOffset = head * headDimension;
        const auto scale = 1.0f / sqrtf(static_cast<float>(headDimension));
        float currentMaximumScore = __int_as_float(0xff800000);
        float probabilitySum{};

        for (std::size_t d{}; d < headDimension; ++d)
            Write(pOutput, outputType, qOffset + d, 0.0f);

        for (std::size_t t{}; t < validTokenCount; ++t)
        {
            const auto kvOffset = (t * keyValueHeadCount + kvHead) * headDimension;
            float score{};

            for (std::size_t d{}; d < headDimension; ++d)
                score += Read(pQuery, queryType, qOffset + d) * Read(pKCache, kCacheType, kvOffset + d);

            score *= scale;

            float correction;
            float probability;

            if (score > currentMaximumScore)
            {
                correction = expf(currentMaximumScore - score);
                probability = 1.0f;
                currentMaximumScore = score;
            }
            else
            {
                correction = 1.0f;
                probability = expf(score - currentMaximumScore);
            }

            probabilitySum = probabilitySum * correction + probability;

            for (std::size_t d{}; d < headDimension; ++d)
            {
                const auto outputValue = Read(pOutput, outputType, qOffset + d);
                const auto value = Read(pVCache, vCacheType, kvOffset + d);

                Write(pOutput, outputType, qOffset + d, outputValue * correction + value * probability);
            }
        }

        const auto inverseProbabilitySum = 1.0f / probabilitySum;

        for (std::size_t d{}; d < headDimension; ++d)
            Write(pOutput, outputType, qOffset + d, Read(pOutput, outputType, qOffset + d) * inverseProbabilitySum);
    }

    __global__ void CopyToCacheKernel(const void* pSrc, DataType srcType, void* pCache, DataType cacheType, std::size_t position,
                                      std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(pCache, cacheType, position * elementCount + i, Read(pSrc, srcType, i));
    }
}

namespace Math::CUDA
{
    void Embedding(const void* pTable, DataType tableType, const std::int32_t* pTokenIds, void* pOutput, DataType outputType, std::size_t tokenCount,
                   std::size_t vocabularySize, std::size_t hiddenSize)
    {
        const auto count = tokenCount * hiddenSize;

        if (count != 0)
        {
            EmbeddingKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pTable, tableType, pTokenIds, pOutput, outputType, tokenCount, vocabularySize, hiddenSize);

            CUDA_CHECK_KERNEL();
        }
    }

    void Linear(const void* pWeights, DataType weightsType, const void* pInput, DataType inputType, void* pOutput, DataType outputType,
                std::size_t rows,
                std::size_t columns)
    {
        if (rows != 0)
        {
            LinearKernel<<<static_cast<unsigned int>((rows + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pWeights, weightsType, pInput, inputType, pOutput, outputType, rows, columns);

            CUDA_CHECK_KERNEL();
        }
    }

    void RMSNorm(const void* pInput, DataType inputType, const void* pWeight, DataType weightType, float epsilon, void* pOutput, DataType outputType,
                 std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            RMSNormKernel<<<1, 1>>>(pInput, inputType, pWeight, weightType, epsilon, pOutput, outputType, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }

    void Add(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType, void* pOutput, DataType outputType,
             std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            AddKernel<<<static_cast<unsigned int>((elementCount + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pInputA, inputAType, pInputB, inputBType, pOutput, outputType, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }

    void Multiply(const void* pInputA, DataType inputAType, const void* pInputB, DataType inputBType, void* pOutput, DataType outputType,
                  std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            MultiplyKernel<<<static_cast<unsigned int>((elementCount + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pInputA, inputAType, pInputB, inputBType, pOutput, outputType, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }

    void SwiGLU(const void* pInput, DataType inputType, const void* pGateUp, DataType gateUpType, void* pOutput, DataType outputType,
                std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            SwiGLUKernel<<<static_cast<unsigned int>((elementCount + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pInput, inputType, pGateUp, gateUpType, pOutput, outputType, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }

    void ComputeRoPECosSin(void* pOutputCos, DataType outputCosType, void* pOutputSin, DataType outputSinType, std::size_t position,
                           std::size_t headDimension, float theta)
    {
        const auto count = headDimension / 2;
        ComputeRoPECosSinKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
            pOutputCos, outputCosType, pOutputSin, outputSinType, position, headDimension, theta);

        CUDA_CHECK_KERNEL();
    }

    void RoPE(void* pData, DataType dataType, const void* pInputCos, DataType inputCosType, const void* pInputSin, DataType inputSinType,
              std::size_t headCount, std::size_t headDimension)
    {
        const auto count = headCount * (headDimension / 2);
        RoPEKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
            pData, dataType, pInputCos, inputCosType, pInputSin, inputSinType, headCount, headDimension);

        CUDA_CHECK_KERNEL();
    }

    void Attention(const void* pQuery, DataType queryType, const void* pKCache, DataType kCacheType, const void* pVCache,
                   DataType vCacheType, void* pOutput, DataType outputType, std::size_t validTokenCount, std::size_t attentionHeadCount,
                   std::size_t keyValueHeadCount, std::size_t headDimension)
    {
        if (validTokenCount != 0)
        {
            AttentionKernel<<<static_cast<unsigned int>((attentionHeadCount + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pQuery, queryType, pKCache, kCacheType, pVCache, vCacheType, pOutput, outputType, validTokenCount,
                attentionHeadCount, keyValueHeadCount, headDimension);

            CUDA_CHECK_KERNEL();
        }
    }

    void CopyToCache(const void* pSrc, DataType srcType, void* pCache, DataType cacheType, std::size_t position, std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            CopyToCacheKernel<<<static_cast<unsigned int>((elementCount + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                pSrc, srcType, pCache, cacheType, position, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }
}
