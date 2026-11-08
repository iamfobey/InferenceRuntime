#include "Math/MathCUDA.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "Utils/Utils.hpp"

namespace
{
    constexpr unsigned int ThreadsPerBlock = 256;

    __device__ float Read(const void* source, const DataType type, const std::size_t index)
    {
        if (type == DataType::Float16)
            return __half2float(static_cast<const __half*>(source)[index]);
        return static_cast<const float*>(source)[index];
    }

    __device__ void Write(void* destination, const DataType type, const std::size_t index, const float value)
    {
        if (type == DataType::Float16)
            static_cast<__half*>(destination)[index] = __float2half_rn(value);
        else
            static_cast<float*>(destination)[index] = value;
    }

    __global__ void EmbeddingKernel(const void* table, const DataType tableType, const std::int32_t* tokenIds,
                                    void* output, const DataType outputType, const std::size_t tokenCount,
                                    const std::size_t vocabularySize, const std::size_t hiddenSize)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            
        if (i >= tokenCount * hiddenSize)
            return;

        const auto token = tokenIds[i / hiddenSize];

        if (token < 0 || static_cast<std::size_t>(token) >= vocabularySize)
            return;

        Write(output, outputType, i, Read(table, tableType, static_cast<std::size_t>(token) * hiddenSize + i % hiddenSize));
    }

    __global__ void LinearKernel(const void* weights, const DataType weightsType, const void* input,
                                 const DataType inputType, void* output, const DataType outputType,
                                 const std::size_t rows, const std::size_t columns)
    {
        const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (row >= rows)
            return;

        float sum{};

        for (std::size_t column{}; column < columns; ++column)
            sum += Read(weights, weightsType, row * columns + column) * Read(input, inputType, column);

        Write(output, outputType, row, sum);
    }

    __global__ void RMSNormKernel(const void* input, const DataType inputType, const void* weight,
                                  const DataType weightType, const float epsilon, void* output,
                                  const DataType outputType, const std::size_t elementCount)
    {
        float meanSquare{};

        for (std::size_t i{}; i < elementCount; ++i)
        {
            const auto value = Read(input, inputType, i);
            meanSquare += value * value;
        }

        meanSquare /= static_cast<float>(elementCount);

        const auto inverseRms = 1.0f / sqrtf(meanSquare + epsilon);

        for (std::size_t i{}; i < elementCount; ++i)
            Write(output, outputType, i, Read(weight, weightType, i) * (Read(input, inputType, i) * inverseRms));
    }

    __global__ void AddKernel(const void* inputA, const DataType inputAType, const void* inputB,
                              const DataType inputBType, void* output, const DataType outputType,
                              const std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(output, outputType, i, Read(inputA, inputAType, i) + Read(inputB, inputBType, i));
    }

    __global__ void MultiplyKernel(const void* inputA, const DataType inputAType, const void* inputB,
                                   const DataType inputBType, void* output, const DataType outputType,
                                   const std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(output, outputType, i, Read(inputA, inputAType, i) * Read(inputB, inputBType, i));
    }

    __global__ void SwiGLUKernel(const void* input, const DataType inputType, const void* gateUp,
                                 const DataType gateUpType, void* output, const DataType outputType,
                                 const std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
        {
            const auto value = Read(input, inputType, i);
            Write(output, outputType, i, value / (1.0f + expf(-value)) * Read(gateUp, gateUpType, i));
        }
    }

    __global__ void ComputeRoPECosSinKernel(void* sourceCos, const DataType cosType, void* sourceSin,
                                            const DataType sinType, const std::size_t position,
                                            const std::size_t headDimension, const float theta)
    {
        const auto p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (p >= headDimension / 2)
            return;

        const auto exponent = static_cast<float>(2 * p) / static_cast<float>(headDimension);
        const auto angle = static_cast<float>(position) / powf(theta, exponent);
        Write(sourceCos, cosType, p, cosf(angle));
        Write(sourceSin, sinType, p, sinf(angle));
    }

    __global__ void RoPEKernel(void* source, const DataType sourceType, const void* inputCos,
                               const DataType cosType, const void* inputSin, const DataType sinType,
                               const std::size_t headCount, const std::size_t headDimension)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const auto halfDimension = headDimension / 2;

        if (i >= headCount * halfDimension)
            return;

        const auto headOffset = (i / halfDimension) * headDimension;
        const auto p = i % halfDimension;
        const auto firstIndex = headOffset + p;
        const auto secondIndex = headOffset + halfDimension + p;
        const auto first = Read(source, sourceType, firstIndex);
        const auto second = Read(source, sourceType, secondIndex);
        const auto cosAngle = Read(inputCos, cosType, p);
        const auto sinAngle = Read(inputSin, sinType, p);
        Write(source, sourceType, firstIndex, first * cosAngle - second * sinAngle);
        Write(source, sourceType, secondIndex, first * sinAngle + second * cosAngle);
    }

    __global__ void AttentionKernel(const void* q, const DataType qType, const void* kCache, const DataType kType,
                                    const void* vCache, const DataType vType, void* output, const DataType outputType,
                                    const std::size_t validTokenCount, const std::size_t attentionHeadCount,
                                    const std::size_t keyValueHeadCount, const std::size_t headDimension)
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
            Write(output, outputType, qOffset + d, 0.0f);

        for (std::size_t t{}; t < validTokenCount; ++t)
        {
            const auto kvOffset = (t * keyValueHeadCount + kvHead) * headDimension;
            float score{};

            for (std::size_t d{}; d < headDimension; ++d)
                score += Read(q, qType, qOffset + d) * Read(kCache, kType, kvOffset + d);

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
                const auto outputValue = Read(output, outputType, qOffset + d);
                const auto value = Read(vCache, vType, kvOffset + d);
                Write(output, outputType, qOffset + d, outputValue * correction + value * probability);
            }
        }
        const auto inverseProbabilitySum = 1.0f / probabilitySum;

        for (std::size_t d{}; d < headDimension; ++d)
            Write(output, outputType, qOffset + d, Read(output, outputType, qOffset + d) * inverseProbabilitySum);
    }

    __global__ void CopyToCacheKernel(const void* source, const DataType sourceType, void* cache,
                                      const DataType cacheType, const std::size_t position,
                                      const std::size_t elementCount)
    {
        const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < elementCount)
            Write(cache, cacheType, position * elementCount + i, Read(source, sourceType, i));
    }
}

namespace Math::CUDA
{
    void Embedding(const void* table, const DataType tableType, const std::int32_t* tokenIds, void* output,
                   const DataType outputType, const std::size_t tokenCount, const std::size_t vocabularySize,
                   const std::size_t hiddenSize)
    {
        const auto count = tokenCount * hiddenSize;

        if (count != 0)
        {
            EmbeddingKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                table, tableType, tokenIds, output, outputType, tokenCount, vocabularySize, hiddenSize);

            CUDA_CHECK_KERNEL();
        }
    }

    void Linear(const void* weights, const DataType weightsType, const void* input, const DataType inputType,
                void* output, const DataType outputType, const std::size_t rows, const std::size_t columns)
    {
        if (rows != 0)
        {
            LinearKernel<<<static_cast<unsigned int>((rows + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                weights, weightsType, input, inputType, output, outputType, rows, columns);

            CUDA_CHECK_KERNEL();
        }
    }

    void RMSNorm(const void* input, const DataType inputType, const void* weight, const DataType weightType,
                 const float epsilon, void* output, const DataType outputType, const std::size_t elementCount)
    {
        if (elementCount != 0)
        {
            RMSNormKernel<<<1, 1>>>(input, inputType, weight, weightType, epsilon, output, outputType, elementCount);

            CUDA_CHECK_KERNEL();
        }
    }

    void Add(const void* a, const DataType aType, const void* b, const DataType bType,
             void* output, const DataType outputType, const std::size_t count)
    {
        if (count != 0)
        {
            AddKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                a, aType, b, bType, output, outputType, count);

            CUDA_CHECK_KERNEL();
        }
    }

    void Multiply(const void* a, const DataType aType, const void* b, const DataType bType,
                  void* output, const DataType outputType, const std::size_t count)
    {
        if (count != 0)
        {
            MultiplyKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                a, aType, b, bType, output, outputType, count);

            CUDA_CHECK_KERNEL();
        }
    }

    void SwiGLU(const void* input, const DataType inputType, const void* gateUp, const DataType gateUpType,
                void* output, const DataType outputType, const std::size_t count)
    {
        if (count != 0)
        {
            SwiGLUKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                input, inputType, gateUp, gateUpType, output, outputType, count);

            CUDA_CHECK_KERNEL();
        }
    }

    void ComputeRoPECosSin(void* cos, const DataType cosType, void* sin, const DataType sinType,
                           const std::size_t position, const std::size_t dimension, const float theta)
    {
        const auto count = dimension / 2;
        ComputeRoPECosSinKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
            cos, cosType, sin, sinType, position, dimension, theta);

        CUDA_CHECK_KERNEL();
    }

    void RoPE(void* source, const DataType sourceType, const void* cos, const DataType cosType,
              const void* sin, const DataType sinType, const std::size_t heads, const std::size_t dimension)
    {
        const auto count = heads * (dimension / 2);
        RoPEKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
            source, sourceType, cos, cosType, sin, sinType, heads, dimension);

        CUDA_CHECK_KERNEL();
    }

    void Attention(const void* q, const DataType qType, const void* k, const DataType kType,
                   const void* v, const DataType vType, void* output, const DataType outputType,
                   const std::size_t tokens, const std::size_t heads, const std::size_t kvHeads,
                   const std::size_t dimension)
    {
        if (tokens != 0)
        {
            AttentionKernel<<<static_cast<unsigned int>((heads + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                q, qType, k, kType, v, vType, output, outputType, tokens, heads, kvHeads, dimension);

            CUDA_CHECK_KERNEL();
        }
    }

    void CopyToCache(const void* source, const DataType sourceType, void* cache, const DataType cacheType,
                     const std::size_t position, const std::size_t count)
    {
        if (count != 0)
        {
            CopyToCacheKernel<<<static_cast<unsigned int>((count + ThreadsPerBlock - 1) / ThreadsPerBlock), ThreadsPerBlock>>>(
                source, sourceType, cache, cacheType, position, count);

            CUDA_CHECK_KERNEL();
        }
    }
}
