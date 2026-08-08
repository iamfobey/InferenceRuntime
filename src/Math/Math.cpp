#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Math/Math.hpp"

#if HAVE_AVX2_SUPPORT
#include <immintrin.h>
#endif

namespace Math
{
    void Embedding(const float* embeddingTable,
                   const std::int32_t* tokenIds, float* output, const std::size_t tokenCount,
                   const std::size_t vocabularySize,
                   const std::size_t hiddenSize)
    {
        for (std::size_t t{}; t < tokenCount; ++t)
        {
            const auto token = tokenIds[t];

            if (token < 0)
                continue;

            const auto tokenIndex = static_cast<std::size_t>(token);

            if (tokenIndex >= vocabularySize)
                continue;

            const auto sourceOffset = tokenIndex * hiddenSize;
            const auto destinationOffset = t * hiddenSize;

            for (std::size_t h{}; h < hiddenSize; ++h)
                output[destinationOffset + h] = embeddingTable[sourceOffset + h];
        }
    }

    void Linear(const float* matrix, const float* input, const float* bias, float* output, const std::size_t rows,
                const std::size_t columns)
    {
#pragma omp parallel for
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows); ++row)
        {
#if HAVE_AVX2_SUPPORT
            __m256 accumulator = _mm256_setzero_ps();

            std::size_t column{};
            for (; column + 8 <= columns; column += 8)
            {
                const auto inputVec = _mm256_loadu_ps(input + column);
                const auto matrixVec = _mm256_loadu_ps(matrix + row * columns + column);
                const auto product = _mm256_mul_ps(matrixVec, inputVec);

                accumulator = _mm256_add_ps(accumulator, product);
            }

            alignas(32) float values[8];
            _mm256_store_ps(values, accumulator);

            auto sum = bias[row];

            for (const auto value : values)
                sum += value;

            for (; column < columns; ++column)
                sum += matrix[row * columns + column] * input[column];

            output[row] = sum;
#else
            auto sum = bias[row];

            for (std::size_t column{}; column < columns; ++column)
                sum += matrix[row * columns + column] * input[column];

            output[row] = sum;
#endif
        }
    }

    void RMSNorm(const float* x, const float* weight, const float epsilon, float* y, const std::size_t elementCount)
    {
        if (elementCount == 0)
            return;

        float meanSquare{};
        float rms{};

        for (std::int64_t i{}; i < elementCount; ++i)
            meanSquare += x[i] * x[i];

        meanSquare /= static_cast<float>(elementCount);
        rms = std::sqrt(meanSquare + epsilon);

        for (std::int64_t i{}; i < elementCount; ++i)
            y[i] = weight[i] * (x[i] / rms);
    }

    void Add(const float* a, const float* b, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
            output[i] = a[i] + b[i];
    }

    void Multiply(const float* a, const float* b, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
            output[i] = a[i] * b[i];
    }

    std::size_t CheckedMultiply(const std::size_t a, const std::size_t b)
    {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::overflow_error("Tensor size overflow");

        return a * b;
    }

    void SiLU(const float* x, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
        {
            const auto x_i = x[i];
            output[i] = x_i / (1.0f + std::exp(-x_i));
        }
    }

    void RoPE(float* source, const std::size_t position, const std::size_t headCount, const std::size_t headDimension,
              const float theta)
    {
        if (headDimension == 0 || headDimension % 2 != 0)
            return;

        const auto halfDimension = headDimension / 2;

        for (std::size_t h{}; h < headCount; ++h)
        {
            const auto headOffset = h * headDimension;

            for (std::size_t p{}; p < halfDimension; ++p)
            {
                const auto exponent = static_cast<float>(2 * p) / static_cast<float>(headDimension);
                const auto inverseFrequency = 1.0f / std::pow(theta, exponent);
                const auto angle = static_cast<float>(position) * inverseFrequency;
                const auto cosAngle = std::cos(angle);
                const auto sinAngle = std::sin(angle);
                const auto firstIndex = headOffset + p;
                const auto secondIndex = headOffset + halfDimension + p;
                const auto first = source[firstIndex];
                const auto second = source[secondIndex];

                source[firstIndex] = first * cosAngle - second * sinAngle;
                source[secondIndex] = first * sinAngle + second * cosAngle;
            }
        }
    }

    void Attention(const float* q, const float* kCache, const float* vCache, float* output,
                   const std::size_t validTokenCount, const std::size_t attentionHeadCount,
                   const std::size_t keyValueHeadCount, const std::size_t headDimension)
    {
        if (validTokenCount == 0 || attentionHeadCount == 0 || keyValueHeadCount == 0 || headDimension == 0)
            return;

        if (attentionHeadCount % keyValueHeadCount != 0)
            return;

        const auto groupSize = attentionHeadCount / keyValueHeadCount;
        const auto scale = 1.0f / std::sqrt(static_cast<float>(headDimension));

        std::vector<float> scores(validTokenCount);

        for (std::size_t h{}; h < attentionHeadCount; ++h)
        {
            const auto kvHead = h / groupSize;
            const auto qOffset = h * headDimension;
            auto maximumScore = -std::numeric_limits<float>::infinity();

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto kOffset = (t * keyValueHeadCount + kvHead) * headDimension;

                float score_h_t{};

                for (std::size_t d{}; d < headDimension; ++d)
                    score_h_t += q[qOffset + d] * kCache[kOffset + d];

                score_h_t *= scale;

                scores[t] = score_h_t;

                maximumScore = std::fmaxf(maximumScore, score_h_t);
            }

            float probabilitySum{};

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                scores[t] = std::exp(scores[t] - maximumScore);

                probabilitySum += scores[t];
            }

            const auto inverseProbabilitySum = 1.0f / probabilitySum;
            const auto outputOffset = h * headDimension;

            for (std::size_t d{}; d < headDimension; ++d)
                output[outputOffset + d] = 0.0f;

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto probability_h_t = scores[t] * inverseProbabilitySum;
                const auto vOffset = (t * keyValueHeadCount + kvHead) * headDimension;

                for (std::size_t d{}; d < headDimension; ++d)
                    output[outputOffset + d] += probability_h_t * vCache[vOffset + d];
            }
        }
    }

    void CopyToCache(const float* source, float* cache, const std::size_t position, const std::size_t elementCount)
    {
        const auto cacheOffset = position * elementCount;

        for (std::size_t i{}; i < elementCount; ++i)
            cache[cacheOffset + i] = source[i];
    }
}
