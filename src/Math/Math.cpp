#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if HAVE_AVX2_SUPPORT
#include <immintrin.h>
#endif

#include "Math/Math.hpp"
#include "Utils/Converters.hpp"

namespace Math
{
    void Embedding(const std::uint16_t* pEmbeddingTable, const std::int32_t* pTokenIds, float* pOutput,
                   size_t tokenCount, size_t vocabularySize, size_t hiddenSize)
    {
        for (std::size_t t{}; t < tokenCount; ++t)
        {
            const auto token = pTokenIds[t];

            if (token < 0)
                continue;

            const auto tokenIndex = static_cast<std::size_t>(token);

            if (tokenIndex >= vocabularySize)
                continue;

            const auto sourceOffset = tokenIndex * hiddenSize;
            const auto destinationOffset = t * hiddenSize;

            Utils::Converters::ConvertFloat16ToFloat32(pEmbeddingTable + sourceOffset, pOutput + destinationOffset,
                                                       hiddenSize);
        }
    }

    void Linear(const std::uint16_t* pMatrix, const float* pInput, float* pOutput, std::size_t rows,
                std::size_t columns)
    {
#pragma omp parallel for
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows); ++row)
        {
            const auto* matrixRow = pMatrix + static_cast<std::size_t>(row) * columns;

#if HAVE_AVX2_SUPPORT
            auto accumulator1 = _mm256_setzero_ps();
            auto accumulator2 = _mm256_setzero_ps();
            auto accumulator3 = _mm256_setzero_ps();
            auto accumulator4 = _mm256_setzero_ps();

            std::size_t column{};
            for (; column + 32 <= columns; column += 32)
            {
                const auto inpVec1 = _mm256_loadu_ps(pInput + column);
                const auto matVec1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrixRow + column)));
                const auto inpVec2 = _mm256_loadu_ps(pInput + column + 8);
                const auto matVec2 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrixRow + column + 8)));
                const auto inpVec3 = _mm256_loadu_ps(pInput + column + 16);
                const auto matVec3 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrixRow + column + 16)));
                const auto inpVec4 = _mm256_loadu_ps(pInput + column + 24);
                const auto matVec4 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrixRow + column + 24)));

                accumulator1 = _mm256_fmadd_ps(matVec1, inpVec1, accumulator1);
                accumulator2 = _mm256_fmadd_ps(matVec2, inpVec2, accumulator2);
                accumulator3 = _mm256_fmadd_ps(matVec3, inpVec3, accumulator3);
                accumulator4 = _mm256_fmadd_ps(matVec4, inpVec4, accumulator4);
            }

            alignas(32) float values[32];
            _mm256_store_ps(values, accumulator1);
            _mm256_store_ps(values + 8, accumulator2);
            _mm256_store_ps(values + 16, accumulator3);
            _mm256_store_ps(values + 24, accumulator4);

            float sum = 0.0f;

            for (const auto value : values)
                sum += value;

            for (; column < columns; ++column)
                sum += Utils::Converters::Float16ToFloat32(matrixRow[column]) * pInput[column];

            pOutput[row] = sum;
#else
            auto sum = 0.0f;

            for (std::size_t column{}; column < columns; ++column)
                sum += Utils::Converters::Float16ToFloat32(matrixRow[column]) * pInput[column];

            pOutput[row] = sum;
#endif
        }
    }

    void RMSNorm(const float* pX, const std::uint16_t* pWeight, float epsilon, float* pY,
                 size_t elementCount)
    {
        if (elementCount == 0)
            return;

        float meanSquare{};

        for (std::size_t i{}; i < elementCount; ++i)
            meanSquare += pX[i] * pX[i];

        meanSquare /= static_cast<float>(elementCount);
        const auto inverseRms = 1.0f / std::sqrt(meanSquare + epsilon);

        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        const auto inverseRmsVec = _mm256_set1_ps(inverseRms);

        for (; i + 8 <= elementCount; i += 8)
        {
            const auto xVec = _mm256_loadu_ps(pX + i);
            const auto weightHalf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pWeight + i));
            const auto weightVec = _mm256_cvtph_ps(weightHalf);
            const auto normalized = _mm256_mul_ps(xVec, inverseRmsVec);
            const auto result = _mm256_mul_ps(weightVec, normalized);
            _mm256_storeu_ps(pY + i, result);
        }
#endif

        for (; i < elementCount; ++i)
            pY[i] = Utils::Converters::Float16ToFloat32(pWeight[i]) * (pX[i] * inverseRms);
    }

    void Add(const float* pA, const float* pB, float* pOutput, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            pOutput[i] = pA[i] + pB[i];
    }

    void Multiply(const float* a, const float* b, float* output, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            output[i] = a[i] * b[i];
    }

    std::size_t CheckedMultiply(size_t a, size_t b)
    {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::overflow_error("Tensor size overflow");

        return a * b;
    }

    void SiLU(const float* pX, float* pOutput, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
        {
            const auto x_i = pX[i];
            pOutput[i] = x_i / (1.0f + std::exp(-x_i));
        }
    }

    void SinCosRoPE(float* pSourceCos, float* pSourceSin, std::size_t position, std::size_t headDimension, float theta)
    {
        if (headDimension == 0 || headDimension % 2 != 0)
            return;

        const auto halfDimension = headDimension / 2;

        for (std::size_t p{}; p < halfDimension; ++p)
        {
            const auto exponent = static_cast<float>(2 * p) / static_cast<float>(headDimension);
            const auto inverseFrequency = 1.0f / std::pow(theta, exponent);
            const auto angle = static_cast<float>(position) * inverseFrequency;
            const auto cosAngle = std::cos(angle);
            const auto sinAngle = std::sin(angle);
            pSourceCos[p] = cosAngle;
            pSourceSin[p] = sinAngle;
        }
    }

    void RoPE(float* pSource, const float* pInputCos, const float* pInputSin, size_t headCount,
              size_t headDimension)
    {
        if (headDimension == 0 || headDimension % 2 != 0)
            return;

        const auto halfDimension = headDimension / 2;

        for (std::size_t h{}; h < headCount; ++h)
        {
            const auto headOffset = h * headDimension;

            for (std::size_t p{}; p < halfDimension; ++p)
            {
                const auto firstIndex = headOffset + p;
                const auto secondIndex = headOffset + halfDimension + p;
                const auto first = pSource[firstIndex];
                const auto second = pSource[secondIndex];

                const auto cosAngle = pInputCos[p];
                const auto sinAngle = pInputSin[p];

                pSource[firstIndex] = first * cosAngle - second * sinAngle;
                pSource[secondIndex] = first * sinAngle + second * cosAngle;
            }
        }
    }

    void Attention(const float* pQ, const float* pKCache, const float* pVCache, float* pOutput,
                   float* scores, size_t validTokenCount,
                   size_t attentionHeadCount, size_t keyValueHeadCount,
                   size_t headDimension)
    {
        if (validTokenCount == 0 || attentionHeadCount == 0 || keyValueHeadCount == 0 || headDimension == 0)
            return;

        if (attentionHeadCount % keyValueHeadCount != 0)
            return;

        const auto groupSize = attentionHeadCount / keyValueHeadCount;
        const auto scale = 1.0f / std::sqrt(static_cast<float>(headDimension));

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
                    score_h_t += pQ[qOffset + d] * pKCache[kOffset + d];
                score_h_t *= scale;
                scores[t] = score_h_t;
                maximumScore = std::fmaxf(maximumScore, score_h_t);
            }

            const auto outputOffset = h * headDimension;
            for (std::size_t d{}; d < headDimension; ++d)
                pOutput[outputOffset + d] = 0.0f;

            float probabilitySum{};
            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto e = std::exp(scores[t] - maximumScore);
                probabilitySum += e;
                const auto vOffset = (t * keyValueHeadCount + kvHead) * headDimension;
                for (std::size_t d{}; d < headDimension; ++d)
                    pOutput[outputOffset + d] += e * pVCache[vOffset + d];
            }

            const auto inverseProbabilitySum = 1.0f / probabilitySum;
            for (std::size_t d{}; d < headDimension; ++d)
                pOutput[outputOffset + d] *= inverseProbabilitySum;
        }
    }

    void CopyToCache(const float* pSource, float* pCache, size_t position, size_t elementCount)
    {
        const auto cacheOffset = position * elementCount;

        for (std::size_t i{}; i < elementCount; ++i)
            pCache[cacheOffset + i] = pSource[i];
    }
}
