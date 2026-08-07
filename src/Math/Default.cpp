#if !HAVE_SUPPORT_AVX2

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Math/Math.hpp"

namespace Math
{
    void Embedding(
        const float* embeddingTable,
        const std::int32_t* tokenIds,
        float* output,
        const std::size_t tokenCount,
        const std::size_t vocabularySize,
        const std::size_t hiddenSize
    )
    {
        for (std::size_t t = 0; t < tokenCount; ++t)
        {
            const std::int32_t token = tokenIds[t];

            if (token < 0)
            {
                continue;
            }

            const auto tokenIndex =
                static_cast<std::size_t>(token);

            if (tokenIndex >= vocabularySize)
            {
                continue;
            }

            const std::size_t sourceOffset =
                tokenIndex * hiddenSize;

            const std::size_t destinationOffset =
                t * hiddenSize;

            for (std::size_t h = 0; h < hiddenSize; ++h)
            {
                output[destinationOffset + h] =
                    embeddingTable[sourceOffset + h];
            }
        }
    }

    void Linear(
        const float* mat,
        const float* vec,
        const float* offset,
        float* output,
        const std::size_t matRows,
        const std::size_t matColumns
    )
    {
#pragma omp parallel for
        for (std::int64_t j = 0; j < matRows; ++j)
        {
            float y_j = offset[j];

            for (std::size_t k = 0; k < matColumns; ++k)
            {
                y_j += mat[j * matColumns + k] * vec[k];
            }

            output[j] = y_j;
        }
    }

    void RMSNorm(
        const float* x,
        const float* weight,
        const float epsilon,
        float* y,
        const std::size_t elementCount
    )
    {
        if (elementCount == 0)
        {
            return;
        }

        float meanSquare = 0.0f;
        float rms = 0.0f;

#pragma omp parallel shared(meanSquare, rms)
        {
#pragma omp for reduction(+:meanSquare)
            for (std::int64_t i = 0; i < elementCount; ++i)
            {
                const float x_i = x[i];

                meanSquare +=
                    x_i * x_i;
            }

#pragma omp single
            {
                meanSquare /=
                    static_cast<float>(elementCount);

                rms =
                    std::sqrt(
                        meanSquare +
                        epsilon
                    );
            }

#pragma omp for
            for (std::int64_t i = 0; i < elementCount; ++i)
            {
                y[i] =
                    weight[i] *
                    (x[i] / rms);
            }
        }
    }

    void Add(
        const float* a,
        const float* b,
        float* output,
        const std::size_t elementCount
    )
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
        {
            output[i] =
                a[i] + b[i];
        }
    }

    void Multiply(
        const float* a,
        const float* b,
        float* output,
        const std::size_t elementCount
    )
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
        {
            output[i] =
                a[i] * b[i];
        }
    }

    [[nodiscard]]
    std::size_t CheckedMultiply(
        const std::size_t a,
        const std::size_t b
    )
    {
        if (
            a != 0 &&
            b >
            std::numeric_limits<std::size_t>::max() / a
        )
        {
            throw std::overflow_error(
                "Tensor size overflow"
            );
        }

        return a * b;
    }

    void SiLU(
        const float* x,
        float* output,
        const std::size_t elementCount
    )
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < elementCount; ++i)
        {
            const float x_i = x[i];

            output[i] =
                x_i /
                (1.0f + std::exp(-x_i));
        }
    }

    void RoPE(
        float* source,
        const std::size_t position,
        const std::size_t headCount,
        const std::size_t headDimension,
        const float theta
    )
    {
        if (
            headDimension == 0 ||
            headDimension % 2 != 0
        )
        {
            return;
        }

        const std::size_t halfDimension =
            headDimension / 2;

        for (
            std::size_t h = 0;
            h < headCount;
            ++h
        )
        {
            const std::size_t headOffset =
                h * headDimension;

            for (
                std::size_t p = 0;
                p < halfDimension;
                ++p
            )
            {
                const float exponent =
                    static_cast<float>(2 * p) /
                    static_cast<float>(headDimension);

                const float inverseFrequency =
                    1.0f /
                    std::pow(theta, exponent);

                const float angle =
                    static_cast<float>(position) *
                    inverseFrequency;

                const float cosAngle =
                    std::cos(angle);

                const float sinAngle =
                    std::sin(angle);

                const std::size_t firstIndex =
                    headOffset + p;

                const std::size_t secondIndex =
                    headOffset +
                    halfDimension +
                    p;

                const float first =
                    source[firstIndex];

                const float second =
                    source[secondIndex];

                source[firstIndex] =
                    first * cosAngle -
                    second * sinAngle;

                source[secondIndex] =
                    first * sinAngle +
                    second * cosAngle;
            }
        }
    }

    void Attention(
        const float* q,
        const float* kCache,
        const float* vCache,
        float* output,
        const std::size_t validTokenCount,
        const std::size_t attentionHeadCount,
        const std::size_t keyValueHeadCount,
        const std::size_t headDimension
    )
    {
        if (
            validTokenCount == 0 ||
            attentionHeadCount == 0 ||
            keyValueHeadCount == 0 ||
            headDimension == 0
        )
        {
            return;
        }

        if (
            attentionHeadCount %
            keyValueHeadCount != 0
        )
        {
            return;
        }

        const std::size_t groupSize =
            attentionHeadCount /
            keyValueHeadCount;

        const float scale =
            1.0f /
            std::sqrt(
                static_cast<float>(
                    headDimension
                )
            );

        std::vector<float> scores(
            validTokenCount
        );

        for (
            std::size_t h = 0;
            h < attentionHeadCount;
            ++h
        )
        {
            const std::size_t kvHead =
                h / groupSize;

            const std::size_t qOffset =
                h * headDimension;

            float maximumScore =
                -std::numeric_limits<float>::infinity();

            for (
                std::size_t t = 0;
                t < validTokenCount;
                ++t
            )
            {
                const std::size_t kOffset =
                    (
                        t *
                        keyValueHeadCount +
                        kvHead
                    ) *
                    headDimension;

                float score_h_t = 0.0f;

                for (
                    std::size_t d = 0;
                    d < headDimension;
                    ++d
                )
                {
                    score_h_t +=
                        q[qOffset + d] *
                        kCache[kOffset + d];
                }

                score_h_t *= scale;

                scores[t] =
                    score_h_t;

                maximumScore =
                    std::max(
                        maximumScore,
                        score_h_t
                    );
            }

            float probabilitySum = 0.0f;

            for (
                std::size_t t = 0;
                t < validTokenCount;
                ++t
            )
            {
                scores[t] =
                    std::exp(
                        scores[t] -
                        maximumScore
                    );

                probabilitySum +=
                    scores[t];
            }

            const float inverseProbabilitySum =
                1.0f /
                probabilitySum;

            const std::size_t outputOffset =
                h * headDimension;

            for (
                std::size_t d = 0;
                d < headDimension;
                ++d
            )
            {
                output[outputOffset + d] =
                    0.0f;
            }

            for (
                std::size_t t = 0;
                t < validTokenCount;
                ++t
            )
            {
                const float probability_h_t =
                    scores[t] *
                    inverseProbabilitySum;

                const std::size_t vOffset =
                    (
                        t *
                        keyValueHeadCount +
                        kvHead
                    ) *
                    headDimension;

                for (
                    std::size_t d = 0;
                    d < headDimension;
                    ++d
                )
                {
                    output[outputOffset + d] +=
                        probability_h_t *
                        vCache[vOffset + d];
                }
            }
        }
    }

    void CopyToCache(
        const float* source,
        float* cache,
        const std::size_t position,
        const std::size_t elementCount
    )
    {
        const std::size_t cacheOffset =
            position * elementCount;

        for (
            std::size_t i = 0;
            i < elementCount;
            ++i
        )
        {
            cache[cacheOffset + i] =
                source[i];
        }
    }
}

#endif
