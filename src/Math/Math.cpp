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

#include "lowp/lowp.hpp"

namespace
{
#if HAVE_AVX2_SUPPORT
    template <class T>
    __m256 Load8AsFloat(const T* source) noexcept
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return _mm256_loadu_ps(source);
        }
        else if constexpr (std::is_same_v<T, lowp::f16>)
        {
            const auto half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));

            return _mm256_cvtph_ps(half);
        }
        else if constexpr (std::is_same_v<T, lowp::f32>)
        {
            return _mm256_loadu_ps(reinterpret_cast<const float*>(source));
        }
    }

    template <class T>
    void Store8FromFloat(T* destination, const __m256 value) noexcept
    {
        if constexpr (std::is_same_v<T, float>)
        {
            _mm256_storeu_ps(destination, value);
        }
        else if constexpr (std::is_same_v<T, lowp::f16>)
        {
            const auto half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), half);
        }
        else if constexpr (std::is_same_v<T, lowp::f32>)
        {
            _mm256_storeu_ps(reinterpret_cast<float*>(destination), value);
        }
    }
#endif
}

namespace Math
{
    template <class EmbeddingType, class OutputType>
    void Embedding(const EmbeddingType* pEmbeddingTable, const std::int32_t* pTokenIds, OutputType* pOutput,
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
            std::size_t i{};

#if HAVE_AVX2_SUPPORT
            for (; i + 8 <= hiddenSize; i += 8)
                Store8FromFloat(pOutput + destinationOffset + i, Load8AsFloat(pEmbeddingTable + sourceOffset + i));
#endif

            for (; i < hiddenSize; ++i)
                pOutput[destinationOffset + i] = static_cast<OutputType>(static_cast<float>(pEmbeddingTable[sourceOffset + i]));
        }
    }

    template void Embedding<lowp::f16, lowp::f16>(const lowp::f16*, const std::int32_t*, lowp::f16*, size_t, size_t, size_t);
    template void Embedding<lowp::f16, lowp::f32>(const lowp::f16*, const std::int32_t*, lowp::f32*, size_t, size_t, size_t);
    template void Embedding<lowp::f32, lowp::f16>(const lowp::f32*, const std::int32_t*, lowp::f16*, size_t, size_t, size_t);
    template void Embedding<lowp::f32, lowp::f32>(const lowp::f32*, const std::int32_t*, lowp::f32*, size_t, size_t, size_t);

    template <class WeightType, class InputType, class OutputType>
    void LinearRange(const WeightType* pMatrix, const InputType* pInput, OutputType* pOutput,
                     std::size_t beginRow, std::size_t endRow, std::size_t columns)
    {
        for (std::size_t row = beginRow; row < endRow; ++row)
        {
            const auto* matrixRow = pMatrix + row * columns;
            std::size_t column{};
            float sum{};

#if HAVE_AVX2_SUPPORT
            auto accumulator1 = _mm256_setzero_ps();
            auto accumulator2 = _mm256_setzero_ps();
            auto accumulator3 = _mm256_setzero_ps();
            auto accumulator4 = _mm256_setzero_ps();

            for (; column + 32 <= columns; column += 32)
            {
                const auto input1 = Load8AsFloat(pInput + column);
                const auto input2 = Load8AsFloat(pInput + column + 8);
                const auto input3 = Load8AsFloat(pInput + column + 16);
                const auto input4 = Load8AsFloat(pInput + column + 24);

                const auto matrix1 = Load8AsFloat(matrixRow + column);
                const auto matrix2 = Load8AsFloat(matrixRow + column + 8);
                const auto matrix3 = Load8AsFloat(matrixRow + column + 16);
                const auto matrix4 = Load8AsFloat(matrixRow + column + 24);

                accumulator1 = _mm256_fmadd_ps(matrix1, input1, accumulator1);
                accumulator2 = _mm256_fmadd_ps(matrix2, input2, accumulator2);
                accumulator3 = _mm256_fmadd_ps(matrix3, input3, accumulator3);
                accumulator4 = _mm256_fmadd_ps(matrix4, input4, accumulator4);
            }

            const auto accumulator12 = _mm256_add_ps(accumulator1, accumulator2);
            const auto accumulator34 = _mm256_add_ps(accumulator3, accumulator4);
            const auto totalAccumulator = _mm256_add_ps(accumulator12, accumulator34);

            const auto hiQuad = _mm256_extractf128_ps(totalAccumulator, 1);
            const auto loQuad = _mm256_castps256_ps128(totalAccumulator);
            auto sum128 = _mm_add_ps(loQuad, hiQuad);

            const auto shuf1 = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
            sum128 = _mm_add_ps(sum128, shuf1);

            const auto shuf2 = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
            sum128 = _mm_add_ps(sum128, shuf2);

            _mm_store_ss(&sum, sum128);
#endif

            for (; column < columns; ++column)
                sum += static_cast<float>(matrixRow[column]) * static_cast<float>(pInput[column]);

            pOutput[row] = static_cast<OutputType>(sum);
        }
    }

    template void LinearRange<lowp::f16, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f16*, lowp::f16*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f16, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f16*, lowp::f32*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f16, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f32*, lowp::f16*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f16, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f32*, lowp::f32*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f32, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f16*, lowp::f16*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f32, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f16*, lowp::f32*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f32, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f32*, lowp::f16*, std::size_t, std::size_t, std::size_t);
    template void LinearRange<lowp::f32, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f32*, lowp::f32*, std::size_t, std::size_t, std::size_t);

    template <class InputType, class WeightType, class OutputType>
    void RMSNorm(const InputType* pInput, const WeightType* pWeight, float epsilon, OutputType* pOutput, size_t elementCount)
    {
        if (elementCount == 0)
            return;

        float meanSquare{};

        for (std::size_t i{}; i < elementCount; ++i)
        {
            const auto input = static_cast<float>(pInput[i]);
            meanSquare += input * input;
        }

        meanSquare /= static_cast<float>(elementCount);
        const auto inverseRms = 1.0f / std::sqrt(meanSquare + epsilon);

        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        const auto inverseRmsVec = _mm256_set1_ps(inverseRms);

        for (; i + 8 <= elementCount; i += 8)
        {
            const auto xVec = Load8AsFloat(pInput + i);
            const auto weightVec = Load8AsFloat(pWeight + i);
            const auto normalized = _mm256_mul_ps(xVec, inverseRmsVec);
            const auto result = _mm256_mul_ps(weightVec, normalized);
            Store8FromFloat(pOutput + i, result);
        }
#endif

        for (; i < elementCount; ++i)
            pOutput[i] = static_cast<OutputType>(static_cast<float>(pWeight[i]) * (static_cast<float>(pInput[i]) * inverseRms));
    }

    template void RMSNorm<lowp::f16, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f16*, float, lowp::f16*, size_t);
    template void RMSNorm<lowp::f16, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f16*, float, lowp::f32*, size_t);
    template void RMSNorm<lowp::f16, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f32*, float, lowp::f16*, size_t);
    template void RMSNorm<lowp::f16, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f32*, float, lowp::f32*, size_t);
    template void RMSNorm<lowp::f32, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f16*, float, lowp::f16*, size_t);
    template void RMSNorm<lowp::f32, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f16*, float, lowp::f32*, size_t);
    template void RMSNorm<lowp::f32, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f32*, float, lowp::f16*, size_t);
    template void RMSNorm<lowp::f32, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f32*, float, lowp::f32*, size_t);

    template <class InputAType, class InputBType, class OutputType>
    void Add(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            pOutput[i] = static_cast<OutputType>(static_cast<float>(pInputA[i]) + static_cast<float>(pInputB[i]));
    }

    template void Add<lowp::f16, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f16*, lowp::f16*, size_t);
    template void Add<lowp::f16, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f16*, lowp::f32*, size_t);
    template void Add<lowp::f16, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f32*, lowp::f16*, size_t);
    template void Add<lowp::f16, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f32*, lowp::f32*, size_t);
    template void Add<lowp::f32, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f16*, lowp::f16*, size_t);
    template void Add<lowp::f32, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f16*, lowp::f32*, size_t);
    template void Add<lowp::f32, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f32*, lowp::f16*, size_t);
    template void Add<lowp::f32, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f32*, lowp::f32*, size_t);

    template <class InputAType, class InputBType, class OutputType>
    void Multiply(const InputAType* pInputA, const InputBType* pInputB, OutputType* pOutput, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            pOutput[i] = static_cast<OutputType>(static_cast<float>(pInputA[i]) * static_cast<float>(pInputB[i]));
    }

    template void Multiply<lowp::f16, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f16*, lowp::f16*, size_t);
    template void Multiply<lowp::f16, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f16*, lowp::f32*, size_t);
    template void Multiply<lowp::f16, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f32*, lowp::f16*, size_t);
    template void Multiply<lowp::f16, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f32*, lowp::f32*, size_t);
    template void Multiply<lowp::f32, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f16*, lowp::f16*, size_t);
    template void Multiply<lowp::f32, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f16*, lowp::f32*, size_t);
    template void Multiply<lowp::f32, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f32*, lowp::f16*, size_t);
    template void Multiply<lowp::f32, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f32*, lowp::f32*, size_t);

    std::size_t CheckedMultiply(size_t inputA, size_t inputB)
    {
        if (inputA != 0 && inputB > std::numeric_limits<std::size_t>::max() / inputA)
            throw std::overflow_error("Tensor size overflow");

        return inputA * inputB;
    }

    template <class InputType, class OutputType>
    void SiLU(const InputType* pInput, OutputType* pOutput, size_t elementCount)
    {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
        {
            const auto input = static_cast<float>(pInput[i]);
            pOutput[i] = static_cast<OutputType>(input / (1.0f + std::exp(-input)));
        }
    }

    template void SiLU<lowp::f16, lowp::f16>(const lowp::f16*, lowp::f16*, size_t);
    template void SiLU<lowp::f16, lowp::f32>(const lowp::f16*, lowp::f32*, size_t);
    template void SiLU<lowp::f32, lowp::f16>(const lowp::f32*, lowp::f16*, size_t);
    template void SiLU<lowp::f32, lowp::f32>(const lowp::f32*, lowp::f32*, size_t);

    template <class CosType, class SinType>
    void ComputeRoPECosSin(CosType* pSourceCos, SinType* pSourceSin, std::size_t position, std::size_t headDimension, float theta)
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
            pSourceCos[p] = static_cast<CosType>(cosAngle);
            pSourceSin[p] = static_cast<SinType>(sinAngle);
        }
    }

    template void ComputeRoPECosSin<lowp::f16, lowp::f16>(lowp::f16*, lowp::f16*, std::size_t, std::size_t, float);
    template void ComputeRoPECosSin<lowp::f16, lowp::f32>(lowp::f16*, lowp::f32*, std::size_t, std::size_t, float);
    template void ComputeRoPECosSin<lowp::f32, lowp::f16>(lowp::f32*, lowp::f16*, std::size_t, std::size_t, float);
    template void ComputeRoPECosSin<lowp::f32, lowp::f32>(lowp::f32*, lowp::f32*, std::size_t, std::size_t, float);

    template <class SourceType, class CosType, class SinType>
    void RoPE(SourceType* pSource, const CosType* pInputCos, const SinType* pInputSin, size_t headCount, size_t headDimension)
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
                const auto first = static_cast<float>(pSource[firstIndex]);
                const auto second = static_cast<float>(pSource[secondIndex]);

                const auto cosAngle = static_cast<float>(pInputCos[p]);
                const auto sinAngle = static_cast<float>(pInputSin[p]);

                pSource[firstIndex] = static_cast<SourceType>(first * cosAngle - second * sinAngle);
                pSource[secondIndex] = static_cast<SourceType>(first * sinAngle + second * cosAngle);
            }
        }
    }

    template void RoPE<lowp::f16, lowp::f16, lowp::f16>(lowp::f16*, const lowp::f16*, const lowp::f16*, size_t, size_t);
    template void RoPE<lowp::f16, lowp::f16, lowp::f32>(lowp::f16*, const lowp::f16*, const lowp::f32*, size_t, size_t);
    template void RoPE<lowp::f16, lowp::f32, lowp::f16>(lowp::f16*, const lowp::f32*, const lowp::f16*, size_t, size_t);
    template void RoPE<lowp::f16, lowp::f32, lowp::f32>(lowp::f16*, const lowp::f32*, const lowp::f32*, size_t, size_t);
    template void RoPE<lowp::f32, lowp::f16, lowp::f16>(lowp::f32*, const lowp::f16*, const lowp::f16*, size_t, size_t);
    template void RoPE<lowp::f32, lowp::f16, lowp::f32>(lowp::f32*, const lowp::f16*, const lowp::f32*, size_t, size_t);
    template void RoPE<lowp::f32, lowp::f32, lowp::f16>(lowp::f32*, const lowp::f32*, const lowp::f16*, size_t, size_t);
    template void RoPE<lowp::f32, lowp::f32, lowp::f32>(lowp::f32*, const lowp::f32*, const lowp::f32*, size_t, size_t);

    template <class QType, class KType, class VType, class OutputType>
    void Attention(const QType* pQ, const KType* pKCache, const VType* pVCache, OutputType* pOutput,
                   size_t validTokenCount, size_t attentionHeadCount, size_t keyValueHeadCount, size_t headDimension)
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
            auto currentMaximumScore = -std::numeric_limits<float>::infinity();

            float probabilitySum{};
            float oldMaximumScore{};

            for (std::size_t d{}; d < headDimension; ++d)
                pOutput[qOffset + d] = static_cast<OutputType>(0.0f);

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto kvOffset = (t * keyValueHeadCount + kvHead) * headDimension;
                float score{};

                for (std::size_t d{}; d < headDimension; ++d)
                    score += static_cast<float>(pQ[qOffset + d]) * static_cast<float>(pKCache[kvOffset + d]);

                score *= scale;
                oldMaximumScore = currentMaximumScore;
                currentMaximumScore = std::fmax(currentMaximumScore, score);

                const auto exp = std::exp(score - currentMaximumScore);
                const auto correction = std::exp(oldMaximumScore - currentMaximumScore);

                (probabilitySum *= correction) += exp;

                for (std::size_t d{}; d < headDimension; ++d)
                {
                    const auto output = static_cast<float>(pOutput[qOffset + d]);
                    pOutput[qOffset + d] = static_cast<OutputType>(output * correction + exp * static_cast<float>(pVCache[kvOffset + d]));
                }
            }

            const auto inverseProbabilitySum = 1.0f / probabilitySum;

            for (std::size_t d{}; d < headDimension; ++d)
                pOutput[qOffset + d] = static_cast<OutputType>(static_cast<float>(pOutput[qOffset + d]) * inverseProbabilitySum);
        }
    }

    template void Attention<lowp::f16, lowp::f16, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f16*, const lowp::f16*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f16, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f16*, const lowp::f16*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f16, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f16*, const lowp::f32*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f16, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f16*, const lowp::f32*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f32, lowp::f16, lowp::f16>(const lowp::f16*, const lowp::f32*, const lowp::f16*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f32, lowp::f16, lowp::f32>(const lowp::f16*, const lowp::f32*, const lowp::f16*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f32, lowp::f32, lowp::f16>(const lowp::f16*, const lowp::f32*, const lowp::f32*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f16, lowp::f32, lowp::f32, lowp::f32>(const lowp::f16*, const lowp::f32*, const lowp::f32*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f16, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f16*, const lowp::f16*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f16, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f16*, const lowp::f16*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f16, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f16*, const lowp::f32*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f16, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f16*, const lowp::f32*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f32, lowp::f16, lowp::f16>(const lowp::f32*, const lowp::f32*, const lowp::f16*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f32, lowp::f16, lowp::f32>(const lowp::f32*, const lowp::f32*, const lowp::f16*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f32, lowp::f32, lowp::f16>(const lowp::f32*, const lowp::f32*, const lowp::f32*, lowp::f16*, size_t,
                                                                        size_t, size_t, size_t);
    template void Attention<lowp::f32, lowp::f32, lowp::f32, lowp::f32>(const lowp::f32*, const lowp::f32*, const lowp::f32*, lowp::f32*, size_t,
                                                                        size_t, size_t, size_t);

    template <class SourceType, class CacheType>
    void CopyToCache(const SourceType* pSource, CacheType* pCache, size_t position, size_t elementCount)
    {
        const auto cacheOffset = position * elementCount;

        for (std::size_t i{}; i < elementCount; ++i)
            pCache[cacheOffset + i] = static_cast<CacheType>(static_cast<float>(pSource[i]));
    }

    template void CopyToCache<lowp::f16, lowp::f16>(const lowp::f16*, lowp::f16*, size_t, size_t);
    template void CopyToCache<lowp::f16, lowp::f32>(const lowp::f16*, lowp::f32*, size_t, size_t);
    template void CopyToCache<lowp::f32, lowp::f16>(const lowp::f32*, lowp::f16*, size_t, size_t);
    template void CopyToCache<lowp::f32, lowp::f32>(const lowp::f32*, lowp::f32*, size_t, size_t);
}
