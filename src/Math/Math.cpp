#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "Utils/Utils.hpp"

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#include "Math/Math.hpp"
#include "lowp/lowp.hpp"

namespace
{
#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
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

        return _mm256_setzero_ps();
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

    template <class Function>
    void DispatchAccumulatorCount(Function&& function)
    {
        static const bool zen3 = Utils::IsZen3();

        if (zen3)
        {
            function.template operator()<8>();
            return;
        }

        function.template operator()<4>();
    }

    // TODO: naive realization of std::exp -> normal
    __m256 exp256_ps(__m256 x)
    {
        const auto inv_ln2 = _mm256_set1_ps(1.4426950408889634074f); // 1/ln(2)
        const auto ln2_hi = _mm256_set1_ps(-0.693145751953125f); // major ln(2)
        const auto ln2_lo = _mm256_set1_ps(-1.428606820309417232e-7f); // minor ln(2)

        // (^5) for e^r on [-ln(2)/2, ln(2)/2]
        const auto c0 = _mm256_set1_ps(1.0f);
        const auto c1 = _mm256_set1_ps(1.0f);
        const auto c2 = _mm256_set1_ps(0.5f);
        const auto c3 = _mm256_set1_ps(0.1666666716337204f);
        const auto c4 = _mm256_set1_ps(0.0416664853692055f);
        const auto c5 = _mm256_set1_ps(0.0083333607763052f);

        // limits
        const auto max_exp = _mm256_set1_ps(88.3762626647949f);
        const auto min_exp = _mm256_set1_ps(-88.3762626647949f);
        x = _mm256_min_ps(x, max_exp);
        x = _mm256_max_ps(x, min_exp);

        // n = round(x / ln(2))
        const auto fx = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const auto n = _mm256_cvtps_epi32(fx);

        // r = x - n * ln(2)
        __m256 r = _mm256_fmadd_ps(fx, ln2_hi, x);
        r = _mm256_fmadd_ps(fx, ln2_lo, r);

        // P(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
        __m256 p = c5;
        p = _mm256_fmadd_ps(p, r, c4);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, c1);
        p = _mm256_fmadd_ps(p, r, c0);

        // 2^n IEEE 754 float
        const auto bias = _mm256_set1_epi32(127);
        const auto twon_bits = _mm256_slli_epi32(_mm256_add_epi32(n, bias), 23);
        const auto twon = _mm256_castsi256_ps(twon_bits);

        // e^x = P(r) * 2^n
        return _mm256_mul_ps(p, twon);
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

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
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
#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
        DispatchAccumulatorCount([&]<std::size_t AccumulatorCount>
        {
            constexpr std::size_t vectorWidth = 8;
            constexpr std::size_t blockSize = AccumulatorCount * vectorWidth;

            for (std::size_t row = beginRow; row < endRow; ++row)
            {
                const auto* matrixRow = pMatrix + row * columns;
                std::size_t column{};
                float sum{};
                __m256 accumulators[AccumulatorCount];

                for (auto& accumulator : accumulators)
                    accumulator = _mm256_setzero_ps();

                for (; column + blockSize <= columns; column += blockSize)
                {
                    for (std::size_t accumulatorIndex{}; accumulatorIndex < AccumulatorCount; ++accumulatorIndex)
                    {
                        const auto offset = column + accumulatorIndex * vectorWidth;
                        const auto input = Load8AsFloat(pInput + offset);
                        const auto matrix = Load8AsFloat(matrixRow + offset);
                        accumulators[accumulatorIndex] = _mm256_fmadd_ps(matrix, input, accumulators[accumulatorIndex]);
                    }
                }

                auto totalAccumulator = accumulators[0];

                for (std::size_t accumulatorIndex = 1; accumulatorIndex < AccumulatorCount; ++accumulatorIndex)
                    totalAccumulator = _mm256_add_ps(totalAccumulator, accumulators[accumulatorIndex]);

                const auto hiQuad = _mm256_extractf128_ps(totalAccumulator, 1);
                const auto loQuad = _mm256_castps256_ps128(totalAccumulator);
                auto sum128 = _mm_add_ps(loQuad, hiQuad);

                const auto shuf1 = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
                sum128 = _mm_add_ps(sum128, shuf1);

                const auto shuf2 = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
                sum128 = _mm_add_ps(sum128, shuf2);

                _mm_store_ss(&sum, sum128);

                for (; column < columns; ++column)
                    sum += static_cast<float>(matrixRow[column]) * static_cast<float>(pInput[column]);

                pOutput[row] = static_cast<OutputType>(sum);
            }
        });
#else
        for (std::size_t row = beginRow; row < endRow; ++row)
        {
            const auto* matrixRow = pMatrix + row * columns;
            float sum{};

            for (std::size_t column{}; column < columns; ++column)
                sum += static_cast<float>(matrixRow[column]) * static_cast<float>(pInput[column]);

            pOutput[row] = static_cast<OutputType>(sum);
        }
#endif
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

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
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
        std::size_t i{};

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
        DispatchAccumulatorCount([&]<std::size_t AccumulatorCount>
        {
            constexpr std::size_t vectorWidth = 8;
            constexpr std::size_t blockSize = AccumulatorCount * vectorWidth;

            for (; i + blockSize <= elementCount; i += blockSize)
            {
                for (std::size_t accumulatorIndex{}; accumulatorIndex < AccumulatorCount; ++accumulatorIndex)
                {
                    const auto offset = i + accumulatorIndex * vectorWidth;
                    Store8FromFloat(pOutput + offset, _mm256_add_ps(Load8AsFloat(pInputA + offset), Load8AsFloat(pInputB + offset)));
                }
            }
        });
#endif

        for (; i < elementCount; ++i)
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
        std::size_t i{};

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
        DispatchAccumulatorCount([&]<std::size_t AccumulatorCount>
        {
            constexpr std::size_t vectorWidth = 8;
            constexpr std::size_t blockSize = AccumulatorCount * vectorWidth;

            for (; i + blockSize <= elementCount; i += blockSize)
            {
                for (std::size_t accumulatorIndex{}; accumulatorIndex < AccumulatorCount; ++accumulatorIndex)
                {
                    const auto offset = i + accumulatorIndex * vectorWidth;
                    Store8FromFloat(pOutput + offset, _mm256_mul_ps(Load8AsFloat(pInputA + offset), Load8AsFloat(pInputB + offset)));
                }
            }
        });
#endif

        for (; i < elementCount; ++i)
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
        std::size_t i{};

#if defined(__AVX2__) && (defined(_MSC_VER) || (defined(__FMA__) && defined(__F16C__)))
        const auto one = _mm256_set1_ps(1.0f);
        const auto zero = _mm256_setzero_ps();

        DispatchAccumulatorCount([&]<std::size_t AccumulatorCount>
        {
            constexpr std::size_t vectorWidth = 8;
            constexpr std::size_t blockSize = AccumulatorCount * vectorWidth;

            for (; i + blockSize <= elementCount; i += blockSize)
            {
                for (std::size_t accumulatorIndex{}; accumulatorIndex < AccumulatorCount; ++accumulatorIndex)
                {
                    const auto offset = i + accumulatorIndex * vectorWidth;
                    const auto input = Load8AsFloat(pInput + offset);
                    const auto exp = exp256_ps(_mm256_sub_ps(zero, input));
                    Store8FromFloat(pOutput + offset, _mm256_div_ps(input, _mm256_add_ps(one, exp)));
                }
            }
        });
#endif

        for (; i < elementCount; ++i)
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
