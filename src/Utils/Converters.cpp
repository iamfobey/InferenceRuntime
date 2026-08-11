#include "Converters.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#if HAVE_AVX2_SUPPORT
#include <immintrin.h>
#endif

namespace Utils::Converters
{
    std::uint16_t Float32ToFloat16(float value) noexcept
    {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        const auto sign = (bits >> 16) & 0x8000u;
        const auto exponent = (bits >> 23) & 0xFFu;
        auto mantissa = bits & 0x7FFFFFu;

        if (exponent == 0xFFu)
        {
            if (mantissa == 0)
                return sign | 0x7C00u;

            return sign | 0x7E00u;
        }

        auto halfExponent = static_cast<std::int32_t>(exponent) - 127 + 15;

        if (halfExponent >= 31)
            return sign | 0x7C00u;

        if (halfExponent <= 0)
        {
            if (halfExponent < -10)
                return sign;

            mantissa |= 0x800000u;

            const auto shift = static_cast<std::uint32_t>(14 - halfExponent);
            auto halfMantissa = mantissa >> shift;
            const auto remainderMask = (1u << shift) - 1u;
            const auto remainder = mantissa & remainderMask;
            const auto halfway = 1u << (shift - 1u);

            if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u) != 0))
                ++halfMantissa;

            return sign | halfMantissa;
        }

        auto halfMantissa = mantissa >> 13;
        const auto remainder = mantissa & 0x1FFFu;

        if (remainder > 0x1000u || (remainder == 0x1000u && (halfMantissa & 1u) != 0))
        {
            ++halfMantissa;

            if (halfMantissa == 0x400u)
            {
                halfMantissa = 0;
                ++halfExponent;

                if (halfExponent >= 31)
                    return sign | 0x7C00u;
            }
        }

        return sign | (static_cast<std::uint32_t>(halfExponent) << 10) | halfMantissa;
    }

    float Float16ToFloat32(const std::uint16_t value) noexcept
    {
        const auto sign = (value & 0x8000u) << 16;
        const auto exponent = (value >> 10) & 0x1Fu;
        auto mantissa = value & 0x03FFu;
        std::uint32_t bits{};

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                std::int32_t normalizedExponent = -14;

                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    --normalizedExponent;
                }

                const auto floatExponent = static_cast<std::uint32_t>(normalizedExponent + 127);

                mantissa &= 0x03FFu;
                bits = sign | (floatExponent << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mantissa << 13);

            if (mantissa != 0)
                bits |= 0x00400000u;
        }
        else
        {
            const auto floatExponent = exponent + (127u - 15u);
            bits = sign | (floatExponent << 23) | (mantissa << 13);
        }

        return std::bit_cast<float>(bits);
    }

    void ConvertFloat32ToFloat16(const float* source, std::uint16_t* destination, const std::size_t elementCount)
    {
        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        for (; i + 8 <= elementCount; i += 8)
        {
            const auto values = _mm256_loadu_ps(source + i);
            const auto halfValues = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + i), halfValues);
        }
#endif

        for (; i < elementCount; ++i)
            destination[i] = Float32ToFloat16(source[i]);
    }

    void ConvertFloat16ToFloat32(const std::uint16_t* source, float* destination, const std::size_t elementCount)
    {
        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        for (; i + 8 <= elementCount; i += 8)
        {
            const auto halfValues = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
            const auto values = _mm256_cvtph_ps(halfValues);
            _mm256_storeu_ps(destination + i, values);
        }
#endif

        for (; i < elementCount; ++i)
            destination[i] = Float16ToFloat32(source[i]);
    }
}
