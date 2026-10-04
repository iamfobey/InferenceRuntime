#pragma once

#include <cstddef>

#include <lowp/config.hpp>
#include <lowp/types/types.hpp>

#if LOWP_HAS_F16C
#include <immintrin.h>

namespace lowp::simd
{
    inline constexpr std::size_t Width = 8;

    [[nodiscard]]
    inline __m256 Load8AsFloat(const float* source) noexcept
    {
        return _mm256_loadu_ps(source);
    }

    [[nodiscard]]
    inline __m256 Load8AsFloat(const f16* source) noexcept
    {
        const auto half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));

        return _mm256_cvtph_ps(half);
    }

    [[nodiscard]]
    inline __m256 Load8AsFloat(const f32* source) noexcept
    {
        const auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));

        return _mm256_castsi256_ps(bits);
    }

    inline void Store8FromFloat(float* destination, const __m256 value) noexcept
    {
        _mm256_storeu_ps(destination, value);
    }

    inline void Store8FromFloat(f16* destination, const __m256 value) noexcept
    {
        const auto half = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), half);
    }

    inline void Store8FromFloat(f32* destination, const __m256 value) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), _mm256_castps_si256(value));
    }
}
#endif
