#pragma once

#include <cstdint>
#include <cstring>

#include <lowp/config.hpp>
#include <lowp/types/types.hpp>

#if LOWP_HAS_F16C
#include <immintrin.h>
#endif

namespace lowp
{
#if LOWP_HAS_F16C
    inline f16::f16(f32 value) noexcept :
        f16(static_cast<float>(value)) {}
#else
    inline f16::f16(f32 value) noexcept
    {
        const auto sign = value.SignBits() >> 16;
        const auto exponent = value.ExponentValue();
        auto mantissa = value.Mantissa();

        if (exponent == 0xFFu)
        {
            if (mantissa == 0)
            {
                m_Bits = sign | 0x7C00u;
                return;
            }

            m_Bits = sign | 0x7E00u;
            return;
        }

        auto halfExponent = static_cast<std::int32_t>(exponent) - 127 + 15;

        if (halfExponent >= 31)
        {
            m_Bits = sign | 0x7C00u;
            return;
        }

        if (halfExponent <= 0)
        {
            if (halfExponent < -10)
            {
                m_Bits = sign;
                return;
            }

            mantissa |= 0x800000u;

            const auto shift = static_cast<std::uint32_t>(14 - halfExponent);
            auto halfMantissa = mantissa >> shift;
            const auto remainderMask = (1u << shift) - 1u;
            const auto remainder = mantissa & remainderMask;
            const auto halfway = 1u << (shift - 1u);

            if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u) != 0))
                ++halfMantissa;

            m_Bits = sign | halfMantissa;
            return;
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
                {
                    m_Bits = sign | 0x7C00u;
                    return;
                }
            }
        }

        m_Bits = sign | (static_cast<std::uint32_t>(halfExponent) << 10) | halfMantissa;
    }
#endif

#if LOWP_HAS_F16C
    inline f16::f16(float value) noexcept
    {
        const auto half = _mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        m_Bits = static_cast<std::uint16_t>(_mm_cvtsi128_si32(half));
    }
#else
    inline f16::f16(float value) noexcept :
        f16(f32{value}) {}
#endif

#if LOWP_HAS_F16C
    inline f16::operator float() const noexcept
    {
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(m_Bits)));
    }
#else
    inline f16::operator float() const noexcept
    {
        return static_cast<float>(f32{*this});
    }
#endif

    inline f16& f16::operator=(float value) noexcept
    {
        m_Bits = f16{value}.m_Bits;

        return *this;
    }

#if LOWP_HAS_F16C
    inline f32::f32(f16 value) noexcept :
        f32(static_cast<float>(value)) {}
#else
    inline f32::f32(f16 value) noexcept
    {
        const auto sign = static_cast<std::uint32_t>(value.SignBits()) << 16;
        const auto exponent = value.ExponentValue();
        auto mantissa = value.Mantissa();

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                m_Bits = sign;
                return;
            }

            std::int32_t normalizedExponent = -14;

            while ((mantissa & 0x0400u) == 0)
            {
                mantissa <<= 1;
                --normalizedExponent;
            }

            const auto floatExponent = static_cast<std::uint32_t>(normalizedExponent + 127);

            mantissa &= 0x03FFu;
            m_Bits = sign | (floatExponent << 23) | (mantissa << 13);
            return;
        }

        if (exponent == 0x1Fu)
        {
            m_Bits = sign | 0x7F800000u | (mantissa << 13);

            if (mantissa != 0)
                m_Bits |= 0x00400000u;

            return;
        }

        const auto floatExponent = exponent + (127u - 15u);
        m_Bits = sign | (floatExponent << 23) | (mantissa << 13);
    }
#endif

    inline f32::f32(float value) noexcept
    {
        std::memcpy(&m_Bits, &value, sizeof(value));
    }

    inline f32::operator float() const noexcept
    {
        float value;
        std::memcpy(&value, &m_Bits, sizeof(m_Bits));
        return value;
    }

    inline f32& f32::operator=(float value) noexcept
    {
        m_Bits = f32{value}.m_Bits;

        return *this;
    }
}
