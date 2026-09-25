// Generated file. Edit include/lowp/* and run scripts/amalgamate.py instead.
#pragma once

// ---- lowp/lowp.hpp ----

// ---- lowp/types/types.hpp ----

// ---- lowp/types/f16.hpp ----

#include <cstdint>

namespace lowp
{
    class f32;

    class f16 final
    {
        using type = std::uint16_t;

    public:
        explicit f16(f32 value) noexcept;
        explicit f16(float value) noexcept;

        [[nodiscard]]
        explicit operator float() const noexcept;
        f16& operator=(float value) noexcept;

        [[nodiscard]]
        constexpr bool SignValue() const noexcept
        {
            return (m_Bits >> 15) & 1u;
        }

        [[nodiscard]]
        constexpr type SignBits() const noexcept
        {
            return m_Bits & 0x8000u;
        }

        [[nodiscard]]
        constexpr type ExponentValue() const noexcept
        {
            return (m_Bits >> 10) & 0x1Fu;
        }

        [[nodiscard]]
        constexpr type ExponentBits() const noexcept
        {
            return m_Bits & 0x7C00u;
        }

        [[nodiscard]]
        constexpr type Mantissa() const noexcept
        {
            return m_Bits & 0x03FFu;
        }

    private:
        type m_Bits{};
    };
}

// ---- lowp/types/f32.hpp ----


namespace lowp
{
    class f16;

    class f32 final
    {
        using type = std::uint32_t;

    public:
        explicit f32(f16 value) noexcept;
        explicit f32(float value) noexcept;

        [[nodiscard]]
        explicit operator float() const noexcept;
        f32& operator=(float value) noexcept;

        [[nodiscard]]
        constexpr bool SignValue() const noexcept
        {
            return (m_Bits >> 31) & 1u;
        }

        [[nodiscard]]
        constexpr type SignBits() const noexcept
        {
            return m_Bits & 0x80000000u;
        }

        [[nodiscard]]
        constexpr type ExponentValue() const noexcept
        {
            return (m_Bits >> 23) & 0xFFu;
        }

        [[nodiscard]]
        constexpr type ExponentBits() const noexcept
        {
            return m_Bits & 0x7F800000u;
        }

        [[nodiscard]]
        constexpr type Mantissa() const noexcept
        {
            return m_Bits & 0x007FFFFFu;
        }

    private:
        type m_Bits{};
    };
}


// ---- lowp/converters.hpp ----

#include <cstring>

namespace lowp
{
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

    inline f16::f16(float value) noexcept :
        f16(f32{value}) {}

    inline f16::operator float() const noexcept
    {
        f32 value{*this};
        return static_cast<float>(value);
    }

    inline f16& f16::operator=(float value) noexcept
    {
        m_Bits = f16{value}.m_Bits;
        return *this;
    }

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
        }
        else
        {
            const auto floatExponent = exponent + (127u - 15u);
            m_Bits = sign | (floatExponent << 23) | (mantissa << 13);
        }
    }

    inline f32::f32(float value) noexcept
    {
        // TODO: Custom allocator support
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

// ---- lowp/version.hpp ----

namespace lowp
{
    inline constexpr int version_major = 0;
    inline constexpr int version_minor = 1;
    inline constexpr int version_patch = 0;
}
