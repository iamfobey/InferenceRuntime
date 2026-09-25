#pragma once

#include <cstdint>

namespace lowp
{
    class f32;

    class f16 final
    {
        using type = std::uint16_t;

    public:
        f16() noexcept = default;

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
