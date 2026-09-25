#pragma once

#include <cstdint>

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
