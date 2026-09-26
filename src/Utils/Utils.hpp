#pragma once

#include "Core/Core.hpp"

namespace Utils
{
    [[nodiscard]]
    inline bool IsZen3() noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_is("znver3");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int registers[4]{};
        __cpuid(registers, 1);

        const auto eax = static_cast<std::uint32_t>(registers[0]);
        auto family = (eax >> 8) & 0x0Fu;
        auto model = (eax >> 4) & 0x0Fu;

        if (family == 0x0Fu)
            family += (eax >> 20) & 0xFFu;

        if (family == 0x06u || family == 0x0Fu)
            model += ((eax >> 16) & 0x0Fu) << 4;

        return family == 0x19u && (model <= 0x0Fu || (model >= 0x20u && model <= 0x5Fu));
#else
        return false;
#endif
    }

    [[nodiscard]]
    std::size_t DataTypeSize(DataType dataType);

    [[nodiscard]]
    TensorDimVec CreateContiguousStrides(const TensorDimVec& shape);

    [[nodiscard]]
    std::size_t ElementCount(const TensorDimVec& shape) noexcept;
}
