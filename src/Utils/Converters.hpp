#pragma once

#include <cstddef>
#include <cstdint>

namespace Utils::Converters
{
    [[nodiscard]]
    std::uint16_t Float32ToFloat16(float value) noexcept;

    [[nodiscard]]
    float Float16ToFloat32(uint16_t value) noexcept;

    void ConvertFloat32ToFloat16(const float* pSrc, std::uint16_t* pDest, size_t elementCount);

    void ConvertFloat16ToFloat32(const std::uint16_t* pSrc, float* pDesc, size_t elementCount);
}
