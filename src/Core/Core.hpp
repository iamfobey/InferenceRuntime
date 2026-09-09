#pragma once

#include <cstdint>
#include "ankerl/svector.h"

enum class DeviceType : std::uint8_t
{
    CPU,
    CUDA
};

enum class DataType : std::uint8_t
{
    Float16,
    Float32
};

using TensorDimVec = ankerl::svector<std::size_t, 4>;
