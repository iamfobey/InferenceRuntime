#pragma once

#include <cstdint>

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


