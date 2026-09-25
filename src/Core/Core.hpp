#pragma once

#include <cstdint>
#include "ankerl/svector.h"

#define IR_DISPATCH_CASE(CONDITIONS, ...) \
    if (CONDITIONS) { \
        dispatch.template operator()<__VA_ARGS__>(); \
        return; \
    }

namespace lowp
{
    class f32;
    class f16;
}

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

template <class T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<lowp::f16>
{
    static constexpr auto value = DataType::Float16;
};

template <>
struct DataTypeTraits<lowp::f32>
{
    static constexpr auto value = DataType::Float32;
};

template <class T>
inline constexpr DataType DataTypeOf = DataTypeTraits<std::remove_cv_t<T>>::value;

using TensorDimVec = ankerl::svector<std::size_t, 4>;
