#pragma once

#include "IBuffer.hpp"
#include "Core/Core.hpp"

#include <memory>
#include <string>
#include <vector>

struct Tensor
{
    std::string name = "Tensor";

    TensorDimVec shape;
    TensorDimVec strides;

    std::shared_ptr<IBuffer> buffer;

    std::size_t byteOffset{0};

    DataType dataType{DataType::Float32};

    void Validate(DeviceType requiredDeviceType = DeviceType::CPU) const;

    [[nodiscard]]
    float* FloatData() const;

    [[nodiscard]]
    std::uint16_t* Float16Data() const;

    template <class T>
    [[nodiscard]]
    T* Data()
    {
        static_assert(!std::is_const_v<T>);

        if (dataType != DataTypeOf<T>)
            throw std::logic_error(name + " has an unexpected data type");

        auto* data = static_cast<std::byte*>(buffer->Data()) + byteOffset;

        return reinterpret_cast<T*>(data);
    }

    template <class T>
    [[nodiscard]]
    const T* Data() const
    {
        static_assert(!std::is_const_v<T>);

        if (dataType != DataTypeOf<T>)
            throw std::logic_error(name + " has an unexpected data type");

        const auto* data = static_cast<const std::byte*>(buffer->Data()) + byteOffset;

        return reinterpret_cast<const T*>(data);
    }

    [[nodiscard]]
    Tensor View(std::size_t offset, TensorDimVec viewShape) const;

    [[nodiscard]]
    Tensor View(std::size_t offset, std::size_t count) const;
};
