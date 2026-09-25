#include "Tensor.hpp"

#include <stdexcept>

#include "Math/Math.hpp"
#include "spdlog/spdlog.h"
#include "Utils/Utils.hpp"

void Tensor::Validate(DeviceType requiredDeviceType) const
{
    if (!buffer)
        throw std::invalid_argument(name + " has no buffer");

    if (buffer->Device() != requiredDeviceType)
        throw std::invalid_argument(name + " is not a required device type tensor");

    const auto alignment = dataType == DataType::Float16 ? alignof(std::uint16_t) : alignof(float);

    if (byteOffset % alignment != 0)
        throw std::invalid_argument(name + " has an invalid byte offset");

    const auto requiredBytes = Math::CheckedMultiply(Utils::ElementCount(shape), Utils::DataTypeSize(dataType));

    if (byteOffset > buffer->SizeBytes() || requiredBytes > buffer->SizeBytes() - byteOffset)
    {
        spdlog::error("Tensor '{}' exceeds buffer: required {} bytes, available {} bytes", name, requiredBytes,
                      buffer->SizeBytes());
        throw std::out_of_range(name + " exceeds its buffer");
    }
}

float* Tensor::FloatData() const
{
    if (dataType != DataType::Float32)
        throw std::logic_error(name + " is not Float32");

    return reinterpret_cast<float*>(static_cast<std::byte*>(buffer->Data()) + byteOffset);
}

std::uint16_t* Tensor::Float16Data() const
{
    if (dataType != DataType::Float16)
        throw std::logic_error(name + " is not Float16");

    return reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(buffer->Data()) + byteOffset);
}

Tensor Tensor::View(std::size_t offset, std::size_t count) const
{
    return View(offset, TensorDimVec{count});
}

Tensor Tensor::View(std::size_t offset, TensorDimVec viewShape) const
{
    const auto elementCount = Utils::ElementCount(shape);

    if (offset > elementCount || Utils::ElementCount(viewShape) > elementCount - offset)
        throw std::out_of_range("Tensor view exceeds source tensor");

    Tensor result{
        .shape = std::move(viewShape),
        .strides = Utils::CreateContiguousStrides(result.shape),
        .buffer = buffer,
        .byteOffset = byteOffset + Math::CheckedMultiply(offset, Utils::DataTypeSize(dataType)),
        .dataType = dataType,
    };

    return result;
}
