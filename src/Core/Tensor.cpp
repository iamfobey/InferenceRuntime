#include "Tensor.hpp"

#include <stdexcept>

#include "spdlog/spdlog.h"
#include "Utils/Utils.hpp"

void Tensor::Validate(const DeviceType requiredDeviceType, const DataType requiredDataType) const
{
    if (!buffer)
        throw std::invalid_argument(tensorName + " has no buffer");

    if (buffer->Device() != requiredDeviceType)
        throw std::invalid_argument(tensorName + " is not a required device type tensor");

    if (dataType != requiredDataType)
        throw std::invalid_argument(tensorName + " has an unexpected data type");

    if (!IsContiguous())
        throw std::invalid_argument(tensorName + " must be contiguous");

    const auto alignment = dataType == DataType::Float16 ? alignof(std::uint16_t) : alignof(float);

    if (byteOffset % alignment != 0)
        throw std::invalid_argument(tensorName + " has an invalid byte offset");

    const auto requiredBytes = Math::CheckedMultiply(Utils::ElementCount(shape), Utils::DataTypeSize(dataType));

    if (byteOffset > buffer->SizeBytes() || requiredBytes > buffer->SizeBytes() - byteOffset)
    {
        spdlog::error("Tensor '{}' exceeds buffer: required {} bytes, available {} bytes", tensorName, requiredBytes,
                      buffer->SizeBytes());
        throw std::out_of_range(tensorName + " exceeds its buffer");
    }
}

float* Tensor::FloatData() const
{
    if (dataType != DataType::Float32)
        throw std::logic_error(tensorName + " is not Float32");

    return reinterpret_cast<float*>(static_cast<std::byte*>(buffer->Data()) + byteOffset);
}

std::uint16_t* Tensor::Float16Data() const
{
    if (dataType != DataType::Float16)
        throw std::logic_error(tensorName + " is not Float16");

    return reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(buffer->Data()) + byteOffset);
}

bool Tensor::IsContiguous() const
{
    return strides == Utils::CreateContiguousStrides(shape);
}
