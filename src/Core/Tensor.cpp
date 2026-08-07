#include "Tensor.hpp"

#include <iostream>
#include <ostream>
#include <stdexcept>

#include "Utils/Utils.hpp"

void Tensor::Validate(const DeviceType requiredDeviceType, const DataType requiredDataType) const
{
    if (!buffer)
    {
        throw std::invalid_argument(tensorName + " has no buffer");
    }

    if (buffer->Device() != requiredDeviceType)
    {
        throw std::invalid_argument(tensorName + " is not a CPU tensor");
    }

    if (dataType != requiredDataType)
    {
        throw std::invalid_argument(tensorName + " must use Float32");
    }

    if (!IsContiguous())
    {
        throw std::invalid_argument(tensorName + " must be contiguous");
    }

    if (byteOffset % alignof(float) != 0)
    {
        throw std::invalid_argument(tensorName + " has an invalid byte offset");
    }

    const std::size_t requiredBytes = Math::CheckedMultiply(ElementCount(), Utils::DataTypeSize(dataType));

    if (byteOffset > buffer->SizeBytes() || requiredBytes > buffer->SizeBytes() - byteOffset)
    {
        std::cout << requiredBytes << ' ' << buffer->SizeBytes() << std::endl;
        throw std::out_of_range(tensorName + " exceeds its buffer");
    }
}

float* Tensor::FloatData() const
{
    return reinterpret_cast<float*>(static_cast<std::byte*>(buffer->Data()) + byteOffset);
}

DeviceType Tensor::Device() const noexcept
{
    return buffer->Device();
}

std::size_t Tensor::ElementCount() const noexcept
{
    return Utils::ElementCount(shape);
}

bool Tensor::IsContiguous() const noexcept
{
    return strides == Utils::CreateContiguousStrides(shape);
}
