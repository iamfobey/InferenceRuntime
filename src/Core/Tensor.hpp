#pragma once

#include "IBuffer.hpp"
#include "Core/Core.hpp"

#include <memory>
#include <string>
#include <vector>

struct Tensor
{
    std::string tensorName = "Tensor";

    TensorDimVec shape;
    TensorDimVec strides;

    std::shared_ptr<IBuffer> buffer;

    std::size_t byteOffset{0};

    DataType dataType{DataType::Float32};

    void Validate(DeviceType requiredDeviceType = DeviceType::CPU, DataType requiredDataType = DataType::Float32) const;

    [[nodiscard]]
    float* FloatData() const;

    [[nodiscard]]
    std::uint16_t* Float16Data() const;

    [[nodiscard]]
    Tensor View(std::size_t offset, TensorDimVec viewShape) const;

    [[nodiscard]]
    Tensor View(std::size_t offset, std::size_t count) const;
};
