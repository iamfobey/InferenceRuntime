#pragma once

#include "Enums.hpp"
#include "IBuffer.hpp"

#include <memory>
#include <string>
#include <vector>

#include "Math/Math.hpp"

struct Tensor
{
    std::string tensorName = "Tensor";

    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;

    std::shared_ptr<IBuffer> buffer;

    std::size_t byteOffset{0};

    DataType dataType{DataType::Float32};

    void Validate(DeviceType requiredDeviceType = DeviceType::CPU, DataType requiredDataType = DataType::Float32) const;

    [[nodiscard]]
    std::size_t ElementCount() const noexcept;

    [[nodiscard]]
    float* FloatData() const;

    [[nodiscard]]
    DeviceType Device() const noexcept;

    [[nodiscard]]
    bool IsContiguous() const noexcept;
};
