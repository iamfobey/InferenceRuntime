#pragma once

#include "Core/Core.hpp"

namespace Utils
{
    [[nodiscard]]
    std::size_t DataTypeSize(DataType dataType);

    [[nodiscard]]
    TensorDimVec CreateContiguousStrides(const TensorDimVec& shape);

    [[nodiscard]]
    std::size_t ElementCount(const TensorDimVec& shape) noexcept;
}
