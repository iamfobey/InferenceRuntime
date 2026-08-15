#pragma once

#include <stdexcept>
#include <vector>

enum class DataType : std::uint8_t;

namespace Utils
{
    [[nodiscard]]
    std::size_t DataTypeSize(DataType dataType);

    [[nodiscard]]
    std::vector<std::size_t> CreateContiguousStrides(const std::vector<std::size_t>& shape);

    [[nodiscard]]
    std::size_t ElementCount(const std::vector<std::size_t>& shape) noexcept;
}
