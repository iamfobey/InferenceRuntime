#pragma once

#include <stdexcept>
#include <vector>

#include "Core/Tensor.hpp"

enum class DataType : std::uint8_t;

namespace Utils
{
    [[nodiscard]]
    bool CPUSupportsFMA() noexcept;
    [[nodiscard]]
    bool CPUSupportsAVX2() noexcept;
    [[nodiscard]]
    bool CanRunAVX2FMAKernel() noexcept;

    [[nodiscard]]
    std::size_t DataTypeSize(DataType dataType);

    [[nodiscard]]
    std::vector<std::size_t> CreateContiguousStrides(const std::vector<std::size_t>& shape);

    [[nodiscard]]
    std::size_t ElementCount(const std::vector<std::size_t>& shape) noexcept;

    inline void RequireSameShape(const Tensor& a, const Tensor& b, const char* message)
    {
        if (a.shape != b.shape)
            throw std::invalid_argument(message);
    }
} // namespace Utils
