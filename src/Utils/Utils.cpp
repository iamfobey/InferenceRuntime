#include "Utils.hpp"

#include <stdexcept>

#include "Core/Enums.hpp"
#include "Math/Math.hpp"

namespace Utils
{
    std::size_t DataTypeSize(const DataType dataType)
    {
        switch (dataType)
        {
        case DataType::Float16:
            return sizeof(std::uint16_t);
        case DataType::Float32:
            return sizeof(float);
        }

        throw std::invalid_argument("Unsupported DataType");
    }

    [[nodiscard]]
    std::vector<std::size_t> CreateContiguousStrides(const std::vector<std::size_t>& shape)
    {
        std::vector<std::size_t> strides(shape.size());

        std::size_t stride = 1;

        for (std::size_t i = shape.size(); i > 0; --i)
        {
            const auto index = i - 1;

            strides[index] = stride;
            stride = Math::CheckedMultiply(stride, shape[index]);
        }

        return strides;
    }

    std::size_t ElementCount(const std::vector<std::size_t>& shape) noexcept
    {
        std::size_t count = 1;

        for (const auto dimension : shape)
            count *= dimension;

        return count;
    }
}
