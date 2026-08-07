#pragma once

#include "Enums.hpp"

#include <cstddef>

class IBuffer
{
public:
    virtual ~IBuffer() = default;

    [[nodiscard]]
    virtual DeviceType Device() const noexcept = 0;

    [[nodiscard]]
    virtual std::size_t SizeBytes() const noexcept = 0;

    [[nodiscard]]
    virtual void* Data() noexcept = 0;

    [[nodiscard]]
    virtual const void* Data() const noexcept = 0;
};
