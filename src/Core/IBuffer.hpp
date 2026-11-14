#pragma once

#include "Core/Core.hpp"

#include <cstddef>

class IBuffer
{
public:
    virtual ~IBuffer() = default;

    [[nodiscard]]
    virtual DeviceType GetDeviceType() const noexcept = 0;

    [[nodiscard]]
    virtual std::size_t GetSize() const noexcept = 0;

    [[nodiscard]]
    virtual void* GetData() noexcept = 0;

    [[nodiscard]]
    virtual const void* GetData() const noexcept = 0;
};
