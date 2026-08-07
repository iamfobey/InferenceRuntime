#pragma once

#include <memory>
#include <string_view>

class IModel;

class ModelFactory final
{
public:
    [[nodiscard]]
    static std::unique_ptr<IModel> Create(std::string_view architecture);
};
