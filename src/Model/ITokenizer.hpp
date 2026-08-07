#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class IModel;

class ITokenizer
{
public:
    virtual ~ITokenizer() = default;

    [[nodiscard]]
    virtual bool Load(const std::filesystem::path& path) = 0;

    [[nodiscard]]
    virtual std::vector<std::int32_t> Encode(std::string_view text) const = 0;

    [[nodiscard]]
    virtual std::string Decode(std::span<const std::int32_t> tokenIds) const = 0;
};
