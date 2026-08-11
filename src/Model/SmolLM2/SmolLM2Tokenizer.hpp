#pragma once

#include "Model/ITokenizer.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

class IModel;

class SmolLM2Tokenizer final : public ITokenizer
{
public:
    [[nodiscard]]
    bool Load(const std::filesystem::path& path) override;

    [[nodiscard]]
    std::vector<std::int32_t> Encode(std::string_view text) const override;

    [[nodiscard]]
    std::string Decode(std::span<const std::int32_t> tokenIds) const override;

private:
    std::unordered_map<std::string, std::size_t> m_Tokens{};
    std::unordered_map<std::string, std::size_t> m_MergeRanks{};
    std::unordered_map<std::string, std::size_t> m_AddedTokens{};
};