#pragma once

#include "Backend/IBackend.hpp"
#include "Model/IModel.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class ITokenizer;

class Runtime
{
public:
    Runtime(std::unique_ptr<IBackend> backend, std::unique_ptr<IModel> model);

    [[nodiscard]]
    bool LoadModel(const std::string& path) const;

    [[nodiscard]]
    std::vector<std::int32_t> Encode(std::string_view text) const;

    [[nodiscard]]
    std::string Decode(std::span<const std::int32_t> tokenIds) const;

    void Prefill(std::span<const std::int32_t> tokenIds) const;

    [[nodiscard]]
    std::int32_t GenerateNextToken(std::int32_t currentToken) const;

    [[nodiscard]]
    std::string Generate(std::string_view prompt, std::size_t maximumNewTokens);

    [[nodiscard]]
    std::string_view ModelArchitecture() const noexcept;

private:
    [[nodiscard]]
    std::int32_t SampleGreedy() const;

    std::unique_ptr<IBackend> m_Backend;
    std::unique_ptr<IModel> m_Model;
};
