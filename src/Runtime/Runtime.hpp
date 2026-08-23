#pragma once

#include "Backend/IBackend.hpp"
#include "Model/IModel.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Backend/CPU/Backend.hpp"

class ITokenizer;

struct RuntimeOptions
{
    std::string_view modelArchitecture;
    std::string backendDriver;
    CpuBackendOptions cpuBackendOptions;
};

class Runtime
{
public:
    Runtime(const RuntimeOptions& options);

    [[nodiscard]]
    bool LoadModel(const std::string& path);

    [[nodiscard]]
    std::vector<std::int32_t> Encode(std::string_view text);

    [[nodiscard]]
    std::string Decode(std::span<const int32_t> tokenIds);

    void Prefill(std::span<const int32_t> tokenIds);

    [[nodiscard]]
    std::int32_t GenerateNextToken(int32_t currentToken);

    [[nodiscard]]
    std::string Generate(std::string_view prompt, size_t maximumNewTokens);

    [[nodiscard]]
    std::string_view ModelArchitecture() const noexcept;

    [[nodiscard]]
    const std::unique_ptr<IBackend>& GetBackend() const noexcept;

    [[nodiscard]]
    const std::unique_ptr<IModel>& GetModel() const noexcept;

private:
    [[nodiscard]]
    std::int32_t SampleGreedy() const;

    std::unique_ptr<IBackend> m_Backend;
    std::unique_ptr<IModel> m_Model;
};
