#include "Runtime/Runtime.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Model/ITokenizer.hpp"

Runtime::Runtime(
    std::unique_ptr<IBackend> backend,
    std::unique_ptr<IModel> model
)
    : m_Backend(std::move(backend)),
      m_Model(std::move(model))
{
}

bool Runtime::LoadModel(const std::string& path) const
{
    if (!m_Backend || !m_Model)
    {
        return false;
    }

    return m_Model->Load(path, *m_Backend);
}

std::vector<std::int32_t> Runtime::Encode(const std::string_view text) const
{
    if (!m_Model || !m_Model->tokenizer)
    {
        return {};
    }

    return m_Model->tokenizer->Encode(text);
}

std::string Runtime::Decode(const std::span<const std::int32_t> tokenIds) const
{
    if (!m_Model || !m_Model->tokenizer)
    {
        return {};
    }

    return m_Model->tokenizer->Decode(tokenIds);
}

void Runtime::Prefill(const std::span<const std::int32_t> tokenIds) const
{
    if (!m_Backend || !m_Model)
    {
        return;
    }

    m_Model->Prefill(tokenIds, *m_Backend);
}

std::int32_t Runtime::GenerateNextToken(const std::int32_t currentToken) const
{
    if (!m_Backend || !m_Model)
    {
        return 0;
    }

    m_Model->DecodeStep(
        currentToken,
        *m_Backend
    );

    return SampleGreedy();
}

std::string Runtime::Generate(const std::string_view prompt, const std::size_t maximumNewTokens)
{
    if (!m_Backend || !m_Model || maximumNewTokens == 0)
        return {};

    const std::vector<std::int32_t> promptTokens = Encode(prompt);

    if (promptTokens.empty())
    {
        return {};
    }

    m_Model->Reset();

    Prefill(promptTokens);

    std::vector<std::int32_t> generatedTokens;
    generatedTokens.reserve(maximumNewTokens);

    std::int32_t currentToken = SampleGreedy();

    // for (std::size_t i = 0; i < maximumNewTokens; ++i)
    // {
    //     generatedTokens.push_back(currentToken);
    //
    //     if (i + 1 < maximumNewTokens)
    //     {
    //         currentToken =
    //             GenerateNextToken(currentToken);
    //     }
    // }

    auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < maximumNewTokens; ++i)
        currentToken = GenerateNextToken(currentToken);

    auto end = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Total: " << elapsed.count() << " us\n";
    std::cout << "Per token: " << static_cast<double>(elapsed.count()) / maximumNewTokens << " us\n";
    std::cout << "Tokens/s: " << maximumNewTokens * 1'000'000.0 / elapsed.count() << '\n';

    return Decode(generatedTokens);
}

std::string_view Runtime::ModelArchitecture() const noexcept
{
    if (!m_Model)
    {
        return {};
    }

    return m_Model->Architecture();
}

std::int32_t Runtime::SampleGreedy() const
{
    const Tensor& logits = m_Model->Logits();
    const float* data = logits.FloatData();
    std::size_t bestIndex = 0;
    for (std::size_t i = 1; i < logits.ElementCount(); ++i)
        if (data[i] > data[bestIndex])
            bestIndex = i;

    return static_cast<std::int32_t>(bestIndex);
}
