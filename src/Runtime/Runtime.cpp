#include "Runtime/Runtime.hpp"

#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Model/ITokenizer.hpp"

Runtime::Runtime(std::unique_ptr<IBackend> backend, std::unique_ptr<IModel> model) : m_Backend(std::move(backend)),
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

    constexpr std::size_t tot = 5;
    std::chrono::microseconds total{};

    for (std::size_t j{}; j < tot; ++j)
    {
        m_Model->Reset();
        Prefill(promptTokens);

        std::int32_t currentToken = SampleGreedy();

        const auto start = std::chrono::steady_clock::now();

        for (std::size_t i{}; i < maximumNewTokens; ++i)
            currentToken = GenerateNextToken(currentToken);

        const auto end = std::chrono::steady_clock::now();

        total += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    const double averageUs = static_cast<double>(total.count()) / tot;

    std::cout << "Per token: " << averageUs / maximumNewTokens << " us\n";
    std::cout << "Tokens/s: " << maximumNewTokens * 1'000'000.0 / averageUs << '\n';

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

    return Decode({});
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
