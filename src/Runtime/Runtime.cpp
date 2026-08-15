#include "Runtime/Runtime.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <chrono>
#include <utility>
#include <vector>

#include "Model/ITokenizer.hpp"
#include "Utils/Utils.hpp"
#include "spdlog/spdlog.h"

Runtime::Runtime(std::unique_ptr<IBackend> backend, std::unique_ptr<IModel> model) : m_Backend(std::move(backend)),
    m_Model(std::move(model))
{
    spdlog::info("[runtime] created: backend={}, model={}", m_Backend ? "available" : "missing",
                 m_Model ? m_Model->Architecture() : "missing");
}

bool Runtime::LoadModel(const std::string& path)
{
    if (!m_Backend || !m_Model)
    {
        spdlog::error("[runtime] cannot load model: backend or model is missing");
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    spdlog::info("[runtime] loading {} model from {}", m_Model->Architecture(), path);
    const bool loaded = m_Model->Load(path, *m_Backend);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    spdlog::info("[runtime] model load {}: {:.2f} s", loaded ? "completed" : "failed", elapsed);
    return loaded;
}

std::vector<std::int32_t> Runtime::Encode(std::string_view text)
{
    if (!m_Model || !m_Model->tokenizer)
    {
        spdlog::error("[runtime] encode requested without a loaded tokenizer");
        return {};
    }

    return m_Model->tokenizer->Encode(text);
}

std::string Runtime::Decode(std::span<const int32_t> tokenIds)
{
    if (!m_Model || !m_Model->tokenizer)
    {
        spdlog::error("[runtime] decode requested without a loaded tokenizer");
        return {};
    }

    return m_Model->tokenizer->Decode(tokenIds);
}

void Runtime::Prefill(std::span<const int32_t> tokenIds)
{
    if (!m_Backend || !m_Model)
    {
        spdlog::error("[runtime] prefill requested without backend or model");
        return;
    }

    m_Model->Prefill(tokenIds, *m_Backend);
}

std::int32_t Runtime::GenerateNextToken(int32_t currentToken)
{
    if (!m_Backend || !m_Model)
    {
        spdlog::error("[runtime] decode requested without backend or model");
        return 0;
    }

    m_Model->DecodeStep(
        currentToken,
        *m_Backend
    );

    return SampleGreedy();
}

std::string Runtime::Generate(std::string_view prompt, size_t maximumNewTokens)
{
    if (!m_Backend || !m_Model || maximumNewTokens == 0)
    {
        spdlog::warn("[runtime] generation skipped: backend/model unavailable or token limit is zero");
        return {};
    }

    const auto totalStart = std::chrono::steady_clock::now();
    const auto tokenizationStart = totalStart;
    const std::vector<std::int32_t> promptTokens = Encode(prompt);
    const auto tokenizationTime = std::chrono::steady_clock::now() - tokenizationStart;

    if (promptTokens.empty())
    {
        spdlog::warn("[runtime] generation skipped: prompt encoded to zero tokens");
        return {};
    }

    m_Model->Reset();
    const auto prefillStart = std::chrono::steady_clock::now();
    Prefill(promptTokens);
    const auto prefillTime = std::chrono::steady_clock::now() - prefillStart;

    std::vector<std::int32_t> generatedTokens;
    generatedTokens.reserve(maximumNewTokens);

    auto samplingTime = std::chrono::steady_clock::duration::zero();
    auto samplingStart = std::chrono::steady_clock::now();
    std::int32_t currentToken = SampleGreedy();
    samplingTime += std::chrono::steady_clock::now() - samplingStart;

    auto generationTime = std::chrono::steady_clock::duration::zero();
    for (std::size_t i{}; i < maximumNewTokens; ++i)
    {
        generatedTokens.push_back(currentToken);

        if (i + 1 < maximumNewTokens)
        {
            const auto decodeStart = std::chrono::steady_clock::now();
            m_Model->DecodeStep(currentToken, *m_Backend);
            generationTime += std::chrono::steady_clock::now() - decodeStart;
            samplingStart = std::chrono::steady_clock::now();
            currentToken = SampleGreedy();
            samplingTime += std::chrono::steady_clock::now() - samplingStart;
        }
    }

    const auto decoded = Decode(generatedTokens);
    const auto millis = [](const auto duration) { return std::chrono::duration<double, std::milli>(duration).count(); };
    const auto seconds = [](const auto duration) { return std::chrono::duration<double>(duration).count(); };
    spdlog::info("[runtime] prompt eval: {} tokens, {:.2f} ms, {:.2f} tok/s", promptTokens.size(), millis(prefillTime),
                 promptTokens.size() / seconds(prefillTime));
    spdlog::info("[runtime] generation: {} tokens, {:.2f} ms, {:.2f} tok/s", generatedTokens.size(),
                 millis(generationTime), generatedTokens.size() / seconds(generationTime));
    spdlog::info("[runtime] sampling: {} calls, {:.2f} ms, {:.2f} tok/s", generatedTokens.size(),
                 millis(samplingTime), generatedTokens.size() / seconds(samplingTime));
    spdlog::info("[runtime] tokenization: {:.2f} ms; output: {} bytes", millis(tokenizationTime), decoded.size());
    spdlog::info("[runtime] total generation request: {:.2f} ms",
                 millis(std::chrono::steady_clock::now() - totalStart));
    return decoded;
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
    const auto& logits = m_Model->Logits();
    const auto* pData = logits.FloatData();

    std::size_t bestIndex{};

    for (std::size_t i = 1; i < Utils::ElementCount(logits.shape); ++i)
        if (pData[i] > pData[bestIndex])
            bestIndex = i;

    return static_cast<std::int32_t>(bestIndex);
}
