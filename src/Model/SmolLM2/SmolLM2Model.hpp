#pragma once

#include "Model/IModel.hpp"
#include "SmolLM2Config.hpp"
#include "Core/Tensor.hpp"

#include <span>
#include <string_view>
#include <vector>

class SmolLM2Model final : public IModel
{
public:
    [[nodiscard]]
    std::string_view Architecture() const noexcept override;

    bool Load(const std::filesystem::path& path, IBackend& backend) override;

    void Reset() override;

    void Prefill(std::span<const std::int32_t> tokenIds, IBackend& backend) override;

    void DecodeStep(std::int32_t tokenId, IBackend& backend) override;

    [[nodiscard]]
    const Tensor& Logits() const noexcept override;

    SmolLM2Config Config;

private:
    struct SmolLM2Layer
    {
        Tensor layernorm;
        Tensor downProj;
        Tensor gateProj;
        Tensor upProj;
        Tensor postAttentionLayernorm;
        Tensor selfAttnK;
        Tensor selfAttnO;
        Tensor selfAttnQ;
        Tensor selfAttnV;
    };

    Tensor m_TokenEmbedding;
    std::vector<SmolLM2Layer> m_Layers;
    Tensor m_FinalNorm;

    Tensor m_Hidden;
    Tensor m_NextHidden;
    Tensor m_Normalized;

    Tensor m_Query;
    Tensor m_Key;
    Tensor m_Value;

    Tensor m_RopeCos;
    Tensor m_RopeSin;

    Tensor m_AttentionScores;
    Tensor m_AttentionOutput;
    Tensor m_AttentionProjected;

    Tensor m_Gate;
    Tensor m_Up;
    Tensor m_ActivatedGate;
    Tensor m_FeedForward;
    Tensor m_DownOutput;

    std::vector<Tensor> m_KeyCaches;
    std::vector<Tensor> m_ValueCaches;

    Tensor m_Logits;

    std::size_t m_Position{};
};
