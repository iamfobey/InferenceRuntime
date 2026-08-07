#include "SmolLM2Model.hpp"

#include <cstdint>
#include <execution>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "SmolLM2Config.hpp"
#include "Backend/IBackend.hpp"
#include "Utils/Utils.hpp"

std::string_view SmolLM2Model::Architecture() const noexcept
{
    return "smollm2";
}

namespace
{
    void CreateAndUploadTensor(IBackend& backend, Tensor& tensor, std::ifstream& file, std::uint64_t headerSize,
                               std::uint64_t startOffset, std::uint64_t endOffset,
                               const std::vector<std::size_t>& shape)
    {
        tensor = backend.CreateTensor(shape, DataType::Float32);

        std::vector<std::uint16_t> rawData{};
        rawData.resize(Utils::ElementCount(shape));

        file.seekg(8 + headerSize + startOffset);
        file.read(reinterpret_cast<char*>(rawData.data()), endOffset - startOffset);

        std::vector<float> data{};
        data.reserve(rawData.size());

        for (auto rd : rawData)
            data.emplace_back(std::bit_cast<float>(static_cast<std::uint32_t>(rd) << 16));

        backend.Upload(tensor, data);
    }
}

bool SmolLM2Model::Load(const std::filesystem::path& path, IBackend& backend)
{
    Config = SmolLM2Config::Load(path / "config.json");

    m_Layers.resize(Config.numHiddenLayers);

    if (std::ifstream file(path / "model.safetensors", std::ios::binary); file)
    {
        std::uint64_t headerSize = 0;
        file.read(reinterpret_cast<char*>(&headerSize), 8);

        std::vector<char> json_buffer(headerSize + simdjson::SIMDJSON_PADDING);
        file.read(json_buffer.data(), headerSize);

        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc;

        if (auto error = parser.iterate(json_buffer.data(), headerSize, json_buffer.size()).get(doc))
        {
            std::cerr << error << '\n';
            return false;
        }

        for (auto root = doc.get_object(); auto field : root)
        {
            std::string_view tensorName = field.unescaped_key();

            if (tensorName == "__metadata__") continue;

            simdjson::ondemand::object tensorInfoJson = field.value().get_object();

            simdjson::ondemand::array offsetsJson = tensorInfoJson["data_offsets"].get_array();
            auto offsetsIt = offsetsJson.begin();
            uint64_t startOffset = (*offsetsIt).get_uint64();
            uint64_t endOffset = (*++offsetsIt).get_uint64();

            simdjson::ondemand::array shapeJson = tensorInfoJson["shape"].get_array();
            std::vector<std::size_t> shape{};
            shape.reserve(shapeJson.count_elements());
            for (auto shapeData : shapeJson)
                shape.emplace_back(static_cast<std::size_t>(shapeData.get_uint64()));

            std::uint8_t currentLayer{};
            if (auto pos2 = tensorName.find(".", 13); pos2 != std::string::npos)
            {
                currentLayer = std::atoi(tensorName.substr(13, pos2 - 13).data());
            }

            const auto makeTensorName = [&](const char* name)
            {
                return "model.layers." + std::to_string(currentLayer) + "." + name + ".weight";
            };

            if (tensorName == "model.embed_tokens.weight")
            {
                CreateAndUploadTensor(backend, m_TokenEmbedding, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == "model.norm.weight")
            {
                CreateAndUploadTensor(backend, m_FinalNorm, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("input_layernorm"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].layernorm, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("mlp.down_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].downProj, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("mlp.gate_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].gateProj, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("mlp.up_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].upProj, file, headerSize, startOffset, endOffset,
                                      shape);
            }
            else if (tensorName == makeTensorName("post_attention_layernorm"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].postAttentionLayernorm, file, headerSize,
                                      startOffset, endOffset,
                                      shape);
            }
            else if (tensorName == makeTensorName("self_attn.k_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].selfAttnK, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.o_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].selfAttnO, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.q_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].selfAttnQ, file, headerSize, startOffset,
                                      endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.v_proj"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].selfAttnV, file, headerSize, startOffset,
                                      endOffset, shape);
            }
        }

        m_Hidden =
            backend.CreateTensor({Config.hiddenSize}, DataType::Float32);

        m_NextHidden =
            backend.CreateTensor({Config.hiddenSize}, DataType::Float32);

        m_Normalized =
            backend.CreateTensor({Config.hiddenSize}, DataType::Float32);

        m_Query = backend.CreateTensor(
            {
                Config.numAttentionHeads,
                Config.headDimension
            },
            DataType::Float32
        );

        m_Key = backend.CreateTensor(
            {
                Config.numKeyValueHeads,
                Config.headDimension
            },
            DataType::Float32
        );

        m_Value = backend.CreateTensor(
            {
                Config.numKeyValueHeads,
                Config.headDimension
            },
            DataType::Float32
        );

        m_AttentionOutput =
            backend.CreateTensor(
                {
                    Config.numAttentionHeads,
                    Config.headDimension
                },
                DataType::Float32
            );

        m_AttentionProjected =
            backend.CreateTensor({Config.hiddenSize}, DataType::Float32);

        m_Gate =
            backend.CreateTensor({Config.intermediateSize}, DataType::Float32);

        m_Up =
            backend.CreateTensor({Config.intermediateSize}, DataType::Float32);

        m_ActivatedGate =
            backend.CreateTensor({Config.intermediateSize}, DataType::Float32);

        m_FeedForward =
            backend.CreateTensor({Config.intermediateSize}, DataType::Float32);

        m_DownOutput =
            backend.CreateTensor({Config.hiddenSize}, DataType::Float32);

        m_Logits =
            backend.CreateTensor({Config.vocabSize}, DataType::Float32);

        m_KeyCaches.resize(Config.numHiddenLayers);
        m_ValueCaches.resize(Config.numHiddenLayers);

        for (std::size_t i = 0; i < Config.numHiddenLayers; ++i)
        {
            m_KeyCaches[i] = backend.CreateTensor(
                {
                    Config.maxPositionEmbeddings,
                    Config.numKeyValueHeads,
                    Config.headDimension
                },
                DataType::Float32
            );

            m_ValueCaches[i] = backend.CreateTensor(
                {
                    Config.maxPositionEmbeddings,
                    Config.numKeyValueHeads,
                    Config.headDimension
                },
                DataType::Float32
            );
        }

        if (tokenizer->Load(path / "tokenizer.json"))
        {
            return true;
        }

        return false;
    }

    std::cerr << "Failed to open model file" << '\n';

    return false;
}

void SmolLM2Model::Reset()
{
    m_Position = 0;
}

void SmolLM2Model::Prefill(
    std::span<const std::int32_t> tokenIds,
    IBackend& backend)
{
    if (tokenIds.size() >
        Config.maxPositionEmbeddings - m_Position)
    {
        throw std::out_of_range(
            "Maximum context length exceeded"
        );
    }

    for (const std::int32_t tokenId : tokenIds)
    {
        DecodeStep(tokenId, backend);
    }
}

void SmolLM2Model::DecodeStep(std::int32_t tokenId, IBackend& backend)
{
    if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= Config.vocabSize)
        throw std::out_of_range("Token id is out of vocabulary range");

    if (m_Position >= Config.maxPositionEmbeddings)
        throw std::out_of_range("Maximum context length exceeded");

    const std::span<const std::int32_t> tokenIds(&tokenId, 1);

    backend.Embedding(
        m_TokenEmbedding,
        tokenIds,
        m_Hidden
    );

    for (std::size_t layerIndex = 0; layerIndex < m_Layers.size(); ++layerIndex)
    {
        auto& [layernorm, downProj, gateProj, upProj, postAttentionLayernorm, selfAttnK, selfAttnO, selfAttnQ,
            selfAttnV] = m_Layers[
            layerIndex];

        backend.RMSNorm(
            m_Hidden,
            layernorm,
            Config.rmsNormEps,
            m_Normalized
        );

        backend.Linear(
            selfAttnQ,
            m_Normalized,
            nullptr,
            m_Query
        );

        backend.Linear(
            selfAttnK,
            m_Normalized,
            nullptr,
            m_Key
        );

        backend.Linear(
            selfAttnV,
            m_Normalized,
            nullptr,
            m_Value
        );

        backend.RoPE(
            m_Query,
            m_Position,
            Config.numAttentionHeads,
            Config.headDimension,
            Config.ropeTheta
        );

        backend.RoPE(
            m_Key,
            m_Position,
            Config.numKeyValueHeads,
            Config.headDimension,
            Config.ropeTheta
        );

        backend.CopyToCache(
            m_Key,
            m_KeyCaches[layerIndex],
            m_Position
        );

        backend.CopyToCache(
            m_Value,
            m_ValueCaches[layerIndex],
            m_Position
        );

        const std::size_t validTokenCount = m_Position + 1;

        backend.Attention(
            m_Query,
            m_KeyCaches[layerIndex],
            m_ValueCaches[layerIndex],
            validTokenCount,
            Config.numAttentionHeads,
            Config.numKeyValueHeads,
            m_AttentionOutput
        );

        backend.Linear(
            selfAttnO,
            m_AttentionOutput,
            nullptr,
            m_AttentionProjected
        );

        backend.Add(
            m_Hidden,
            m_AttentionProjected,
            m_NextHidden
        );

        std::swap(
            m_Hidden,
            m_NextHidden
        );

        backend.RMSNorm(
            m_Hidden,
            postAttentionLayernorm,
            Config.rmsNormEps,
            m_Normalized
        );

        backend.Linear(
            gateProj,
            m_Normalized,
            nullptr,
            m_Gate
        );

        backend.Linear(
            upProj,
            m_Normalized,
            nullptr,
            m_Up
        );

        backend.SiLU(
            m_Gate,
            m_ActivatedGate
        );

        backend.Multiply(
            m_ActivatedGate,
            m_Up,
            m_FeedForward
        );

        backend.Linear(
            downProj,
            m_FeedForward,
            nullptr,
            m_DownOutput
        );

        backend.Add(
            m_Hidden,
            m_DownOutput,
            m_NextHidden
        );

        std::swap(
            m_Hidden,
            m_NextHidden
        );
    }

    backend.RMSNorm(
        m_Hidden,
        m_FinalNorm,
        Config.rmsNormEps,
        m_Normalized
    );

    backend.Linear(
        m_TokenEmbedding,
        m_Normalized,
        nullptr,
        m_Logits
    );

    ++m_Position;
}

const Tensor& SmolLM2Model::Logits() const noexcept
{
    return m_Logits;
}
