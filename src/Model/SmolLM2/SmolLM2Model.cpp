#include "SmolLM2Model.hpp"

#include <bit>
#include <execution>
#include <fstream>
#include <chrono>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "SmolLM2Config.hpp"
#include "Backend/IBackend.hpp"
#include "Utils/Utils.hpp"
#include "Model/ITokenizer.hpp"
#include "spdlog/spdlog.h"

std::string_view SmolLM2Model::Architecture() const noexcept
{
    return "smollm2";
}

namespace
{
    std::string ShapeString(const TensorDimVec& shape)
    {
        std::string result{"["};
        for (std::size_t index{}; index < shape.size(); ++index)
        {
            if (index != 0)
                result += ", ";
            result += std::to_string(shape[index]);
        }
        return result + ']';
    }

    void UploadTensor(IBackend& backend, Tensor& tensor, std::ifstream& file, std::uint64_t headerSize,
                      std::uint64_t startOffset, std::uint64_t endOffset,
                      const TensorDimVec& shape)
    {
        std::vector<std::uint16_t> rawData(Utils::ElementCount(shape));

        file.seekg(static_cast<std::streamoff>(8 + headerSize + startOffset));
        file.read(reinterpret_cast<char*>(rawData.data()), static_cast<std::streamsize>(endOffset - startOffset));

        std::vector<float> data(rawData.size());

        for (std::size_t i{}; i < data.size(); ++i)
            data[i] = std::bit_cast<float>(static_cast<std::uint32_t>(rawData[i]) << 16);

        backend.Upload(tensor, data);
    }

    void CreateAndUploadTensor(IBackend& backend, Tensor& tensor, std::ifstream& file, std::uint64_t headerSize,
                               std::uint64_t startOffset, std::uint64_t endOffset,
                               const TensorDimVec& shape)
    {
        tensor = backend.CreateTensor(shape, DataType::Float16);

        UploadTensor(backend, tensor, file, headerSize, startOffset, endOffset, shape);
    }
}

bool SmolLM2Model::Load(const std::filesystem::path& path, IBackend& backend)
{
    const auto loadStart = std::chrono::steady_clock::now();
    Config = SmolLM2Config::Load(path / "config.json");
    spdlog::info(
        "[model] SmolLM2: dtype={}, vocab={}, layers={}, hidden={}, intermediate={}, heads={}/{}, head dim={}, m_Query={}, KV={}, context={}, rope theta={}, rms eps={}, tied embeddings={}",
        Config.torchDtype, Config.vocabSize, Config.numHiddenLayers, Config.hiddenSize, Config.intermediateSize,
        Config.numAttentionHeads, Config.numKeyValueHeads, Config.headDimension, Config.querySize,
        Config.keyValueSize, Config.maxPositionEmbeddings, Config.ropeTheta, Config.rmsNormEps,
        Config.tieWordEmbeddings);

    m_Layers.resize(Config.numHiddenLayers);
    for (auto& layer : m_Layers)
    {
        layer.selfAttnQKV = backend.CreateTensor({Config.querySize + 2 * Config.keyValueSize, Config.hiddenSize},
                                                 DataType::Float16);
        layer.gateUpProj = backend.CreateTensor({Config.intermediateSize * 2, Config.hiddenSize}, DataType::Float16);
    }

    if (std::ifstream file(path / "model.safetensors", std::ios::binary); file)
    {
        const auto modelFile = path / "model.safetensors";
        spdlog::info("[model] loading weights: {} ({:.2f} MiB)", modelFile.string(),
                     static_cast<double>(std::filesystem::file_size(modelFile)) / (1024.0 * 1024.0));
        std::uint64_t headerSize{};
        file.read(reinterpret_cast<char*>(&headerSize), 8);

        std::vector<char> json_buffer(headerSize + simdjson::SIMDJSON_PADDING);
        file.read(json_buffer.data(), static_cast<std::streamsize>(headerSize));

        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc;

        if (auto error = parser.iterate(json_buffer.data(), headerSize, json_buffer.size()).get(doc))
        {
            spdlog::error("[model] invalid safetensors header: {}", simdjson::error_message(error));
            return false;
        }

        std::size_t tensorCount{};
        std::uint64_t weightBytes{};

        for (auto root = doc.get_object(); auto field : root)
        {
            const std::string_view tensorName = field.unescaped_key();

            if (tensorName == "__metadata__") continue;

            ++tensorCount;

            auto tensorInfoJson = field.value().get_object();

            auto offsetsJson = tensorInfoJson["data_offsets"].get_array();
            auto offsetsIt = offsetsJson.begin();
            const std::uint64_t startOffset = (*offsetsIt).get_uint64();
            const std::uint64_t endOffset = (*++offsetsIt).get_uint64();
            weightBytes += endOffset - startOffset;

            auto shapeJson = tensorInfoJson["shape"].get_array();

            TensorDimVec shape{};
            shape.reserve(shapeJson.count_elements());
            for (auto shapeData : shapeJson)
                shape.emplace_back(static_cast<std::size_t>(shapeData.get_uint64()));

            spdlog::debug("[model] tensor: {} shape={} bytes={}", tensorName, ShapeString(shape),
                          endOffset - startOffset);

            std::uint8_t currentLayer{};
            if (auto pos2 = tensorName.find('.', 13); pos2 != std::string::npos)
            {
                currentLayer = std::atoi(tensorName.substr(13, pos2 - 13).data()); // NOLINT(*-err34-c)
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
                auto gate = m_Layers[currentLayer].gateUpProj.View(0, {Config.intermediateSize, Config.hiddenSize});
                UploadTensor(backend, gate, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("mlp.up_proj"))
            {
                auto up = m_Layers[currentLayer].gateUpProj.View(Config.intermediateSize * Config.hiddenSize,
                                                                 {Config.intermediateSize, Config.hiddenSize});
                UploadTensor(backend, up, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("post_attention_layernorm"))
            {
                CreateAndUploadTensor(backend, m_Layers[currentLayer].postAttentionLayernorm, file, headerSize,
                                      startOffset, endOffset,
                                      shape);
            }
            else if (tensorName == makeTensorName("self_attn.q_proj"))
            {
                auto q = m_Layers[currentLayer].selfAttnQKV.View(0, Config.querySize * Config.hiddenSize);
                UploadTensor(backend, q, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.k_proj"))
            {
                auto k = m_Layers[currentLayer].selfAttnQKV.View(
                    Config.querySize * Config.hiddenSize,
                    Config.keyValueSize * Config.hiddenSize);

                UploadTensor(backend, k, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.v_proj"))
            {
                auto v = m_Layers[currentLayer].selfAttnQKV.View(
                    (Config.querySize + Config.keyValueSize) * Config.hiddenSize,
                    Config.keyValueSize * Config.hiddenSize);

                UploadTensor(backend, v, file, headerSize, startOffset, endOffset, shape);
            }
            else if (tensorName == makeTensorName("self_attn.o_proj"))
            {
                CreateAndUploadTensor(
                    backend,
                    m_Layers[currentLayer].selfAttnO,
                    file,
                    headerSize,
                    startOffset,
                    endOffset,
                    shape);
            }
        }

        m_Hidden = backend.CreateTensor({Config.hiddenSize}, DataType::Float32);
        m_NextHidden = backend.CreateTensor({Config.hiddenSize}, DataType::Float32);
        m_Normalized = backend.CreateTensor({Config.hiddenSize}, DataType::Float32);
        m_QKV = backend.CreateTensor({Config.querySize + 2 * Config.keyValueSize}, DataType::Float32);

        m_Query = m_QKV.View(0, {Config.numAttentionHeads, Config.headDimension});

        m_Key = m_QKV.View(Config.querySize, {Config.numKeyValueHeads, Config.headDimension});

        m_Value = m_QKV.View(Config.querySize + Config.keyValueSize,
                             {Config.numKeyValueHeads, Config.headDimension});

        const auto halfDimension = Config.headDimension / 2;

        m_RopeCos = backend.CreateTensor(
            {Config.maxPositionEmbeddings, halfDimension},
            DataType::Float32);

        m_RopeSin = backend.CreateTensor(
            {Config.maxPositionEmbeddings, halfDimension},
            DataType::Float32);

        for (std::size_t position{}; position < Config.maxPositionEmbeddings; ++position)
        {
            backend.SinCosRoPE(
                m_RopeCos,
                m_RopeSin,
                position,
                Config.headDimension,
                static_cast<float>(Config.ropeTheta));
        }

        m_AttentionOutput = backend.CreateTensor({Config.numAttentionHeads, Config.headDimension}, DataType::Float32);
        m_AttentionProjected = backend.CreateTensor({Config.hiddenSize}, DataType::Float32);
        m_GateUp = backend.CreateTensor({Config.intermediateSize * 2}, DataType::Float32);
        m_Gate = m_GateUp.View(0, Config.intermediateSize);
        m_Up = m_GateUp.View(Config.intermediateSize, Config.intermediateSize);
        m_ActivatedGate = backend.CreateTensor({Config.intermediateSize}, DataType::Float32);
        m_FeedForward = backend.CreateTensor({Config.intermediateSize}, DataType::Float32);
        m_DownOutput = backend.CreateTensor({Config.hiddenSize}, DataType::Float32);
        m_Logits = backend.CreateTensor({Config.vocabSize}, DataType::Float32);

        m_KeyCaches.resize(Config.numHiddenLayers);
        m_ValueCaches.resize(Config.numHiddenLayers);

        for (std::size_t i{}; i < Config.numHiddenLayers; ++i)
        {
            m_KeyCaches[i] = backend.CreateTensor({
                                                      Config.maxPositionEmbeddings, Config.numKeyValueHeads,
                                                      Config.headDimension
                                                  }, DataType::Float32);
            m_ValueCaches[i] = backend.CreateTensor({
                                                        Config.maxPositionEmbeddings, Config.numKeyValueHeads,
                                                        Config.headDimension
                                                    }, DataType::Float32);
        }

        if (tokenizer->Load(path / "tokenizer.json"))
        {
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count();
            spdlog::info("[model] loaded {} tensors, {:.2f} MiB weights in {:.2f} s", tensorCount,
                         static_cast<double>(weightBytes) / (1024.0 * 1024.0), elapsed);
            return true;
        }

        spdlog::error("[model] failed to load tokenizer");
        return false;
    }

    spdlog::error("[model] failed to open {}", (path / "model.safetensors").string());

    return false;
}

void SmolLM2Model::Reset()
{
    m_Position = 0;
    spdlog::debug("[model] KV cache reset");
}

void SmolLM2Model::Prefill(std::span<const std::int32_t> tokenIds, IBackend& backend)
{
    if (tokenIds.size() > Config.maxPositionEmbeddings - m_Position)
        throw std::out_of_range("Maximum context length exceeded");

    for (const auto tokenId : tokenIds)
        DecodeStep(tokenId, backend);
}

void SmolLM2Model::DecodeStep(std::int32_t tokenId, IBackend& backend)
{
    if (tokenId < 0 || static_cast<std::size_t>(tokenId) >= Config.vocabSize)
        throw std::out_of_range("Token id is out of vocabulary range");

    if (m_Position >= Config.maxPositionEmbeddings)
        throw std::out_of_range("Maximum context length exceeded");

    const std::span<const std::int32_t> tokenIds(&tokenId, 1);

    backend.Embedding(m_TokenEmbedding, tokenIds, m_Hidden);

    for (std::size_t layerIndex{}; layerIndex < m_Layers.size(); ++layerIndex)
    {
        auto& [layernorm, downProj, gateUpProj, postAttentionLayernorm, selfAttnQKV, selfAttnO] = m_Layers[
            layerIndex];

        backend.RMSNorm(m_Hidden, layernorm, static_cast<float>(Config.rmsNormEps), m_Normalized);
        backend.Linear(selfAttnQKV, m_Normalized, m_QKV);
        backend.RoPE(m_Query, m_RopeCos, m_RopeSin, Config.numAttentionHeads, m_Position, Config.headDimension);
        backend.RoPE(m_Key, m_RopeCos, m_RopeSin, Config.numKeyValueHeads, m_Position, Config.headDimension);
        backend.CopyToCache(m_Key, m_KeyCaches[layerIndex], m_Position);
        backend.CopyToCache(m_Value, m_ValueCaches[layerIndex], m_Position);

        const auto validTokenCount = m_Position + 1;

        backend.Attention(m_Query, m_KeyCaches[layerIndex], m_ValueCaches[layerIndex], validTokenCount,
                          Config.numAttentionHeads,
                          Config.numKeyValueHeads, m_AttentionOutput);
        backend.Linear(selfAttnO, m_AttentionOutput, m_AttentionProjected);
        backend.Add(m_Hidden, m_AttentionProjected, m_NextHidden);

        std::swap(m_Hidden, m_NextHidden);

        backend.RMSNorm(m_Hidden, postAttentionLayernorm, static_cast<float>(Config.rmsNormEps), m_Normalized);
        backend.Linear(gateUpProj, m_Normalized, m_GateUp);
        backend.SiLU(m_Gate, m_ActivatedGate);
        backend.Multiply(m_ActivatedGate, m_Up, m_FeedForward);
        backend.Linear(downProj, m_FeedForward, m_DownOutput);
        backend.Add(m_Hidden, m_DownOutput, m_NextHidden);

        std::swap(m_Hidden, m_NextHidden);
    }

    backend.RMSNorm(m_Hidden, m_FinalNorm, static_cast<float>(Config.rmsNormEps), m_Normalized);

    backend.Linear(m_TokenEmbedding, m_Normalized, m_Logits);

    ++m_Position;
}

const Tensor& SmolLM2Model::Logits() const noexcept
{
    return m_Logits;
}
