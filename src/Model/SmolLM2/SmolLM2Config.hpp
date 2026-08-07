#pragma once

#include <filesystem>
#include <string>

#include "simdjson/simdjson.h"

namespace std::filesystem
{
    class path;
}

struct SmolLM2Config
{
    std::string_view torchDtype;
    double rmsNormEps;
    std::size_t hiddenSize;
    std::size_t intermediateSize;
    std::size_t maxPositionEmbeddings;
    std::size_t numAttentionHeads;
    std::size_t numHiddenLayers;
    std::size_t numKeyValueHeads;
    std::size_t ropeTheta;
    std::size_t vocabSize;
    std::size_t headDimension;
    std::size_t querySize;
    std::size_t keyValueSize;
    bool tieWordEmbeddings;

    static SmolLM2Config Load(const std::filesystem::path& path)
    {
        using namespace simdjson;
        ondemand::parser parser;
        const padded_string json = padded_string::load(path.string());
        ondemand::document config = parser.iterate(json);
        SmolLM2Config smolLm2Config = {
            .torchDtype = config["torch_dtype"],
            .rmsNormEps = config["rms_norm_eps"],
            .hiddenSize = config["hidden_size"],
            .intermediateSize = config["intermediate_size"],
            .maxPositionEmbeddings = config["max_position_embeddings"],
            .numAttentionHeads = config["num_attention_heads"],
            .numHiddenLayers = config["num_hidden_layers"],
            .numKeyValueHeads = config["num_key_value_heads"],
            .ropeTheta = config["rope_theta"],
            .vocabSize = config["vocab_size"],
            .tieWordEmbeddings = config["tie_word_embeddings"],
        };
        smolLm2Config.headDimension = smolLm2Config.hiddenSize / smolLm2Config.numAttentionHeads;
        smolLm2Config.querySize = smolLm2Config.numAttentionHeads * smolLm2Config.headDimension;
        smolLm2Config.keyValueSize = smolLm2Config.numKeyValueHeads * smolLm2Config.headDimension;

        if (smolLm2Config.numAttentionHeads % smolLm2Config.numKeyValueHeads != 0)
            return {};

        if (smolLm2Config.hiddenSize % smolLm2Config.numAttentionHeads != 0)
            return {}; // TODO: Error

        return smolLm2Config;
    }
};
