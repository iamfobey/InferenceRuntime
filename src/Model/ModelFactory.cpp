#include "ModelFactory.hpp"

#include <memory>
#include <string_view>

#include "SmolLM2/SmolLM2Model.hpp"
#include "SmolLM2/SmolLM2Tokenizer.hpp"
#include "spdlog/spdlog.h"

std::unique_ptr<IModel> ModelFactory::Create(std::string_view architecture)
{
    if (architecture == "smollm2")
    {
        spdlog::info("[model] creating architecture: {}", architecture);
        auto ptr = std::make_unique<SmolLM2Model>();
        ptr->tokenizer = std::make_unique<SmolLM2Tokenizer>();
        return ptr;
    }

    spdlog::error("[model] unsupported architecture: {}", architecture);
    return nullptr;
}
