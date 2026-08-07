#include "ModelFactory.hpp"

#include "SmolLM2/SmolLM2Model.hpp"
#include "SmolLM2/SmolLM2Tokenizer.hpp"

#include <memory>
#include <string_view>

std::unique_ptr<IModel> ModelFactory::Create(const std::string_view architecture)
{
    if (architecture == "smollm2")
    {
        auto ptr = std::make_unique<SmolLM2Model>();
        ptr->tokenizer = std::make_unique<SmolLM2Tokenizer>();
        return ptr;
    }

    return nullptr;
}
