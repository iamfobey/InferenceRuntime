#include "Backend/CPU/Backend.hpp"
#include "Model/IModel.hpp"
#include "Runtime/Runtime.hpp"
#include "Model/ModelFactory.hpp"
#include "Model/ITokenizer.hpp"
#include "spdlog/spdlog.h"

#include <iostream>
#include <memory>
#include <string>

int main(const int argc, char** argv)
{
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    const auto* level = std::getenv("INFERENCE_RUNTIME_LOG_LEVEL");
    spdlog::set_level(level == nullptr ? spdlog::level::info : spdlog::level::from_str(level));

    if (argc < 5)
    {
        spdlog::error("Usage: InferenceRuntime <architecture> <path to directory with model> <prompt> <thread_count>");

        return 1;
    }

    const auto architecture = argv[1];
    const auto modelPath = argv[2];
    const auto prompt = argv[3];
    const auto threadCount = argv[4];

    try
    {
        auto model = ModelFactory::Create(architecture);

        if (!model)
        {
            spdlog::error("Unsupported model architecture: {}", architecture);

            return 1;
        }

        CpuBackendOptions backendOptions = {
            .threadCount = std::atoi(threadCount)
        };

        auto backend = std::make_unique<CpuBackend>(backendOptions);

        Runtime runtime(std::move(backend), std::move(model));

        if (!runtime.LoadModel(modelPath))
        {
            spdlog::error("Failed to load model");
            return 1;
        }

        constexpr std::size_t maximumNewTokens = 300;

        const std::string generatedText = runtime.Generate(prompt, maximumNewTokens);

        spdlog::info("Generation complete: {} bytes", generatedText.size());
        std::cout << generatedText << '\n';
    }
    catch (const std::exception& e)
    {
        spdlog::error("{}", e.what());
        return 1;
    }

    return 0;
}
