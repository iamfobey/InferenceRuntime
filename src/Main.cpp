#include "Backend/CPU/Backend.hpp"
#include "Model/IModel.hpp"
#include "Runtime/Runtime.hpp"

#include <iostream>
#include <memory>
#include <string>

#include "Model/ModelFactory.hpp"

int main(const int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr
            << "Usage: InferenceRuntime <architecture> <path to directory with model> <prompt>\n";

        return 1;
    }

    const auto architecture = argv[1];
    const auto modelPath = argv[2];
    const auto prompt = argv[3];

    try
    {
        auto model = ModelFactory::Create(architecture);

        if (!model)
        {
            std::cerr << "Unsupported model architecture: " << architecture << '\n';

            return 1;
        }

        CpuBackendOptions backendOptions = {
            .threadCount = 6
        };

        auto backend = std::make_unique<CpuBackend>(backendOptions);

        Runtime runtime(std::move(backend), std::move(model));

        if (!runtime.LoadModel(modelPath))
        {
            std::cerr << "Failed to load model\n";
            return 1;
        }

        constexpr std::size_t maximumNewTokens = 300;

        const std::string generatedText = runtime.Generate(prompt, maximumNewTokens);

        std::cout << generatedText << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
