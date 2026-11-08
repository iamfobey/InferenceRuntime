#include <benchmark/benchmark.h>

#include "Backend/CPU/Backend.hpp"
#include "Model/ModelFactory.hpp"
#include "Runtime/Runtime.hpp"
#include "Model/ITokenizer.hpp"
#include "spdlog/spdlog.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    constexpr std::size_t GenerationTokens = 128;
    constexpr std::int32_t BenchmarkToken = 1;

    void BenchmarkModelLoad(benchmark::State& state, const RuntimeOptions& options,
                            const std::string& modelPath)
    {
        for (auto _ : state)
        {
            Runtime runtime(options);

            const auto start = std::chrono::steady_clock::now();

            if (!runtime.LoadModel(modelPath))
            {
                state.SkipWithError("Failed to load model");
                return;
            }

            state.SetIterationTime(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
    }

    void BenchmarkGenerationTG(benchmark::State& state, const RuntimeOptions& options,
                               const std::string& modelPath)
    {
        Runtime runtime(options);

        if (!runtime.LoadModel(modelPath))
        {
            state.SkipWithError("Failed to load model");
            return;
        }

        const auto& model = runtime.GetModel();
        const auto& backend = runtime.GetBackend();

        model->Reset();

        for (std::size_t i{}; i < 8; ++i)
            model->DecodeStep(BenchmarkToken, *backend);

        benchmark::DoNotOptimize(model->Logits().FloatData());

        model->Reset();

        for (auto _ : state)
        {
            model->Reset();

            const auto start = std::chrono::steady_clock::now();

            for (std::size_t i = 0; i < GenerationTokens; ++i)
                model->DecodeStep(BenchmarkToken, *backend);

            const auto end = std::chrono::steady_clock::now();

            benchmark::DoNotOptimize(model->Logits().FloatData());

            const std::chrono::duration<double> elapsed = end - start;

            state.SetIterationTime(elapsed.count());
        }

        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(GenerationTokens));
    }
}

int main(const int argc, char** argv)
{
    spdlog::set_level(spdlog::level::off);

    if (argc < 5)
    {
        spdlog::error(
            "Usage: RuntimeBenchmark <backend driver> <architecture> <path to directory with model> <thread_count> [Google Benchmark options]");

        return 1;
    }

    const auto backendDriver = argv[1];
    const auto modelArchitecture = argv[2];
    const auto modelPath = argv[3];
    const auto threadCount = argv[4];

    RuntimeOptions options = {
        .modelArchitecture = modelArchitecture,
        .backendDriver = backendDriver,
        .cpuBackendOptions = {
            .threadCount = std::atoi(threadCount)
        }
    };

    std::vector benchmarkArguments = {argv[0]};

    benchmarkArguments.insert(benchmarkArguments.end(), argv + 5, argv + argc);

    auto benchmarkArgumentCount = static_cast<int>(benchmarkArguments.size());

    benchmark::Initialize(&benchmarkArgumentCount, benchmarkArguments.data());

    if (benchmark::ReportUnrecognizedArguments(benchmarkArgumentCount, benchmarkArguments.data()))
        return 1;

    spdlog::set_level(spdlog::level::off);

    benchmark::RegisterBenchmark(
            "ModelLoad",
            BenchmarkModelLoad,
            options,
            modelPath)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    benchmark::RegisterBenchmark(
            "GenerationTG",
            BenchmarkGenerationTG,
            options,
            modelPath)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
