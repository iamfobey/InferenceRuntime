#include <benchmark/benchmark.h>

#include "Backend/CPU/Backend.hpp"
#include "Model/ModelFactory.hpp"
#include "Runtime/Runtime.hpp"
#include "Model/ITokenizer.hpp"
#include "spdlog/spdlog.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr std::size_t GenerationTokens = 300;
    constexpr std::size_t BenchmarkIterations = 10;
    constexpr std::int32_t BenchmarkToken = 1;

    struct RuntimeBenchmarkOptions
    {
        std::string architecture;
        std::string modelPath;
        int threadCount;
    };

    void BenchmarkModelLoad(benchmark::State& state, const RuntimeBenchmarkOptions& options)
    {
        for (auto _ : state)
        {
            auto model = ModelFactory::Create(options.architecture);

            if (!model)
            {
                state.SkipWithError("Unsupported model architecture: " + options.architecture);
                return;
            }

            auto backend = std::make_unique<CpuBackend>(CpuBackendOptions{.threadCount = options.threadCount});
            Runtime runtime(std::move(backend), std::move(model));
            const auto start = std::chrono::steady_clock::now();

            if (!runtime.LoadModel(options.modelPath))
            {
                state.SkipWithError("Failed to load model");
                return;
            }

            state.SetIterationTime(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
    }

    void BenchmarkGenerationTG300(benchmark::State& state, const RuntimeBenchmarkOptions& options)
    {
        auto model = ModelFactory::Create(options.architecture);

        if (!model)
        {
            state.SkipWithError("Unsupported model architecture: " + options.architecture);
            return;
        }

        auto* const modelPointer = model.get();

        const CpuBackendOptions backendOptions = {
            .threadCount = options.threadCount
        };

        auto backend = std::make_unique<CpuBackend>(backendOptions);

        auto* const backendPointer = backend.get();

        Runtime runtime(std::move(backend), std::move(model));

        if (!runtime.LoadModel(options.modelPath))
        {
            state.SkipWithError("Failed to load model");
            return;
        }

        modelPointer->Reset();

        for (std::size_t i{}; i < 8; ++i)
            modelPointer->DecodeStep(BenchmarkToken, *backendPointer);

        benchmark::DoNotOptimize(modelPointer->Logits().FloatData());

        modelPointer->Reset();

        for (auto _ : state)
        {
            modelPointer->Reset();

            const auto start = std::chrono::steady_clock::now();

            for (std::size_t i = 0; i < GenerationTokens; ++i)
                modelPointer->DecodeStep(BenchmarkToken, *backendPointer);

            const auto end = std::chrono::steady_clock::now();

            benchmark::DoNotOptimize(modelPointer->Logits().FloatData());

            const std::chrono::duration<double> elapsed = end - start;

            state.SetIterationTime(elapsed.count());
        }

        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(GenerationTokens));
    }
}

int main(const int argc, char** argv)
{
    spdlog::set_level(spdlog::level::off);

    if (argc < 4)
    {
        spdlog::error(
            "Usage: RuntimeBenchmark <architecture> <path to directory with model> <thread_count> [Google Benchmark options]");

        return 1;
    }

    const RuntimeBenchmarkOptions options = {
        .architecture = argv[1],
        .modelPath = argv[2],
        .threadCount = std::atoi(argv[3])
    };

    std::vector benchmarkArguments = {argv[0]};

    benchmarkArguments.insert(benchmarkArguments.end(), argv + 4, argv + argc);

    auto benchmarkArgumentCount = static_cast<int>(benchmarkArguments.size());

    benchmark::Initialize(&benchmarkArgumentCount, benchmarkArguments.data());

    if (benchmark::ReportUnrecognizedArguments(benchmarkArgumentCount, benchmarkArguments.data()))
        return 1;

    spdlog::set_level(spdlog::level::off);

    benchmark::RegisterBenchmark(
            "ModelLoad",
            BenchmarkModelLoad,
            options)
        ->Iterations(BenchmarkIterations)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    benchmark::RegisterBenchmark(
            "GenerationTG300",
            BenchmarkGenerationTG300,
            options)
        ->Iterations(BenchmarkIterations)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
