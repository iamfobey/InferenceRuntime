#include <benchmark/benchmark.h>

#include "Backend/CPU/Backend.hpp"
#include "Model/ModelFactory.hpp"
#include "Runtime/Runtime.hpp"

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
    constexpr std::size_t BenchmarkRepetitions = 10;
    constexpr std::int32_t BenchmarkToken = 1;

    struct RuntimeBenchmarkOptions
    {
        std::string architecture;
        std::string modelPath;
    };

    void BenchmarkRuntimeTG300(
        benchmark::State& state,
        const RuntimeBenchmarkOptions& options)
    {
        auto model = ModelFactory::Create(options.architecture);

        if (!model)
        {
            state.SkipWithError(
                "Unsupported model architecture: " + options.architecture);
            return;
        }

        IModel* const modelPointer = model.get();

        const CpuBackendOptions backendOptions = {
            .threadCount = 6
        };

        auto backend = std::make_unique<CpuBackend>(backendOptions);

        IBackend* const backendPointer = backend.get();

        Runtime runtime(std::move(backend), std::move(model));

        if (!runtime.LoadModel(options.modelPath))
        {
            state.SkipWithError("Failed to load model");
            return;
        }

        // Небольшой warmup.
        // Нужен в том числе чтобы OpenMP worker threads уже были подняты.
        modelPointer->Reset();

        for (std::size_t i = 0; i < 8; ++i)
        {
            modelPointer->DecodeStep(
                BenchmarkToken,
                *backendPointer);
        }

        benchmark::DoNotOptimize(
            modelPointer->Logits().FloatData());

        modelPointer->Reset();

        for (auto _ : state)
        {
            modelPointer->Reset();

            const auto start =
                std::chrono::steady_clock::now();

            for (std::size_t i = 0; i < GenerationTokens; ++i)
            {
                modelPointer->DecodeStep(
                    BenchmarkToken,
                    *backendPointer);
            }

            const auto end =
                std::chrono::steady_clock::now();

            benchmark::DoNotOptimize(
                modelPointer->Logits().FloatData());

            const std::chrono::duration<double> elapsed =
                end - start;

            state.SetIterationTime(elapsed.count());
        }

        state.SetItemsProcessed(
            static_cast<std::int64_t>(state.iterations()) *
            static_cast<std::int64_t>(GenerationTokens));
    }
}

int main(const int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr
            << "Usage: RuntimeBenchmark "
            "<architecture> "
            "<path to directory with model> "
            "[Google Benchmark options]\n";

        return 1;
    }

    const RuntimeBenchmarkOptions options = {
        .architecture = argv[1],
        .modelPath = argv[2],
    };

    std::vector<char*> benchmarkArguments = {
        argv[0]
    };

    benchmarkArguments.insert(
        benchmarkArguments.end(),
        argv + 3,
        argv + argc);

    int benchmarkArgumentCount =
        static_cast<int>(benchmarkArguments.size());

    benchmark::Initialize(
        &benchmarkArgumentCount,
        benchmarkArguments.data());

    if (benchmark::ReportUnrecognizedArguments(
        benchmarkArgumentCount,
        benchmarkArguments.data()))
    {
        return 1;
    }

    benchmark::RegisterBenchmark(
            "RuntimeTG300",
            BenchmarkRuntimeTG300,
            options)
        ->Iterations(1)
        ->Repetitions(BenchmarkRepetitions)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
