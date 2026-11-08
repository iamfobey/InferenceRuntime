#include "Backend.hpp"

#include <cuda_runtime.h>
#include <lowp/lowp.hpp>
#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Backend/CUDA/Buffer.hpp"
#include "Math/MathCUDA.hpp"
#include "Math/MathCPU.hpp"
#include "Utils/Utils.hpp"

CudaBackend::CudaBackend()
{
    int deviceCount{};

    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
        throw std::runtime_error("No CUDA device is available");

    spdlog::info("[cuda] backend initialized");
}

DeviceType CudaBackend::Device() const noexcept
{
    return DeviceType::CUDA;
}

Tensor CudaBackend::CreateTensor(TensorDimVec shape, const DataType dataType)
{
    const auto bytes = Math::CheckedMultiply(Utils::ElementCount(shape), Utils::DataTypeSize(dataType));

    Tensor tensor = {
        .shape = std::move(shape),
        .strides = Utils::CreateContiguousStrides(tensor.shape),
        .buffer = std::make_shared<CudaBuffer>(bytes),
        .byteOffset = 0,
        .dataType = dataType,
    };

    spdlog::debug("[cuda] allocated tensor: {} bytes, {} dimensions, dtype={}", bytes, tensor.shape.size(),
                  dataType == DataType::Float16 ? "f16" : "f32");

    return tensor;
}

void CudaBackend::Upload(Tensor& destination, const std::span<const float> source)
{
    destination.Validate(DeviceType::CUDA);

    if (Utils::ElementCount(destination.shape) != source.size())
        throw std::invalid_argument("Upload source size does not match destination tensor");
    if (source.empty())
        return;

    if (destination.dataType == DataType::Float32)
    {
        CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(destination.buffer->Data()) + destination.byteOffset,
            source.data(), source.size_bytes(), cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<lowp::f16> values(source.size());

        for (std::size_t i{}; i < source.size(); ++i)
            values[i] = source[i];

        CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(destination.buffer->Data()) + destination.byteOffset,
            values.data(), source.size_bytes() / 2, cudaMemcpyHostToDevice));
    }
}

void CudaBackend::Download(const Tensor& source, const std::span<float> destination)
{
    source.Validate(DeviceType::CUDA);

    if (Utils::ElementCount(source.shape) != destination.size())
        throw std::invalid_argument("Download destination size does not match source tensor");
    if (destination.empty())
        return;

    if (source.dataType == DataType::Float32)
    {
        CUDA_CHECK(cudaMemcpy(destination.data(),
            static_cast<const std::byte*>(source.buffer->Data()) + source.byteOffset,
            destination.size_bytes(), cudaMemcpyDeviceToHost));
    }
    else
    {
        std::vector<lowp::f16> values(destination.size());

        CUDA_CHECK(cudaMemcpy(values.data(),
            static_cast<const std::byte*>(source.buffer->Data()) + source.byteOffset,
            destination.size_bytes() / 2, cudaMemcpyDeviceToHost));

        for (std::size_t i{}; i < destination.size(); ++i)
            destination[i] = static_cast<float>(values[i]);
    }
}

void CudaBackend::Embedding(const Tensor& embeddingTable, const std::span<const std::int32_t> tokenIds, Tensor& output)
{
    embeddingTable.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (embeddingTable.shape.size() != 2)
        throw std::invalid_argument("Embedding table must have shape [vocabularySize, hiddenSize]");

    const auto vocabularySize = embeddingTable.shape[0];
    const auto hiddenSize = embeddingTable.shape[1];
    const auto tokenCount = tokenIds.size();
    const auto expectedElementCount = Math::CheckedMultiply(tokenCount, hiddenSize);

    if (Utils::ElementCount(output.shape) != expectedElementCount)
        throw std::invalid_argument("Embedding output element count does not match tokenCount * hiddenSize");

    if (tokenCount == 1)
    {
        if (output.shape.size() != 1 || output.shape[0] != hiddenSize)
            throw std::invalid_argument("Single-token embedding output must have shape [hiddenSize]");
    }
    else if (output.shape.size() != 2 || output.shape[0] != tokenCount || output.shape[1] != hiddenSize)
    {
        throw std::invalid_argument("Multi-token embedding output must have shape [tokenCount, hiddenSize]");
    }

    if (tokenIds.empty() || hiddenSize == 0)
        return;

    CudaBuffer deviceTokenIds(Math::CheckedMultiply(tokenCount, sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(deviceTokenIds.Data(), tokenIds.data(), tokenIds.size_bytes(), cudaMemcpyHostToDevice));

    Math::CUDA::Embedding(static_cast<const std::byte*>(embeddingTable.buffer->Data()) + embeddingTable.byteOffset,
                          embeddingTable.dataType, static_cast<const std::int32_t*>(deviceTokenIds.Data()),
                          static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                          tokenCount, vocabularySize, hiddenSize);
}

void CudaBackend::Linear(const Tensor& weights, const Tensor& input, Tensor& output)
{
    weights.Validate(DeviceType::CUDA);
    input.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (weights.shape.size() != 2)
        throw std::invalid_argument("weights must have shape [matRows, matColumns]");

    const auto rows = weights.shape[0];
    const auto columns = weights.shape[1];

    if (Utils::ElementCount(input.shape) != columns)
        throw std::invalid_argument("input element count must equal weights matColumns");

    if (Utils::ElementCount(output.shape) != rows)
        throw std::invalid_argument("output element count must equal weights matRows");

    Math::CUDA::Linear(static_cast<const std::byte*>(weights.buffer->Data()) + weights.byteOffset, weights.dataType,
                       static_cast<const std::byte*>(input.buffer->Data()) + input.byteOffset, input.dataType,
                       static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType, rows, columns);
}

void CudaBackend::RMSNorm(const Tensor& input, const Tensor& weight, const float epsilon, Tensor& output)
{
    input.Validate(DeviceType::CUDA);
    weight.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (input.shape != weight.shape)
        throw std::invalid_argument("input and weight must have equal shapes");

    if (input.shape != output.shape)
        throw std::invalid_argument("input and output must have equal shapes");

    Math::CUDA::RMSNorm(static_cast<const std::byte*>(input.buffer->Data()) + input.byteOffset, input.dataType,
                        static_cast<const std::byte*>(weight.buffer->Data()) + weight.byteOffset, weight.dataType,
                        epsilon, static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                        Utils::ElementCount(input.shape));
}

void CudaBackend::Add(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    inputA.Validate(DeviceType::CUDA);
    inputB.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (inputA.shape != inputB.shape)
        throw std::invalid_argument("inputA and inputB must have equal shapes");
    if (inputA.shape != output.shape)
        throw std::invalid_argument("inputA and output must have equal shapes");

    Math::CUDA::Add(static_cast<const std::byte*>(inputA.buffer->Data()) + inputA.byteOffset, inputA.dataType,
                    static_cast<const std::byte*>(inputB.buffer->Data()) + inputB.byteOffset, inputB.dataType,
                    static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                    Utils::ElementCount(inputA.shape));
}

void CudaBackend::Multiply(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    inputA.Validate(DeviceType::CUDA);
    inputB.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (inputA.shape != inputB.shape)
        throw std::invalid_argument("inputA and inputB must have equal shapes");
    if (inputA.shape != output.shape)
        throw std::invalid_argument("inputA and output must have equal shapes");

    Math::CUDA::Multiply(static_cast<const std::byte*>(inputA.buffer->Data()) + inputA.byteOffset, inputA.dataType,
                         static_cast<const std::byte*>(inputB.buffer->Data()) + inputB.byteOffset, inputB.dataType,
                         static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                         Utils::ElementCount(inputA.shape));
}

void CudaBackend::SwiGLU(const Tensor& input, const Tensor& gateUp, Tensor& output)
{
    input.Validate(DeviceType::CUDA);
    gateUp.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (input.shape != gateUp.shape || input.shape != output.shape)
        throw std::invalid_argument("input, gateUp and output must have equal shapes");

    Math::CUDA::SwiGLU(static_cast<const std::byte*>(input.buffer->Data()) + input.byteOffset, input.dataType,
                       static_cast<const std::byte*>(gateUp.buffer->Data()) + gateUp.byteOffset, gateUp.dataType,
                       static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                       Utils::ElementCount(input.shape));
}

void CudaBackend::ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, const std::size_t position,
                                    const std::size_t headDimension, const float theta)
{
    sourceCos.Validate(DeviceType::CUDA);
    sourceSin.Validate(DeviceType::CUDA);

    if (headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires even headDimension");
    if (theta <= 0.0f)
        throw std::invalid_argument("RoPE theta must be positive");

    const auto offset = Math::CheckedMultiply(position, headDimension / 2);
    const auto pairCount = headDimension / 2;

    if (offset > Utils::ElementCount(sourceCos.shape) || pairCount > Utils::ElementCount(sourceCos.shape) - offset ||
        offset > Utils::ElementCount(sourceSin.shape) || pairCount > Utils::ElementCount(sourceSin.shape) - offset)
        throw std::out_of_range("RoPE position exceeds cosine or sine tensor");

    Math::CUDA::ComputeRoPECosSin(static_cast<std::byte*>(sourceCos.buffer->Data()) + sourceCos.byteOffset +
                                  offset * Utils::DataTypeSize(sourceCos.dataType),
                                  sourceCos.dataType,
                                  static_cast<std::byte*>(sourceSin.buffer->Data()) + sourceSin.byteOffset +
                                  offset * Utils::DataTypeSize(sourceSin.dataType),
                                  sourceSin.dataType, position, headDimension, theta);
}

void CudaBackend::RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, const std::size_t headCount,
                       const std::size_t position, const std::size_t headDimension)
{
    source.Validate(DeviceType::CUDA);
    inputCos.Validate(DeviceType::CUDA);
    inputSin.Validate(DeviceType::CUDA);

    if (headCount == 0 || headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires positive headCount and even headDimension");
    if (Utils::ElementCount(source.shape) != Math::CheckedMultiply(headCount, headDimension))
        throw std::invalid_argument("q sizes do not match headCount * headDimension");

    const auto offset = Math::CheckedMultiply(position, headDimension / 2);
    const auto pairCount = headDimension / 2;

    if (offset > Utils::ElementCount(inputCos.shape) || pairCount > Utils::ElementCount(inputCos.shape) - offset ||
        offset > Utils::ElementCount(inputSin.shape) || pairCount > Utils::ElementCount(inputSin.shape) - offset)
        throw std::out_of_range("RoPE position exceeds cosine or sine tensor");

    Math::CUDA::RoPE(static_cast<std::byte*>(source.buffer->Data()) + source.byteOffset, source.dataType,
                     static_cast<const std::byte*>(inputCos.buffer->Data()) + inputCos.byteOffset +
                     offset * Utils::DataTypeSize(inputCos.dataType),
                     inputCos.dataType,
                     static_cast<const std::byte*>(inputSin.buffer->Data()) + inputSin.byteOffset +
                     offset * Utils::DataTypeSize(inputSin.dataType),
                     inputSin.dataType,
                     headCount, headDimension);
}

void CudaBackend::Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                            const std::size_t validTokenCount, const std::size_t attentionHeadCount,
                            const std::size_t keyValueHeadCount, Tensor& output)
{
    q.Validate(DeviceType::CUDA);
    kCache.Validate(DeviceType::CUDA);
    vCache.Validate(DeviceType::CUDA);
    output.Validate(DeviceType::CUDA);

    if (attentionHeadCount == 0 || keyValueHeadCount == 0)
        throw std::invalid_argument("Attention head counts must be positive");
    if (attentionHeadCount % keyValueHeadCount != 0)
        throw std::invalid_argument("attentionHeadCount must be divisible by keyValueHeadCount");
    if (q.shape.size() != 2 || q.shape[0] != attentionHeadCount)
        throw std::invalid_argument("q must have shape [attentionHeadCount, headDimension]");

    const auto headDimension = q.shape[1];

    if (headDimension == 0)
        throw std::invalid_argument("headDimension must be positive");
    if (output.shape != q.shape)
        throw std::invalid_argument("output shape must equal q shape");
    if (kCache.shape.size() != 3 || vCache.shape.size() != 3)
        throw std::invalid_argument("kCache and vCache must have shape [contextLength, keyValueHeadCount, headDimension]");
    if (kCache.shape != vCache.shape)
        throw std::invalid_argument("kCache and vCache must have equal shapes");
    if (kCache.shape[0] < validTokenCount || kCache.shape[1] != keyValueHeadCount || kCache.shape[2] != headDimension)
        throw std::invalid_argument("Cache shape does not match Attention parameters");

    Math::CUDA::Attention(static_cast<const std::byte*>(q.buffer->Data()) + q.byteOffset, q.dataType,
                          static_cast<const std::byte*>(kCache.buffer->Data()) + kCache.byteOffset, kCache.dataType,
                          static_cast<const std::byte*>(vCache.buffer->Data()) + vCache.byteOffset, vCache.dataType,
                          static_cast<std::byte*>(output.buffer->Data()) + output.byteOffset, output.dataType,
                          validTokenCount, attentionHeadCount, keyValueHeadCount, headDimension);
}

void CudaBackend::CopyToCache(const Tensor& source, Tensor& cache, const std::size_t position)
{
    source.Validate(DeviceType::CUDA);
    cache.Validate(DeviceType::CUDA);

    if (cache.shape.size() < 2)
        throw std::invalid_argument("cache must have at least two dimensions");
    if (position >= cache.shape[0])
        throw std::out_of_range("Cache position exceeds cache capacity");

    std::size_t cacheRowElementCount = 1;

    for (std::size_t i = 1; i < cache.shape.size(); ++i)
        cacheRowElementCount = Math::CheckedMultiply(cacheRowElementCount, cache.shape[i]);

    const auto elementCount = Utils::ElementCount(source.shape);

    if (elementCount != cacheRowElementCount)
        throw std::invalid_argument("source size does not match one cache position");

    Math::CUDA::CopyToCache(static_cast<const std::byte*>(source.buffer->Data()) + source.byteOffset,
                            source.dataType,
                            static_cast<std::byte*>(cache.buffer->Data()) + cache.byteOffset,
                            cache.dataType, position, elementCount);
}

void CudaBackend::Synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}
