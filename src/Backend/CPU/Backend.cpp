#include "Backend.hpp"

#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Backend/CPU/Buffer.hpp"
#include "Math/Math.hpp"
#include "Utils/Utils.hpp"

#include <omp.h>

#include "Utils/Converters.hpp"

CpuBackend::CpuBackend(const CpuBackendOptions options) :
    m_Options(options)
{
    if (m_Options.threadCount == 0)
        m_Options.threadCount = 1;

    omp_set_dynamic(0);
    omp_set_num_threads(m_Options.threadCount);
}

DeviceType CpuBackend::Device() const noexcept
{
    return DeviceType::CPU;
}

Tensor CpuBackend::CreateTensor(std::vector<std::size_t> shape, const DataType dataType)
{
    Tensor tensor = {
        .shape = std::move(shape),
        .strides = Utils::CreateContiguousStrides(tensor.shape),
        .buffer = std::make_shared<CpuBuffer>(
            Math::CheckedMultiply(Utils::ElementCount(tensor.shape), Utils::DataTypeSize(dataType))),
        .byteOffset = 0,
        .dataType = dataType,
    };

    return tensor;
}

void CpuBackend::Upload(Tensor& destination, const std::span<const float> source)
{
    destination.Validate(DeviceType::CPU, destination.dataType);

    if (destination.ElementCount() != source.size())
        throw std::invalid_argument("Upload source size does not match destination tensor");

    if (source.empty())
        return;

    switch (destination.dataType)
    {
    case DataType::Float16:
        Utils::Converters::ConvertFloat32ToFloat16(source.data(), destination.Float16Data(), source.size());
        break;
    case DataType::Float32:
        std::memcpy(destination.FloatData(), source.data(), source.size_bytes());
        break;
    }
}

void CpuBackend::Download(const Tensor& source, const std::span<float> destination)
{
    source.Validate(DeviceType::CPU, source.dataType);

    if (source.ElementCount() != destination.size())
        throw std::invalid_argument("Download destination size does not match source tensor");

    if (destination.empty())
        return;

    switch (source.dataType)
    {
    case DataType::Float16:
        Utils::Converters::ConvertFloat16ToFloat32(source.Float16Data(), destination.data(), destination.size());
        break;
    case DataType::Float32:
        std::memcpy(destination.data(), source.FloatData(), destination.size_bytes());
        break;
    }
}

void CpuBackend::Embedding(const Tensor& embeddingTable, const std::span<const std::int32_t> tokenIds, Tensor& output)
{
    embeddingTable.Validate(DeviceType::CPU, DataType::Float16);
    output.Validate(DeviceType::CPU, DataType::Float32);

    if (embeddingTable.shape.size() != 2)
        throw std::invalid_argument("Embedding table must have shape [vocabularySize, hiddenSize]");

    const auto vocabularySize = embeddingTable.shape[0];
    const auto hiddenSize = embeddingTable.shape[1];
    const auto tokenCount = tokenIds.size();
    const auto expectedElementCount = tokenCount * hiddenSize;

    if (output.ElementCount() != expectedElementCount)
        throw std::invalid_argument("Embedding output element count does not match tokenCount * hiddenSize");

    if (tokenCount == 1)
    {
        if (output.shape.size() != 1 || output.shape[0] != hiddenSize)
            throw std::invalid_argument("Single-token embedding output must have shape [hiddenSize]");
    }
    else
    {
        if (output.shape.size() != 2 || output.shape[0] != tokenCount || output.shape[1] != hiddenSize)
            throw std::invalid_argument("Multi-token embedding output must have shape [tokenCount, hiddenSize]");
    }

    Math::Embedding(embeddingTable.Float16Data(), tokenIds.data(), output.FloatData(), tokenCount, vocabularySize,
                    hiddenSize);
}

void CpuBackend::Linear(const Tensor& W, const Tensor& x, Tensor& y)
{
    W.Validate(DeviceType::CPU, DataType::Float16);
    x.Validate(DeviceType::CPU, DataType::Float32);
    y.Validate(DeviceType::CPU, DataType::Float32);

    if (W.shape.size() != 2)
        throw std::invalid_argument("W must have shape [matRows, matColumns]");

    const auto matRows = W.shape[0];
    const auto matColumns = W.shape[1];

    if (x.ElementCount() != matColumns)
        throw std::invalid_argument("x element count must equal W matColumns");

    if (y.ElementCount() != matRows)
        throw std::invalid_argument("y element count must equal W matRows");

    Math::Linear(W.Float16Data(), x.FloatData(), y.FloatData(), matRows, matColumns);
}

void CpuBackend::RMSNorm(const Tensor& x, const Tensor& weight, const float epsilon, Tensor& y)
{
    x.Validate(DeviceType::CPU, DataType::Float32);
    weight.Validate(DeviceType::CPU, DataType::Float16);
    y.Validate(DeviceType::CPU, DataType::Float32);

    Utils::RequireSameShape(x, weight, "x and weight must have equal shapes");
    Utils::RequireSameShape(x, y, "x and y must have equal shapes");

    Math::RMSNorm(x.FloatData(), weight.Float16Data(), epsilon, y.FloatData(), x.ElementCount());
}

void CpuBackend::Add(const Tensor& a, const Tensor& b, Tensor& output)
{
    a.Validate(DeviceType::CPU, DataType::Float32);
    b.Validate(DeviceType::CPU, DataType::Float32);
    output.Validate(DeviceType::CPU, DataType::Float32);

    Utils::RequireSameShape(a, b, "a and b must have equal shapes");
    Utils::RequireSameShape(a, output, "a and output must have equal shapes");

    Math::Add(a.FloatData(), b.FloatData(), output.FloatData(), a.ElementCount());
}

void CpuBackend::Multiply(const Tensor& a, const Tensor& b, Tensor& output)
{
    a.Validate(DeviceType::CPU, DataType::Float32);
    b.Validate(DeviceType::CPU, DataType::Float32);
    output.Validate(DeviceType::CPU, DataType::Float32);

    Utils::RequireSameShape(a, b, "a and b must have equal shapes");
    Utils::RequireSameShape(a, output, "a and output must have equal shapes");

    Math::Multiply(a.FloatData(), b.FloatData(), output.FloatData(), a.ElementCount());
}

void CpuBackend::SiLU(const Tensor& x, Tensor& output)
{
    x.Validate(DeviceType::CPU, DataType::Float32);
    output.Validate(DeviceType::CPU, DataType::Float32);

    Utils::RequireSameShape(x, output, "x and output must have equal shapes");

    Math::SiLU(x.FloatData(), output.FloatData(), x.ElementCount());
}

void CpuBackend::PreCalc(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                         float theta)
{
    sourceCos.Validate(DeviceType::CPU, DataType::Float32);
    sourceSin.Validate(DeviceType::CPU, DataType::Float32);

    if (headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires even headDimension");

    if (theta <= 0.0f)
        throw std::invalid_argument("RoPE theta must be positive");

    const auto offset = position * (headDimension / 2);

    Math::PreCalc(sourceCos.FloatData() + offset, sourceSin.FloatData() + offset, position, headDimension, theta);
}

void CpuBackend::RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
                      size_t position, std::size_t headDimension)
{
    source.Validate(DeviceType::CPU, DataType::Float32);
    inputCos.Validate(DeviceType::CPU, DataType::Float32);
    inputSin.Validate(DeviceType::CPU, DataType::Float32);

    if (headCount == 0 || headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires positive headCount and even headDimension");

    const auto expectedElementCount = Math::CheckedMultiply(headCount, headDimension);

    if (source.ElementCount() != expectedElementCount)
        throw std::invalid_argument("q sizes do not match headCount * headDimension");

    const auto offset = position * (headDimension / 2);

    Math::RoPE(source.FloatData(), inputCos.FloatData() + offset, inputSin.FloatData() + offset, headCount,
               headDimension);
}

void CpuBackend::Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                           Tensor& scores, const std::size_t validTokenCount,
                           const std::size_t attentionHeadCount, const std::size_t keyValueHeadCount, Tensor& output)
{
    q.Validate(DeviceType::CPU, DataType::Float32);
    kCache.Validate(DeviceType::CPU, DataType::Float32);
    vCache.Validate(DeviceType::CPU, DataType::Float32);
    output.Validate(DeviceType::CPU, DataType::Float32);

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
        throw std::invalid_argument(
            "kCache and vCache must have shape [contextLength, keyValueHeadCount, headDimension]");

    if (kCache.shape != vCache.shape)
        throw std::invalid_argument("kCache and vCache must have equal shapes");

    if (kCache.shape[0] < validTokenCount || kCache.shape[1] != keyValueHeadCount || kCache.shape[2] != headDimension)
        throw std::invalid_argument("Cache shape does not match Attention parameters");

    Math::Attention(q.FloatData(), kCache.FloatData(), vCache.FloatData(), output.FloatData(),
                    scores.FloatData() + validTokenCount,
                    validTokenCount, attentionHeadCount, keyValueHeadCount, headDimension);
}

void CpuBackend::CopyToCache(const Tensor& source, Tensor& cache, const std::size_t position)
{
    source.Validate(DeviceType::CPU, DataType::Float32);
    cache.Validate(DeviceType::CPU, DataType::Float32);

    if (cache.shape.size() < 2)
        throw std::invalid_argument("cache must have at least two dimensions");

    if (position >= cache.shape[0])
        throw std::out_of_range("Cache position exceeds cache capacity");

    std::size_t cacheRowElementCount = 1;

    for (std::size_t i = 1; i < cache.shape.size(); ++i)
        cacheRowElementCount = Math::CheckedMultiply(cacheRowElementCount, cache.shape[i]);

    if (source.ElementCount() != cacheRowElementCount)
        throw std::invalid_argument("source size does not match one cache position");

    Math::CopyToCache(source.FloatData(), cache.FloatData(), position, source.ElementCount());
}

void CpuBackend::Synchronize()
{
    // ...
}
