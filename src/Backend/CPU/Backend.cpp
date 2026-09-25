#include "Backend.hpp"

#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include "Backend/CPU/Buffer.hpp"
#include "Math/Math.hpp"
#include "Utils/Utils.hpp"
#include "Utils/Converters.hpp"
#include "spdlog/spdlog.h"
#include "lowp/lowp.hpp"

CpuBackend::CpuBackend(const CpuBackendOptions options) :
    m_Options(options),
    m_ThreadPool(options.threadCount <= 0 ? 1 : static_cast<std::size_t>(options.threadCount))
{
    m_Options.threadCount = static_cast<int>(m_ThreadPool.ThreadCount());

    spdlog::info("[cpu] backend initialized: threads={}, AVX2={}", m_Options.threadCount, HAVE_AVX2_SUPPORT);
}

DeviceType CpuBackend::Device() const noexcept
{
    return DeviceType::CPU;
}

Tensor CpuBackend::CreateTensor(TensorDimVec shape, DataType dataType)
{
    const auto bytes = Math::CheckedMultiply(Utils::ElementCount(shape), Utils::DataTypeSize(dataType));
    Tensor tensor = {
        .shape = std::move(shape),
        .strides = Utils::CreateContiguousStrides(tensor.shape),
        .buffer = std::make_shared<CpuBuffer>(bytes),
        .byteOffset = 0,
        .dataType = dataType,
    };

    spdlog::debug("[cpu] allocated tensor: {} bytes, {} dimensions, dtype={}", bytes, tensor.shape.size(),
                  dataType == DataType::Float16 ? "f16" : "f32");

    return tensor;
}

void CpuBackend::Upload(Tensor& destination, const std::span<const float> source)
{
    destination.Validate(DeviceType::CPU);

    if (Utils::ElementCount(destination.shape) != source.size())
        throw std::invalid_argument("Upload source size does not match destination tensor");

    if (source.empty())
        return;

    switch (destination.dataType)
    {
    case DataType::Float16: Utils::Converters::ConvertFloat32ToFloat16(source.data(), destination.Float16Data(),
                                                                       source.size());
        break;
    case DataType::Float32: std::memcpy(destination.FloatData(), source.data(), source.size_bytes());
        break;
    }
}

void CpuBackend::Download(const Tensor& source, const std::span<float> destination)
{
    source.Validate(DeviceType::CPU);

    if (Utils::ElementCount(source.shape) != destination.size())
        throw std::invalid_argument("Download destination size does not match source tensor");

    if (destination.empty())
        return;

    switch (source.dataType)
    {
    case DataType::Float16: Utils::Converters::ConvertFloat16ToFloat32(source.Float16Data(), destination.data(),
                                                                       destination.size());
        break;
    case DataType::Float32: std::memcpy(destination.data(), source.FloatData(), destination.size_bytes());
        break;
    }
}

void CpuBackend::Embedding(const Tensor& embeddingTable, const std::span<const std::int32_t> tokenIds, Tensor& output)
{
    embeddingTable.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (embeddingTable.shape.size() != 2)
        throw std::invalid_argument("Embedding table must have shape [vocabularySize, hiddenSize]");

    const auto vocabularySize = embeddingTable.shape[0];
    const auto hiddenSize = embeddingTable.shape[1];
    const auto tokenCount = tokenIds.size();
    const auto expectedElementCount = tokenCount * hiddenSize;

    if (Utils::ElementCount(output.shape) != expectedElementCount)
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

    const auto dispatch = [&]<class EmbeddingType, class OutputType>()
    {
        Math::Embedding(embeddingTable.Data<EmbeddingType>(), tokenIds.data(), output.Data<OutputType>(), tokenCount,
                        vocabularySize, hiddenSize);
    };

    IR_DISPATCH_CASE(embeddingTable.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(embeddingTable.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(embeddingTable.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(embeddingTable.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported Embedding data type combination");
}

void CpuBackend::Linear(const Tensor& weights, const Tensor& input, Tensor& output)
{
    weights.Validate(DeviceType::CPU);
    input.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (weights.shape.size() != 2)
        throw std::invalid_argument("weights must have shape [matRows, matColumns]");

    const auto matRows = weights.shape[0];
    const auto matColumns = weights.shape[1];

    if (Utils::ElementCount(input.shape) != matColumns)
        throw std::invalid_argument("input element count must equal weights matColumns");

    if (Utils::ElementCount(output.shape) != matRows)
        throw std::invalid_argument("output element count must equal weights matRows");

    const auto dispatch = [&]<class WeightType, class InputType, class OutputType>()
    {
        m_ThreadPool.ParallelFor(
            0,
            matRows,
            [ weights = weights.Data<WeightType>(),
                input = input.Data<InputType>(),
                output = output.Data<OutputType>(),
                matColumns](const auto beginRow, const auto endRow)
            {
                Math::LinearRange(weights, input, output, beginRow, endRow, matColumns);
            });
    };

    IR_DISPATCH_CASE(weights.dataType == DataType::Float16 && input.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float16 && input.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float16 && input.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float16 && input.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float32 && input.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float32 && input.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float32 && input.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(weights.dataType == DataType::Float32 && input.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported Linear data type combination");
}

void CpuBackend::RMSNorm(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output)
{
    input.Validate(DeviceType::CPU);
    weight.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (input.shape != weight.shape)
        throw std::invalid_argument("input and weight must have equal shapes");
    if (input.shape != output.shape)
        throw std::invalid_argument("input and output must have equal shapes");

    const auto elementCount = Utils::ElementCount(input.shape);
    const auto dispatch = [&]<class InputType, class WeightType, class OutputType>()
    {
        Math::RMSNorm(input.Data<InputType>(), weight.Data<WeightType>(), epsilon, output.Data<OutputType>(), elementCount);
    };

    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && weight.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && weight.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && weight.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && weight.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && weight.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && weight.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && weight.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && weight.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported RMSNorm data type combination");
}

void CpuBackend::Add(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    inputA.Validate(DeviceType::CPU);
    inputB.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (inputA.shape != inputB.shape)
        throw std::invalid_argument("inputA and inputB must have equal shapes");
    if (inputA.shape != output.shape)
        throw std::invalid_argument("inputA and output must have equal shapes");

    const auto elementCount = Utils::ElementCount(inputA.shape);
    const auto dispatch = [&]<class InputAType, class InputBType, class OutputType>()
    {
        Math::Add(inputA.Data<InputAType>(), inputB.Data<InputBType>(), output.Data<OutputType>(), elementCount);
    };

    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported Add data type combination");
}

void CpuBackend::Multiply(const Tensor& inputA, const Tensor& inputB, Tensor& output)
{
    inputA.Validate(DeviceType::CPU);
    inputB.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (inputA.shape != inputB.shape)
        throw std::invalid_argument("inputA and inputB must have equal shapes");
    if (inputA.shape != output.shape)
        throw std::invalid_argument("inputA and output must have equal shapes");

    const auto elementCount = Utils::ElementCount(inputA.shape);
    const auto dispatch = [&]<class InputAType, class InputBType, class OutputType>()
    {
        Math::Multiply(inputA.Data<InputAType>(), inputB.Data<InputBType>(), output.Data<OutputType>(), elementCount);
    };

    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float16 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float16 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float16,
                     lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(inputA.dataType == DataType::Float32 && inputB.dataType == DataType::Float32 && output.dataType == DataType::Float32,
                     lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported Multiply data type combination");
}

void CpuBackend::SiLU(const Tensor& input, Tensor& output)
{
    input.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

    if (input.shape != output.shape)
        throw std::invalid_argument("input and output must have equal shapes");

    const auto elementCount = Utils::ElementCount(input.shape);
    const auto dispatch = [&]<class InputType, class OutputType>()
    {
        Math::SiLU(input.Data<InputType>(), output.Data<OutputType>(), elementCount);
    };

    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && output.dataType == DataType::Float16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float16 && output.dataType == DataType::Float32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && output.dataType == DataType::Float16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(input.dataType == DataType::Float32 && output.dataType == DataType::Float32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported SiLU data type combination");
}

void CpuBackend::ComputeRoPECosSin(Tensor& sourceCos, Tensor& sourceSin, std::size_t position, std::size_t headDimension,
                                   float theta)
{
    sourceCos.Validate(DeviceType::CPU);
    sourceSin.Validate(DeviceType::CPU);

    if (headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires even headDimension");

    if (theta <= 0.0f)
        throw std::invalid_argument("RoPE theta must be positive");

    const auto offset = position * (headDimension / 2);
    const auto dispatch = [&]<class CosType, class SinType>()
    {
        Math::ComputeRoPECosSin(sourceCos.Data<CosType>() + offset, sourceSin.Data<SinType>() + offset, position, headDimension, theta);
    };

    IR_DISPATCH_CASE(sourceCos.dataType == DataType::Float16 && sourceSin.dataType == DataType::Float16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(sourceCos.dataType == DataType::Float16 && sourceSin.dataType == DataType::Float32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(sourceCos.dataType == DataType::Float32 && sourceSin.dataType == DataType::Float16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(sourceCos.dataType == DataType::Float32 && sourceSin.dataType == DataType::Float32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported ComputeRoPECosSin data type combination");
}

void CpuBackend::RoPE(Tensor& source, const Tensor& inputCos, const Tensor& inputSin, std::size_t headCount,
                      size_t position, std::size_t headDimension)
{
    source.Validate(DeviceType::CPU);
    inputCos.Validate(DeviceType::CPU);
    inputSin.Validate(DeviceType::CPU);

    if (headCount == 0 || headDimension == 0 || headDimension % 2 != 0)
        throw std::invalid_argument("RoPE requires positive headCount and even headDimension");

    const auto expectedElementCount = Math::CheckedMultiply(headCount, headDimension);

    if (Utils::ElementCount(source.shape) != expectedElementCount)
        throw std::invalid_argument("q sizes do not match headCount * headDimension");

    const auto offset = position * (headDimension / 2);
    const auto dispatch = [&]<class SourceType, class CosType, class SinType>()
    {
        Math::RoPE(source.Data<SourceType>(), inputCos.Data<CosType>() + offset, inputSin.Data<SinType>() + offset, headCount,
                   headDimension);
    };

    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && inputCos.dataType == DataType::Float16 && inputSin.dataType == DataType::Float16,
                     lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && inputCos.dataType == DataType::Float16 && inputSin.dataType == DataType::Float32,
                     lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && inputCos.dataType == DataType::Float32 && inputSin.dataType == DataType::Float16,
                     lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && inputCos.dataType == DataType::Float32 && inputSin.dataType == DataType::Float32,
                     lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && inputCos.dataType == DataType::Float16 && inputSin.dataType == DataType::Float16,
                     lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && inputCos.dataType == DataType::Float16 && inputSin.dataType == DataType::Float32,
                     lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && inputCos.dataType == DataType::Float32 && inputSin.dataType == DataType::Float16,
                     lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && inputCos.dataType == DataType::Float32 && inputSin.dataType == DataType::Float32,
                     lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported RoPE data type combination");
}

void CpuBackend::Attention(const Tensor& q, const Tensor& kCache, const Tensor& vCache,
                           size_t validTokenCount,
                           size_t attentionHeadCount, size_t keyValueHeadCount, Tensor& output)
{
    q.Validate(DeviceType::CPU);
    kCache.Validate(DeviceType::CPU);
    vCache.Validate(DeviceType::CPU);
    output.Validate(DeviceType::CPU);

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

    const auto dispatch = [&]<class QType, class KType, class VType, class OutputType>()
    {
        Math::Attention(q.Data<QType>(), kCache.Data<KType>(), vCache.Data<VType>(), output.Data<OutputType>(), validTokenCount,
                        attentionHeadCount, keyValueHeadCount, headDimension);
    };

    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float16,
        lowp::f16, lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float32,
        lowp::f16, lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float16,
        lowp::f16, lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float32,
        lowp::f16, lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float16,
        lowp::f16, lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float32,
        lowp::f16, lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float16,
        lowp::f16, lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float16 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float32,
        lowp::f16, lowp::f32, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float16,
        lowp::f32, lowp::f16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float32,
        lowp::f32, lowp::f16, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float16,
        lowp::f32, lowp::f16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float16 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float32,
        lowp::f32, lowp::f16, lowp::f32, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float16,
        lowp::f32, lowp::f32, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float16 && output.dataType == DataType
        ::Float32,
        lowp::f32, lowp::f32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float16,
        lowp::f32, lowp::f32, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(
        q.dataType == DataType::Float32 && kCache.dataType == DataType::Float32 && vCache.dataType == DataType::Float32 && output.dataType == DataType
        ::Float32,
        lowp::f32, lowp::f32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported Attention data type combination");
}

void CpuBackend::CopyToCache(const Tensor& source, Tensor& cache, size_t position)
{
    source.Validate(DeviceType::CPU);
    cache.Validate(DeviceType::CPU);

    if (cache.shape.size() < 2)
        throw std::invalid_argument("cache must have at least two dimensions");

    if (position >= cache.shape[0])
        throw std::out_of_range("Cache position exceeds cache capacity");

    std::size_t cacheRowElementCount = 1;

    for (std::size_t i = 1; i < cache.shape.size(); ++i)
        cacheRowElementCount = Math::CheckedMultiply(cacheRowElementCount, cache.shape[i]);

    if (Utils::ElementCount(source.shape) != cacheRowElementCount)
        throw std::invalid_argument("source size does not match one cache position");

    const auto elementCount = Utils::ElementCount(source.shape);
    const auto dispatch = [&]<class SourceType, class CacheType>()
    {
        Math::CopyToCache(source.Data<SourceType>(), cache.Data<CacheType>(), position, elementCount);
    };

    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && cache.dataType == DataType::Float16, lowp::f16, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float16 && cache.dataType == DataType::Float32, lowp::f16, lowp::f32)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && cache.dataType == DataType::Float16, lowp::f32, lowp::f16)
    IR_DISPATCH_CASE(source.dataType == DataType::Float32 && cache.dataType == DataType::Float32, lowp::f32, lowp::f32)

    throw std::invalid_argument("Unsupported CopyToCache data type combination");
}

void CpuBackend::Synchronize() {}
