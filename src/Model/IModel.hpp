#pragma once

#include "ITokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>

class IBackend;
struct Tensor;

class IModel
{
public:
	virtual ~IModel() = default;

	[[nodiscard]]
	virtual std::string_view Architecture() const noexcept = 0;

	virtual bool Load(const std::filesystem::path& path, IBackend& backend) = 0;

	virtual void Reset() = 0;

	virtual void Prefill(std::span<const std::int32_t> tokenIds, IBackend& backend) = 0;

	virtual void DecodeStep(std::int32_t tokenId, IBackend& backend) = 0;

	[[nodiscard]]
	virtual const Tensor& Logits() const noexcept = 0;

	std::unique_ptr<ITokenizer> tokenizer;
};