#include "SmolLM2Tokenizer.hpp"

#include "SmolLM2Model.hpp"
#include "Model/IModel.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace
{
    bool IsDirectByte(const std::uint16_t value)
    {
        return (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || (value >= 174 && value <= 255);
    }

    void AppendUtf8(std::string& result, const char32_t codePoint)
    {
        if (codePoint <= 0x7F)
        {
            result.push_back(static_cast<char>(codePoint));
        }
        else if (codePoint <= 0x7FF)
        {
            result.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
            result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
        else if (codePoint <= 0xFFFF)
        {
            result.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
            result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
        else
        {
            result.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
            result.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
    }

    const std::array<std::string, 256>& ByteEncoder()
    {
        static const std::array<std::string, 256> encoder = []
        {
            std::array<std::string, 256> result{};
            char32_t extraCodePoint = 256;

            for (std::uint16_t byte = 0; byte < 256; ++byte)
            {
                const char32_t codePoint = IsDirectByte(byte) ? static_cast<char32_t>(byte) : extraCodePoint++;
                AppendUtf8(result[byte], codePoint);
            }

            return result;
        }();

        return encoder;
    }

    const std::unordered_map<char32_t, std::uint8_t>& ByteDecoder()
    {
        static const std::unordered_map<char32_t, std::uint8_t> decoder = []
        {
            std::unordered_map<char32_t, std::uint8_t> result{};
            char32_t extraCodePoint = 256;

            for (std::uint16_t byte = 0; byte < 256; ++byte)
            {
                const char32_t codePoint = IsDirectByte(byte) ? static_cast<char32_t>(byte) : extraCodePoint++;
                result.emplace(codePoint, static_cast<std::uint8_t>(byte));
            }

            return result;
        }();

        return decoder;
    }

    char32_t ReadUtf8CodePoint(std::string_view text, std::size_t& position)
    {
        const auto first = static_cast<std::uint8_t>(text[position++]);

        if ((first & 0x80) == 0)
            return first;

        if ((first & 0xE0) == 0xC0)
        {
            const auto second = static_cast<std::uint8_t>(text[position++]);
            return static_cast<char32_t>(((first & 0x1F) << 6) | (second & 0x3F));
        }

        if ((first & 0xF0) == 0xE0)
        {
            const auto second = static_cast<std::uint8_t>(text[position++]);
            const auto third = static_cast<std::uint8_t>(text[position++]);
            return static_cast<char32_t>(((first & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F));
        }

        if ((first & 0xF8) == 0xF0)
        {
            const auto second = static_cast<std::uint8_t>(text[position++]);
            const auto third = static_cast<std::uint8_t>(text[position++]);
            const auto fourth = static_cast<std::uint8_t>(text[position++]);
            return static_cast<char32_t>(((first & 0x07) << 18) | ((second & 0x3F) << 12) | ((third & 0x3F) << 6) | (
                fourth & 0x3F));
        }

        throw std::runtime_error("Invalid UTF-8");
    }

    std::string MergeKey(std::string_view left, std::string_view right)
    {
        std::string key{};
        key.reserve(left.size() + right.size() + 1);
        key.append(left);
        key.push_back('\0');
        key.append(right);
        return key;
    }

    bool IsDigit(const unsigned char ch)
    {
        return ch >= '0' && ch <= '9';
    }

    bool IsWhitespace(const unsigned char ch)
    {
        return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v';
    }

    bool IsLetter(const unsigned char ch)
    {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch >= 0x80;
    }

    bool IsOther(const unsigned char ch)
    {
        return !IsWhitespace(ch) && !IsLetter(ch) && !IsDigit(ch);
    }

    void ByteLevelSplit(std::string_view text, std::vector<std::string_view>& result)
    {
        static constexpr std::array<std::string_view, 7> contractions{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

        std::size_t position = 0;

        while (position < text.size())
        {
            bool contractionFound = false;

            for (const std::string_view contraction : contractions)
            {
                if (text.substr(position, contraction.size()) == contraction)
                {
                    result.emplace_back(text.substr(position, contraction.size()));
                    position += contraction.size();
                    contractionFound = true;
                    break;
                }
            }

            if (contractionFound)
                continue;

            const auto current = static_cast<unsigned char>(text[position]);

            if (current == ' ' && position + 1 < text.size() &&
                IsLetter(static_cast<unsigned char>(text[position + 1])))
            {
                std::size_t end = position + 2;
                while (end < text.size() && IsLetter(static_cast<unsigned char>(text[end])))
                    ++end;

                result.emplace_back(text.substr(position, end - position));
                position = end;
                continue;
            }

            if (IsLetter(current))
            {
                std::size_t end = position + 1;
                while (end < text.size() && IsLetter(static_cast<unsigned char>(text[end])))
                    ++end;

                result.emplace_back(text.substr(position, end - position));
                position = end;
                continue;
            }

            if (IsDigit(current))
            {
                result.emplace_back(text.substr(position, 1));
                ++position;
                continue;
            }

            if (current == ' ' && position + 1 < text.size() && IsOther(static_cast<unsigned char>(text[position + 1])))
            {
                std::size_t end = position + 2;
                while (end < text.size() && IsOther(static_cast<unsigned char>(text[end])))
                    ++end;

                result.emplace_back(text.substr(position, end - position));
                position = end;
                continue;
            }

            if (IsOther(current))
            {
                std::size_t end = position + 1;
                while (end < text.size() && IsOther(static_cast<unsigned char>(text[end])))
                    ++end;

                result.emplace_back(text.substr(position, end - position));
                position = end;
                continue;
            }

            std::size_t end = position + 1;
            while (end < text.size() && IsWhitespace(static_cast<unsigned char>(text[end])))
                ++end;

            if (end < text.size() && end - position > 1 && text[end - 1] == ' ')
            {
                result.emplace_back(text.substr(position, end - position - 1));
                position = end - 1;
            }
            else
            {
                result.emplace_back(text.substr(position, end - position));
                position = end;
            }
        }
    }

    std::vector<std::string_view> PreTokenize(std::string_view text)
    {
        std::vector<std::string_view> result{};
        std::size_t begin = 0;

        for (std::size_t position = 0; position < text.size(); ++position)
        {
            if (!IsDigit(static_cast<unsigned char>(text[position])))
                continue;

            if (position > begin)
                ByteLevelSplit(text.substr(begin, position - begin), result);

            ByteLevelSplit(text.substr(position, 1), result);
            begin = position + 1;
        }

        if (begin < text.size())
            ByteLevelSplit(text.substr(begin), result);

        return result;
    }
}

bool SmolLM2Tokenizer::Load(const std::filesystem::path& path)
{
    using namespace simdjson;

    ondemand::parser parser;
    const padded_string json = padded_string::load(path.string());
    ondemand::document tokenizerJson = parser.iterate(json);

    m_Tokens.clear();
    m_MergeRanks.clear();
    m_AddedTokens.clear();

    for (auto tokenValue : tokenizerJson["added_tokens"].get_array())
    {
        ondemand::object tokenObject = tokenValue.get_object();
        const std::uint64_t tokenId = tokenObject["id"].get_uint64();
        const std::string_view token = tokenObject["content"].get_string();

        m_AddedTokens.emplace(std::string(token), static_cast<std::size_t>(tokenId));
    }

    ondemand::object model = tokenizerJson["model"].get_object();

    for (auto field : model["vocab"].get_object())
    {
        const std::string_view token = field.unescaped_key();
        const std::uint64_t tokenId = field.value().get_uint64();

        m_Tokens.emplace(std::string(token), static_cast<std::size_t>(tokenId));
    }

    std::size_t rank = 0;

    for (auto mergeValue : model["merges"].get_array())
    {
        const std::string_view merge = mergeValue.get_string();
        const std::size_t separator = merge.find(' ');

        if (separator == std::string_view::npos)
            throw std::runtime_error("Invalid BPE merge");

        m_MergeRanks.emplace(MergeKey(merge.substr(0, separator), merge.substr(separator + 1)), rank++);
    }

    return true;
}

std::vector<std::int32_t> SmolLM2Tokenizer::Encode(std::string_view text) const
{
    std::vector<std::int32_t> tokens{};

    const auto encodeNormalText = [this, &tokens](std::string_view normalText)
    {
        const std::vector<std::string_view> pieces = PreTokenize(normalText);
        const auto& byteEncoder = ByteEncoder();

        for (const std::string_view piece : pieces)
        {
            std::vector<std::string> symbols{};
            symbols.reserve(piece.size());

            for (const unsigned char byte : piece)
                symbols.emplace_back(byteEncoder[byte]);

            while (symbols.size() > 1)
            {
                std::size_t bestRank = std::numeric_limits<std::size_t>::max();
                std::size_t bestPosition = std::numeric_limits<std::size_t>::max();

                for (std::size_t position = 0; position + 1 < symbols.size(); ++position)
                {
                    const auto merge = m_MergeRanks.find(MergeKey(symbols[position], symbols[position + 1]));

                    if (merge != m_MergeRanks.end() && merge->second < bestRank)
                    {
                        bestRank = merge->second;
                        bestPosition = position;
                    }
                }

                if (bestPosition == std::numeric_limits<std::size_t>::max())
                    break;

                const std::string left = symbols[bestPosition];
                const std::string right = symbols[bestPosition + 1];

                std::vector<std::string> merged{};
                merged.reserve(symbols.size());

                for (std::size_t position = 0; position < symbols.size();)
                {
                    if (position + 1 < symbols.size() && symbols[position] == left && symbols[position + 1] == right)
                    {
                        merged.emplace_back(symbols[position] + symbols[position + 1]);
                        position += 2;
                    }
                    else
                    {
                        merged.emplace_back(std::move(symbols[position]));
                        ++position;
                    }
                }

                symbols = std::move(merged);
            }

            for (const std::string& symbol : symbols)
            {
                const auto token = m_Tokens.find(symbol);

                if (token == m_Tokens.end())
                    throw std::runtime_error("BPE token not found in vocabulary");

                tokens.emplace_back(static_cast<std::int32_t>(token->second));
            }
        }
    };

    std::size_t normalTextBegin = 0;
    std::size_t position = 0;

    while (position < text.size())
    {
        const std::pair<const std::string, std::size_t>* matchedToken = nullptr;

        for (const auto& token : m_AddedTokens)
        {
            if (text.substr(position, token.first.size()) == token.first && (matchedToken == nullptr || token.first.
                size() > matchedToken->first.size()))
                matchedToken = &token;
        }

        if (matchedToken == nullptr)
        {
            ++position;
            continue;
        }

        if (position > normalTextBegin)
            encodeNormalText(text.substr(normalTextBegin, position - normalTextBegin));

        tokens.emplace_back(static_cast<std::int32_t>(matchedToken->second));
        position += matchedToken->first.size();
        normalTextBegin = position;
    }

    if (normalTextBegin < text.size())
        encodeNormalText(text.substr(normalTextBegin));

    return tokens;
}

std::string SmolLM2Tokenizer::Decode(std::span<const std::int32_t> tokenIds) const
{
    std::string byteLevelText{};

    for (const std::int32_t tokenId : tokenIds)
    {
        bool found = false;

        for (const auto& [token, id] : m_Tokens)
        {
            if (id != static_cast<std::size_t>(tokenId))
                continue;

            byteLevelText += token;
            found = true;
            break;
        }

        if (!found)
            throw std::runtime_error("Token ID not found in vocabulary");
    }

    std::string result{};
    const auto& byteDecoder = ByteDecoder();
    std::size_t position = 0;

    while (position < byteLevelText.size())
    {
        const char32_t codePoint = ReadUtf8CodePoint(byteLevelText, position);
        const auto byte = byteDecoder.find(codePoint);

        if (byte == byteDecoder.end())
            throw std::runtime_error("Invalid ByteLevel code point");

        result.push_back(static_cast<char>(byte->second));
    }

    return result;
}