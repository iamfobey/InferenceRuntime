#include "Utils.hpp"

#include <cstdint>
#include <stdexcept>

#include "Core/Enums.hpp"
#include "Math/Math.hpp"


#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace Utils::Detail
{
#if defined(_M_X64) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(__i386__)

    inline bool ReadCPUID(
        unsigned leaf,
        unsigned subleaf,
        unsigned& eax,
        unsigned& ebx,
        unsigned& ecx,
        unsigned& edx
    ) noexcept
    {



#if defined(_MSC_VER)
    int registers[4]{};

    __cpuid(registers, 0);

        if (static_cast<unsigned>(registers[0])< leaf)
        {
            return false;
        }

    __cpuidex(
        registers,
        static_cast<int>(leaf),
        static_cast<int>(subleaf)
    );

    eax=static_cast<unsigned>(registers[0]);
    ebx=static_cast<unsigned>(registers[1]);
    ecx=static_cast<unsigned>(registers[2]);
    edx=static_cast<unsigned>(registers[3]);

        return true;

#elif defined(__GNUC__) || defined(__clang__)
    if (__get_cpuid_max(0, nullptr) < leaf)
        {
            return false;
        }

        return __get_cpuid_count(
        leaf,
        subleaf,
        &eax,
        &ebx,
        &ecx,
        &edx
    );
#endif
    }

    inline std::uint64_t ReadXCR0() noexcept
    {



#if defined(_MSC_VER)
    return _xgetbv (0);

#elif defined(__GNUC__) || defined(__clang__)
    unsigned eax;
    unsigned edx;

    __asm__ volatile (
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0)
    );

        return
            static_cast<std::uint64_t>(eax)|
    (static_cast<std::uint64_t>(edx)<< 32);
#endif
    }

    inline bool OSAllowsAVX() noexcept
    {
        unsigned eax;
        unsigned ebx;
        unsigned ecx;
        unsigned edx;

        if (!ReadCPUID(1, 0, eax, ebx, ecx, edx))
        {
            return false;
        }

        constexpr unsigned OSXSAVE_BIT = 1u << 27;
        constexpr unsigned AVX_BIT = 1u << 28;

        if ((ecx & OSXSAVE_BIT) == 0 ||
            (ecx & AVX_BIT) == 0)
        {
            return false;
        }

        return (ReadXCR0() & 0x6u) == 0x6u;
    }

#endif
} // namespace Utils::Detail


namespace Utils
{
    bool CPUSupportsFMA() noexcept
    {
#if defined(_M_X64) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(__i386__)

        if (!Detail::OSAllowsAVX())
        {
            return false;
        }

        unsigned eax;
        unsigned ebx;
        unsigned ecx;
        unsigned edx;

        if (!Detail::ReadCPUID(1, 0, eax, ebx, ecx, edx))
        {
            return false;
        }

        constexpr unsigned FMA_BIT = 1u << 12;

        return (ecx & FMA_BIT) != 0;
#else
        return false;
#endif
    }

    bool CPUSupportsAVX2() noexcept
    {
#if defined(_M_X64) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(__i386__)

        if (!Detail::OSAllowsAVX())
        {
            return false;
        }

        unsigned eax;
        unsigned ebx;
        unsigned ecx;
        unsigned edx;

        if (!Detail::ReadCPUID(7, 0, eax, ebx, ecx, edx))
        {
            return false;
        }

        constexpr unsigned AVX2_BIT = 1u << 5;

        return (ebx & AVX2_BIT) != 0;
#else
        return false;
#endif
    }

    bool CanRunAVX2FMAKernel() noexcept
    {
        static const bool supported =
            CPUSupportsAVX2() &&
            CPUSupportsFMA();

        return supported;
    }

    std::size_t DataTypeSize(const DataType dataType)
    {
        switch (dataType)
        {
        case DataType::Float16:
            return sizeof(std::uint16_t);
        case DataType::Float32:
            return sizeof(float);
        }

        throw std::invalid_argument("Unsupported DataType");
    }

    [[nodiscard]]
    std::vector<std::size_t> CreateContiguousStrides(
        const std::vector<std::size_t>& shape
    )
    {
        std::vector<std::size_t> strides(shape.size());

        std::size_t stride = 1;

        for (std::size_t i = shape.size(); i > 0; --i)
        {
            const std::size_t index = i - 1;

            strides[index] = stride;
            stride = Math::CheckedMultiply(stride, shape[index]);
        }

        return strides;
    }

    std::size_t ElementCount(const std::vector<std::size_t>& shape) noexcept
    {
        std::size_t count = 1;

        for (const std::size_t dimension : shape)
            count *= dimension;

        return count;
    }
}


