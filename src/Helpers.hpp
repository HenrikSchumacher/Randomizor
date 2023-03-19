#pragma once

namespace Randomizor
{
    force_inline constexpr float FloatFrom32Bits( const std::uint32_t i ) noexcept
    {
        return (i >> 8) * 0x1.0p-24f;
    }
    
    force_inline constexpr float FloatFromBits( const std::uint64_t i ) noexcept
    {
        return (i >> 8) * 0x1.0p-24f;
    }
    
    force_inline void FloatPairFromBits( const std::uint64_t i, float & a, float & b ) noexcept
    {
        a = ( reinterpret_cast<const std::uint32_t*>(&i)[0] >> 8) * 0x1.0p-24f;
        b = ( reinterpret_cast<const std::uint32_t*>(&i)[1] >> 8) * 0x1.0p-24f;
    }
    
    force_inline constexpr double DoubleFromBits( const std::uint64_t i ) noexcept
    {
        return (i >> 11) * 0x1.0p-53;
    }
}
