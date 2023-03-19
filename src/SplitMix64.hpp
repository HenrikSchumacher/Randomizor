#pragma once

namespace Randomizor
{
    // SplitMix64
    // Output: 64 bits
    // Period: 2^64
    // Footprint: 8 bytes
    // Original implementation: http://prng.di.unimi.it/splitmix64.c
    class SplitMix64
    {
    public:
        
        using UInt         = std::uint64_t;
        using state_type   = UInt;
        using result_type  = UInt;
        
        explicit constexpr SplitMix64(const state_type state_ ) noexcept
        :   state(state_)
        {}
        
        constexpr result_type operator()() noexcept
        {
            UInt z = (state += static_cast<UInt>(0x9e3779b97f4a7c15));
            z = (z ^ (z >> 30)) * static_cast<UInt>(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)) * static_cast<UInt>(0x94d049bb133111eb);
            return z ^ (z >> 31);
        }
        
        template <std::size_t N>
        constexpr std::array<UInt,N> generateSeedSequence() noexcept
        {
            std::array<UInt, N> seeds = {};
            
            for( auto& seed : seeds )
            {
                seed = operator()();
            }
            
            return seeds;
        }
        
        constexpr result_type min() noexcept
        {
            return std::numeric_limits<result_type>::lowest();
        }
        
        constexpr result_type max() noexcept
        {
            return std::numeric_limits<result_type>::max();
        }
        
        constexpr state_type serialize() const noexcept
        {
            return state;
        }
        
        constexpr void deserialize(const state_type state_) noexcept
        {
            state = state_;
        }
        
        friend bool operator ==(const SplitMix64& lhs, const SplitMix64& rhs) noexcept
        {
            return (lhs.state == rhs.state);
        }
        
        friend bool operator !=(const SplitMix64& lhs, const SplitMix64& rhs) noexcept
        {
            return (lhs.state != rhs.state);
        }
        
    private:
        
        state_type state;
    };
    
}
