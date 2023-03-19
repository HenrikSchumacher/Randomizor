#pragma once
#include <cstdint>
#include <array>
#include <limits>
#include <type_traits>

namespace Randomizor
{
    // Xoshiro256+
    // Output: 64 bits
    // Period: 2^256 - 1
    // Footprint: 32 bytes
    // Original implementation: http://prng.di.unimi.it/xoshiro256plus.c
    // Version: 1.0
    class Xoshiro256Plus
    {
    public:
        
        using UInt = std::uint64_t;
        
        using state_type  = std::array<UInt,4>;
        using result_type = UInt;
        
        explicit constexpr Xoshiro256Plus(const UInt seed) noexcept
        :   state( SplitMix64{ seed }.generateSeedSequence<4>() )
        {}
        
        explicit constexpr Xoshiro256Plus(const state_type state_) noexcept
        :   state(state_)
        {}
        
        constexpr result_type operator()() noexcept
        {
            const UInt result = state[0] + state[3];
            const UInt t = state[1] << 17;
            state[2] ^= state[0];
            state[3] ^= state[1];
            state[1] ^= state[2];
            state[0] ^= state[3];
            state[2] ^= t;
            state[3] = (state[3] << 45) | (state[3] >> 19);
            return result;
        }
        
        // This is the jump function for the generator. It is equivalent
        // to 2^128 calls to operator(); it can be used to generate 2^128
        // non-overlapping subsequences for parallel computations.
        constexpr void Jump() noexcept
        {
            constexpr UInt jump [4] = {
                0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                0xa9582618e03fc9aa, 0x39abdc4529b1661c
            };
            
            state_type s = {};
            
            for( int a = 0; a < 4; ++a )
            {
                const UInt j = jump[a];
                
                for( int b = 0; b < 64; ++b )
                {
                    if( j & static_cast<UInt>(1) << b )
                    {
                        s[0] ^= state[0];
                        s[1] ^= state[1];
                        s[2] ^= state[2];
                        s[3] ^= state[3];
                    }
                    operator()();
                }
            }
            
            state = s;
        }
        
        // This is the long-jump function for the generator. It is equivalent to
        // 2^192 calls to operator(); it can be used to generate 2^64 starting points,
        // from each of which jump() will generate 2^64 non-overlapping
        // subsequences for parallel distributed computations.
        constexpr void LongJump() noexcept
        {
            constexpr UInt jump [4] = {
                0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635
            };
            
            state_type s = {};
            
            for( int a = 0; a < 4; ++a )
            {
                const UInt j = jump[a];
                
                for( int b = 0; b < 64; ++b )
                {
                    if( j & static_cast<UInt>(1) << b )
                    {
                        s[0] ^= state[0];
                        s[1] ^= state[1];
                        s[2] ^= state[2];
                        s[3] ^= state[3];
                    }
                    operator()();
                }
            }
            
            state = s;
        }
        
        static constexpr result_type min() noexcept
        {
            return std::numeric_limits<result_type>::lowest();
        }
        
        static constexpr result_type max() noexcept
        {
            return std::numeric_limits<result_type>::max();
        }
        
        constexpr state_type State() const noexcept
        {
            return state;
        }
        
        constexpr void SetState(const state_type state_) noexcept
        {
            state = state_;
        }
        
        friend bool operator ==(const Xoshiro256Plus& lhs, const Xoshiro256Plus& rhs) noexcept
        {
            return (lhs.state == rhs.state);
        }
        
        friend bool operator !=(const Xoshiro256Plus& lhs, const Xoshiro256Plus& rhs) noexcept
        {
            return (lhs.state != rhs.state);
        }
        
    private:
        
        state_type state;
    };
    
    force_inline void getNormalFloatPair( Xoshiro256Plus & random_engine, float & a, float & b )
    {
        std::int64_t ix;
        std::int64_t iy;
        std::int64_t is;
        
        constexpr std::int64_t threshold = (std::int64_t(1) << 46);
        
        do
        {
            const std::uint64_t bits = random_engine();
            
            ix = static_cast<std::int64_t>(
                reinterpret_cast<const std::int32_t*>(&bits)[0] >> 8
            );
            
            iy = static_cast<std::int64_t>(
                reinterpret_cast<const std::int32_t*>(&bits)[1] >> 8
            );
            
            is = ix * ix + iy * iy;
        } while ( is > threshold || is == std::int64_t(0) );
        
        const float x = 0x1.0p-23f * static_cast<float>( ix );
        const float y = 0x1.0p-23f * static_cast<float>( iy );
        const float s = 0x1.0p-46f * static_cast<float>( is );
        
        const float r = std::sqrt(- 2.0f * std::log(s) / s );
        a = r * x;
        b = r * y;
    }
    
}
