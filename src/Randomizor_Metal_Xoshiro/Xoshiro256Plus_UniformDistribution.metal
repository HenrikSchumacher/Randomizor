R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Xoshiro256Plus_UniformDistribution.metal -o Xoshiro256Plus_UniformDistribution.air && xcrun -sdk macosx metallib Xoshiro256Plus_UniformDistribution.air -o Xoshiro256Plus_UniformDistribution.metallib

#include <metal_stdlib>

using namespace metal;

[[kernel]] void Xoshiro256Plus_UniformDistribution(
          device   ulong4 * states            [[buffer(0)]], // seeded states
          device   float4 * a                 [[buffer(1)]], // buffer for results
    const constant size_t & chunks_per_grid   [[buffer(2)]],
                                   
    const uint thread_position_in_grid          [[thread_position_in_grid]],
    const uint threads_per_grid                 [[threads_per_grid]]
)
{
    const uint i = thread_position_in_grid;
    
    thread ulong4 state = states[i];

//    // On some Apple GPU devices, this loop design might work faster.
//    uint chunks_per_thread = chunks_per_grid / threads_per_grid;
//    for( uint j = 0; j < chunks_per_thread; ++j)
//    {
//        size_t pos = chunks_per_thread*i+j;
        
    // On non-Apple GPU devices, this loop might work faster.
    for( uint j = 0; j < chunks_per_grid; j += threads_per_grid )
    {
        size_t pos = j + i;

        if( pos < chunks_per_grid )
        {
            float4 u; // To be filled from uniform distribution on [0,1).
            
            {
                // Xoshiro256+ implementation: http://prng.di.unimi.it/xoshiro256plus.c
                const uint64_t bits = state[0] + state[3];
                const uint64_t t = state[1] << 17;
                state[2] ^= state[0];
                state[3] ^= state[1];
                state[1] ^= state[2];
                state[0] ^= state[3];
                state[2] ^= t;
                state[3] = (state[3] << 45) | (state[3] >> 19);
                
                // Use half of bits for each float.
                u[0] = 0x1.0p-24f * static_cast<float>(
                    reinterpret_cast<thread const uint32_t*>(&bits)[0] >> 8
                );
                u[1] = 0x1.0p-24f * static_cast<float>(
                    reinterpret_cast<thread const uint32_t*>(&bits)[1] >> 8
                );
            }
            {
                // Xoshiro256+ implementation: http://prng.di.unimi.it/xoshiro256plus.c
                const uint64_t bits = state[0] + state[3];
                const uint64_t t = state[1] << 17;
                state[2] ^= state[0];
                state[3] ^= state[1];
                state[1] ^= state[2];
                state[0] ^= state[3];
                state[2] ^= t;
                state[3] = (state[3] << 45) | (state[3] >> 19);
                
                // Use half of bits for each float.
                u[2] = 0x1.0p-24f * static_cast<float>(
                    reinterpret_cast<thread const uint32_t*>(&bits)[0] >> 8
                );
                u[3] = 0x1.0p-24f * static_cast<float>(
                    reinterpret_cast<thread const uint32_t*>(&bits)[1] >> 8
                );
            }
            
            a[pos] = u;
        }
        
        // We have to update the states in case we want to call this function again.
        states[i] = state;
    }
    
}

// FIXME: Comment-out the following line for run-time compilation:
)"
