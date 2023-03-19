R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Xoshiro128Plus.metal -o Xoshiro128Plus.air && xcrun -sdk macosx metallib Xoshiro128Plus.air -o Xoshiro128Plus.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "block_size" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr size_t  block_size      = 64;

#include <metal_stdlib>

using namespace metal;

[[kernel]] void PCG_UniformDistribution(
          device   ulong2 * states            [[buffer(0)]], // seeded states
          device   float4 * a                 [[buffer(1)]], // buffer for results
    const constant size_t & chunks_per_grid   [[buffer(2)]],
                                   
    const uint thread_position_in_grid          [[thread_position_in_grid]],
    const uint threads_per_grid                 [[threads_per_grid]]
)
{
    const uint i = thread_position_in_grid;
    
    thread ulong2 s = states[i];
    
          uint64_t state = s[0];
    const uint64_t inc   = (s[1] | 1);

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
                const uint64_t oldstate = state;
                // Advance internal state
                state = oldstate * 6364136223846793005ULL + inc;
                // Calculate output function (XSH RR), uses old state for max ILP
                const uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
                const uint32_t rot = oldstate >> 59u;
                const uint32_t bits = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                
                u[0] = 0x1.0p-24f * static_cast<float>( bits >> 8 );
            }
            {
                const uint64_t oldstate = state;
                // Advance internal state
                state = oldstate * 6364136223846793005ULL + inc;
                // Calculate output function (XSH RR), uses old state for max ILP
                const uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
                const uint32_t rot = oldstate >> 59u;
                const uint32_t bits = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                
                u[1] = 0x1.0p-24f * static_cast<float>( bits >> 8 );
            }
            {
                const uint64_t oldstate = state;
                // Advance internal state
                state = oldstate * 6364136223846793005ULL + inc;
                // Calculate output function (XSH RR), uses old state for max ILP
                const uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
                const uint32_t rot = oldstate >> 59u;
                const uint32_t bits = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                
                u[2] = 0x1.0p-24f * static_cast<float>( bits >> 8 );
            }
            {
                const uint64_t oldstate = state;
                // Advance internal state
                state = oldstate * 6364136223846793005ULL + inc;
                // Calculate output function (XSH RR), uses old state for max ILP
                const uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
                const uint32_t rot = oldstate >> 59u;
                const uint32_t bits = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                
                u[3] = 0x1.0p-24f * static_cast<float>( bits >> 8 );
            }
            
            a[pos] = u;
        }
        
        // We have to update the states in case we want to call this function again.
        states[i] = state;
    }
    
} // SampleNormalDistribution

// FIXME: Comment-out the following line for run-time compilation:
)"

