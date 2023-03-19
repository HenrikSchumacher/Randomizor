#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Accelerate/Accelerate.h>

#include <iostream>
#include <random>
#include <cmath>

#define TOOLS_ENABLE_PROFILER // enable profiler

#include "../Randomizor_Metal_Xoshiro.hpp"
#include "../Randomizor_Metal_PCG.hpp"



using namespace Tools;

int main(int argc, const char * argv[])
{
    const size_t thread_count = 8;
    
//    const size_t n_ = size_t(1024) * size_t(1024) * size_t(1024);
    const size_t n_ = size_t(1024) * size_t(1024) * size_t(1024);
    
    
    NS::SharedPtr<MTL::Device> device = NS::TransferPtr(
        reinterpret_cast<MTL::Device *>( MTL::CopyAllDevices()->object(0) )
    );
    
    Randomizor::Randomizor_Metal_Xoshiro gen_Xoshiro ( device, 24576 * 4, 1024/2/2 );
    gen_Xoshiro.RequirePipeline();
    gen_Xoshiro.RequireSeed();
    
    gen_Xoshiro.RequireReservoir(n_);
    
    const size_t n = gen_Xoshiro.ReservoirSize();
    
    float * restrict b = gen_Xoshiro.Reservoir();

    
    Randomizor::Randomizor_Metal_PCG gen_PCG ( device, 24576 * 4, 1024/2/2 );
    gen_PCG.RequirePipeline();
    gen_PCG.RequireSeed();
    gen_PCG.LoadReservoir( b, n );
    
//    tic("BNNS (uniform)");
//    #pragma omp parallel for num_threads(thread_count)
//    for( size_t thread = 0; thread < thread_count; ++thread )
//    {
//        BNNSNDArrayDescriptor desc;
//        desc.layout     = BNNSDataLayoutVector;
//        desc.size[0]    = v.Dimension(1);
//        desc.stride[0]  = 1;
//        desc.data       = v.data(thread);
//        desc.data_type  = BNNSDataTypeFloat32;
//        desc.table_data = nullptr;
//
//        BNNSRandomGenerator gen = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nullptr);
//
//        BNNSRandomFillUniformFloat(gen, &desc, float(0), float(1));
//
//        BNNSDestroyRandomGenerator(gen);
//    }
//    toc("BNNS (uniform)");
//
//    tic("BNNS (normal)");
//    #pragma omp parallel for num_threads(thread_count)
//    for( size_t thread = 0; thread < thread_count; ++thread )
//    {
//        BNNSNDArrayDescriptor desc;
//        desc.layout     = BNNSDataLayoutVector;
//        desc.size[0]    = v.Dimension(1);
//        desc.stride[0]  = 1;
//        desc.data       = v.data(thread);
//        desc.data_type  = BNNSDataTypeFloat32;
//        desc.table_data = nullptr;
//
//        BNNSRandomGenerator gen = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nullptr);
//
//        BNNSRandomFillNormalFloat(gen, &desc, float(0), float(1));
//
//        BNNSDestroyRandomGenerator(gen);
//    }
//    toc("BNNS (normal)");

    std::random_device r;
    std::vector<std::uint64_t> seeds ( thread_count);

    for( size_t i = 0; i < thread_count; ++i )
    {
        reinterpret_cast<std::uint32_t*>(&seeds[i])[0] = r();
        reinterpret_cast<std::uint32_t*>(&seeds[i])[1] = r();
    }



    tic("STL (unif)");
    #pragma omp parallel for num_threads(thread_count)
    for( size_t thread = 0; thread < thread_count; ++thread )
    {
        // Create the actual random engine.
        std::mt19937_64 random_engine ( seeds[thread] );

        std::uniform_real_distribution<float> dist (0,1);

        const size_t i_begin = JobPointer<size_t>(n,thread_count,thread  );
        const size_t i_end   = JobPointer<size_t>(n,thread_count,thread+1);

        for( size_t i = i_begin; i < i_end; ++i )
        {
            b[i] = dist( random_engine );
        }
    }
    toc("STL (unif)");

    tic("STL (normal)");
    #pragma omp parallel for num_threads(thread_count)
    for( size_t thread = 0; thread < thread_count; ++thread )
    {
        // Create the actual random engine.
        std::mt19937_64 random_engine ( seeds[thread] );

        std::normal_distribution<float> dist (0,1);

        const size_t i_begin = JobPointer<size_t>(n,thread_count,thread  );
        const size_t i_end   = JobPointer<size_t>(n,thread_count,thread+1);

        for( size_t i = i_begin; i < i_end; ++i )
        {
            b[i] = dist( random_engine );
        }
    }
    toc("STL (normal)");
    
    
    tic("Xoshiro (normal, rejection, int)");
    #pragma omp parallel for num_threads(thread_count)
    for( size_t thread = 0; thread < thread_count; ++thread )
    {
        // Create the actual random engine.
        Randomizor::Xoshiro256Plus random_engine ( seeds[thread] );
        
        const size_t i_begin = JobPointer<size_t>(n,thread_count,thread  );
        const size_t i_end   = JobPointer<size_t>(n,thread_count,thread+1);
        
        if( i_end > i_begin )
        {
            const size_t i_begin_odd = i_begin % 2;
            const size_t i_end_odd   = i_end   % 2;
            
            float x;
            float y;
            
            getNormalFloatPair( random_engine, x, y );
            
            b[i_begin] = x;
            
            for( size_t i = i_begin + i_begin_odd; i < i_end - i_end_odd; i+=2 )
            {
                getNormalFloatPair( random_engine, b[i+0], b[i+1] );
            }
            
            b[i_end-1] = y;
        }
    }
    toc("Xoshiro (normal, rejection, int)");

    
    tic(gen_Xoshiro.ClassName()+"Fill_Normal");
    gen_Xoshiro.Fill_Normal();
    toc(gen_Xoshiro.ClassName()+"Fill_Normal");
    tic(gen_Xoshiro.ClassName()+"Fill_Normal");
    gen_Xoshiro.Fill_Normal();
    toc(gen_Xoshiro.ClassName()+"Fill_Normal");
    
    tic(gen_PCG.ClassName()+"Fill_Normal");
    gen_PCG.Fill_Normal();
    toc(gen_PCG.ClassName()+"Fill_Normal");
    tic(gen_PCG.ClassName()+"Fill_Normal");
    gen_PCG.Fill_Normal();
    toc(gen_PCG.ClassName()+"Fill_Normal");
    
    tic(gen_Xoshiro.ClassName()+"Fill_Uniform");
    gen_Xoshiro.Fill_Uniform();
    toc(gen_Xoshiro.ClassName()+"Fill_Uniform");
    tic(gen_Xoshiro.ClassName()+"Fill_Uniform");
    gen_Xoshiro.Fill_Uniform();
    toc(gen_Xoshiro.ClassName()+"Fill_Uniform");
    
    tic(gen_PCG.ClassName()+"Fill_Uniform");
    gen_PCG.Fill_Uniform();
    toc(gen_PCG.ClassName()+"Fill_Uniform");
    tic(gen_Xoshiro.ClassName()+"Fill_Uniform");
    gen_PCG.Fill_Uniform();
    toc(gen_PCG.ClassName()+"Fill_Uniform");
    
//    dump(b[0]);
//    dump(b[1]);
//    dump(b[2]);
//    dump(b[3]);
//    
//    dump(b[n-4]);
//    dump(b[n-3]);
//    dump(b[n-2]);
//    dump(b[n-1]);
    
}
