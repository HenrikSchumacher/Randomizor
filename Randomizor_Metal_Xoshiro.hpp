#pragma once

#include "src/Randomizor_Metal.hpp"

#include "src/SplitMix64.hpp"
#include "src/Xoshiro256Plus.hpp"

namespace Randomizor
{
    class Randomizor_Metal_Xoshiro : public Randomizor_Metal
    {
    public:
        
        using NS::StringEncoding::UTF8StringEncoding;
        
        using UInt        = typename Xoshiro256Plus::UInt;
        using state_type  = typename Xoshiro256Plus::state_type;
        using result_type = float;
        
        
        explicit Randomizor_Metal_Xoshiro(
            NS::SharedPtr<MTL::Device> & device_,
            NS::Integer threads_per_device_ = 24576*4,  // for M1 Max with 32 cores.
            NS::Integer threads_per_threadgroup_ = 1024 // for M1
        )
        :   Randomizor_Metal( device_, threads_per_device_, threads_per_threadgroup_ )
        {}
        
        ~Randomizor_Metal_Xoshiro() = default;
        
        
    protected:
        
        using Randomizor_Metal::device;
        using Randomizor_Metal::pipelines;
        using Randomizor_Metal::command_queue;
        using Randomizor_Metal::reservoir;
        using Randomizor_Metal::reservoir_size;
        using Randomizor_Metal::states;
        using Randomizor_Metal::OMP_thread_count;
        
    protected:
        
        virtual NS::Integer SampleChunkSize() const override
        {
            return 4;
        }
        
        virtual void Seed() override
        {
            ptic(ClassName()+"::Seed");
            
            states = NS::TransferPtr(
                 device->newBuffer( threads_per_device * 4 * sizeof(uint64_t), Managed )
            );
            
            uint64_t * restrict states_ptr = reinterpret_cast<uint64_t *>(states->contents());
            
            std::random_device r;
            
            state_type seed;
            {
                std::uint32_t* seed_ = reinterpret_cast<std::uint32_t*>(&seed);
                for( int i = 0; i < 8; ++i )
                {
                    seed_[i] = r();
                }
            }
            
            // Create the actual random engine.
            Xoshiro256Plus seeder ( seed );
            
            std::vector<state_type> seeder_states ( OMP_thread_count);
            
            for( size_t i = 0; i < OMP_thread_count; ++i )
            {
                seeder.LongJump();
                seeder_states[i] = seeder.State();
            }
            
            #pragma omp parallel for num_threads(OMP_thread_count)
            for( size_t thread = 0; thread < OMP_thread_count; ++thread )
            {
                // Create the actual random engine.
                Xoshiro256Plus random_engine ( seeder_states[thread] );
                
                std::array<uint64_t,4> s;
                
                const size_t i_begin = JobPointer<size_t>(threads_per_device,OMP_thread_count,thread  );
                const size_t i_end   = JobPointer<size_t>(threads_per_device,OMP_thread_count,thread+1);
                
                for( size_t i = i_begin; i < i_end; ++i )
                {
                    random_engine.Jump();
                    s = random_engine.State();
                    states_ptr[4*i+0] = s[0];
                    states_ptr[4*i+1] = s[1];
                    states_ptr[4*i+2] = s[2];
                    states_ptr[4*i+3] = s[3];
                }
            }
            
            states->didModifyRange({0,states->length()});
            
            ptoc(ClassName()+"::Seed");
        }
        
        void Compile() override
        {
            CompilePipeline(
                "Xoshiro256Plus_UniformDistribution",
                std::string(
                #include "src/Randomizor_Metal_Xoshiro/Xoshiro256Plus_UniformDistribution.metal"
                ),
                {},{},{}
            );
            
            CompilePipeline(
                "Xoshiro256Plus_NormalDistribution",
                std::string(
                #include "src/Randomizor_Metal_Xoshiro/Xoshiro256Plus_NormalDistribution.metal"
                ),
                {},{},{}
            );
        }
        
    public:
        
        virtual void Fill_Uniform() override
        {
            ptic(ClassName()+"::Fill_Uniform");
            RandomizeReservoir("Xoshiro256Plus_UniformDistribution");
            ptoc(ClassName()+"::Fill_Uniform");
        }
        
        virtual void Fill_Normal() override
        {
            ptic(ClassName()+"::Fill_Normal");
            RandomizeReservoir("Xoshiro256Plus_NormalDistribution");
            ptoc(ClassName()+"::Fill_Normal");
        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return "Randomizor_Metal_Xoshiro";
        }
        
    };
}
