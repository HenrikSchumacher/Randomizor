#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Accelerate/Accelerate.h>

#include "../Tools/Tools.hpp"
#include "Helpers.hpp"

namespace Randomizor
{
    using namespace Tools;
    
    // A base class for all Metal-based samplers in this library.
    class Randomizor_Metal
    {
    public:
        
        static constexpr auto Managed = MTL::ResourceStorageModeManaged;
        
        // Each thread will handle these many floats at the same time.
        // This allows the GPU to write in 32-Byte chunks.
        static constexpr size_t sample_chunk_size = 4;
        
        const NS::Integer threads_per_device;
        
        const NS::Integer threads_per_threadgroup;
        
        const size_t OMP_thread_count = 1;
        
        explicit Randomizor_Metal(
            NS::SharedPtr<MTL::Device> & device_,
            NS::Integer threads_per_device_ = 24576*4,   // for M1 Max with 32 cores.
            NS::Integer threads_per_threadgroup_ = 1024, // for M1
            size_t      OMP_thread_count_ = 8            // for M1 Max; only performance cores
        )
        :   threads_per_device      ( threads_per_device_      )
        ,   threads_per_threadgroup ( threads_per_threadgroup_ )
        ,   OMP_thread_count        ( OMP_thread_count_        )
        ,   device                  ( device_                  )
        {
            command_queue = NS::TransferPtr(device->newCommandQueue());
            
            if( command_queue.get() == nullptr )
            {
                eprint(ClassName()+": Failed to find the command queue." );
                return;
            }
        }
        
        ~Randomizor_Metal()
        {
            pipelines = std::map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> ();
        }
        
        
    protected:
        
        NS::SharedPtr<MTL::Device> device;
        
        std::map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pipelines;
        
        NS::SharedPtr<MTL::CommandQueue> command_queue;
        
        NS::SharedPtr<MTL::Buffer> reservoir;
        
        size_t reservoir_size = 0;
        
        NS::SharedPtr<MTL::Buffer> states;
        
    protected:
        
        
        void CompilePipeline(
            const std::string & fun_name,                 // name of function in code string
            const std::string & code,                     // string of actual Metal code
            const std::vector<std::string> & param_types, // types of compile-time parameters (converted to string)
            const std::vector<std::string> & param_names, // name of compile-time parameters
            const std::vector<std::string> & param_vals   // values of compile-time parameters
        )
        {
            std::string fun_fullname = FullPipelineName(fun_name,param_vals);
            
            std::string tag = ClassName()+"::CompilePipeline(" + fun_fullname + ")";
            
            ptic(tag);
            
            std::stringstream full_code;
            
            if( param_types.size() != param_names.size() )
            {
                eprint(tag+": param_types.size() != param_names.size().");
                ptoc(tag);
                std::exit(-1);
            }
            
            if( param_types.size() != param_vals.size() )
            {
                eprint(tag+": param_types.size() != param_vals.size().");
                ptoc(tag);
                std::exit(-1);
            }
            
            std::size_t param_count = param_types.size();
            
            // Create compile-time constant. Will be prependend to code string.
            for( std::size_t i = 0; i < param_count; ++i )
            {
                full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
            }
            
            full_code << code;
            
            NS::SharedPtr<NS::String> code_NS_String = NS::TransferPtr( NS::String::string(full_code.str().c_str(), NS::StringEncoding::UTF8StringEncoding) );
            
            NS::Error * error = nullptr;
            
            NS::SharedPtr<MTL::Library> lib = NS::TransferPtr(
                device->newLibrary(
                    code_NS_String.get(),
                    nullptr, // <-- for distinguishing from the function that loads from file
                    &error
                )
            );
            
            if( lib.get() == nullptr )
            {
                eprint(tag+": Failed to compile library from string for function.");
                valprint("Error message", error->description()->utf8String() );
                std::exit(-1);
            }
            
            bool found = false;
            
            // Go through all functions in the library to find ours.
            for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
            {
                auto name_nsstring = lib->functionNames()->object(i)->description();
                
                if( fun_name == name_nsstring->utf8String() )
                {
                    found = true;
                    
                    
                    // This MTL::Function object is needed only temporarily.
                    NS::SharedPtr<MTL::Function> fun = NS::TransferPtr(lib->newFunction(name_nsstring));
                    
                    // Create pipeline from function.
                    pipelines[fun_fullname] = NS::TransferPtr(device->newComputePipelineState(fun.get(), &error));
                    
                    if( pipelines[fun_fullname].get() == nullptr )
                    {
                        eprint(tag+": Failed to created pipeline state object.");
                        valprint("Error message", error->description()->utf8String() );
                        std::exit(-1);
                    }
                }
            }
            
            if( found )
            {
                ptoc(tag);
            }
            else
            {
                eprint(tag+": Metal kernel not found in source code.");
                ptoc(tag);
                std::exit(-1);
            }
        }
        
        std::string FullPipelineName(
            const std::string & fun_name,                 // name of function in code string
            const std::vector<std::string> & param_vals   // values of compile-time parameters
        )
        {
            std::stringstream fun_fullname_stream;
            
            fun_fullname_stream << fun_name;
            
            for( const auto & s : param_vals )
            {
                fun_fullname_stream << "_" << s;
            }
            
            return fun_fullname_stream.str();
        }
        
        NS::SharedPtr<MTL::ComputePipelineState> GetPipeline(
            const std::string & fun_name,                 // name of function in code string
            const std::vector<std::string> & param_types, // types of compile-time parameters (converted to string)
            const std::vector<std::string> & param_names, // name of compile-time parameters
            const std::vector<std::string> & param_vals   // values of compile-time parameters
        )
        {
            std::string fun_fullname = FullPipelineName(fun_name,param_vals);
            
            std::string tag = ClassName()+"::GetPipeline(" + fun_fullname + ")";
            
            ptic(tag);
            
            if( pipelines.count(fun_fullname) == 0 )
            {
                eprint(tag+": Pipeline not found.");
                ptoc(tag);
                std::exit(-1);
            }
            else
            {
                ptoc(tag);
                return pipelines[fun_fullname];
            }
        }
        
    protected:
        
        virtual void Seed() = 0;
        
        virtual void Compile() = 0;
        
        virtual NS::Integer SampleChunkSize() const = 0;
        
        virtual void Fill_Uniform() = 0;
        
        virtual void Fill_Normal() = 0;
        
    public:
        
        NS::Integer ReservoirSize( const size_t n )
        {
            
            const size_t samples_per_thread = SampleChunkSize() * (n + threads_per_device - 1) / (SampleChunkSize() * threads_per_device);
            
            reservoir_size = samples_per_thread * threads_per_device;
            
//            valprint("requested resevoir size",n);
//            valprint("samples_per_thread     ",samples_per_thread);
//            valprint("allocated resevoir size",reservoir_size);
            
            return reservoir_size;
        }
        
        NS::Integer ReservoirSize() const
        {            
            return reservoir_size;
        }
        
        void RequireReservoir( const size_t n )
        {
            reservoir = NS::TransferPtr(
                device->newBuffer(ReservoirSize(n) * sizeof(float), Managed)
            );
        }
        
        void LoadReservoir( float * external_reservoir, const size_t external_size )
        {
            NS::Integer internal_size = ReservoirSize(external_size);
            
            if( internal_size == external_size )
            {
                reservoir = NS::TransferPtr(
                    device->newBuffer(
                        external_reservoir,
                        internal_size * sizeof(float),
                        Managed
                    )
                );
            }
            else
            {
                eprint(ClassName()+"::LoadReservoir: ReservoirSize(external_size) != external_size. Please allocate memory for ReservoirSize(external_size) floats.");
            }
        }
        
        float * Reservoir()
        {
            return reinterpret_cast<float *>(reservoir->contents());
        }

        void RequireSeed()
        {
            if( states->length() <= 0 )
            {
                this->Seed();
            }
        }
        
        void RequirePipeline()
        {
            if( pipelines.size() <= 0 )
            {
                this->Compile();
            }
        }
        
    protected:

        void RandomizeReservoir( const std::string & name )
        {
            
            RequirePipeline();
            
            RequireSeed();
            
            const size_t n = reservoir->length() / sizeof(float);
            
            if( n <= 0 )
            {
                eprint(ClassName()+"::RandomizeReservoir: Empty reservoir. Create a reservoir with RequireReservoir or with LoadReservoir.");
                return;
            }

            const size_t chunks_per_grid =  n / SampleChunkSize();

            NS::SharedPtr<MTL::ComputePipelineState> pipeline = GetPipeline( name, {},{},{} );
            assert( pipeline.get() != nullptr );

            const NS::Integer max_threads_per_gp = pipeline->maxTotalThreadsPerThreadgroup();

            if( threads_per_threadgroup > max_threads_per_gp )
            {
                eprint("Too many threads per threadgroup requested.");
                valprint("threads per threadgroup requested", threads_per_threadgroup );
                valprint("threads per threadgroup allowed  ", max_threads_per_gp      );
            }


            // Now we can proceed to set up the MTL::CommandBuffer.
            // Create a command buffer to hold commands.
            NS::SharedPtr<MTL::CommandBuffer> command_buffer = NS::TransferPtr(command_queue->commandBuffer());
            assert( command_buffer.get() != nullptr );

            // Create an encoder that translates our command to something the
            // device understands
            NS::SharedPtr<MTL::ComputeCommandEncoder> compute_encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
            assert( compute_encoder.get() != nullptr );

            // Encode the pipeline state object and its parameters.
            compute_encoder->setComputePipelineState( pipeline.get() );

            // Place data in encoder
            compute_encoder->setBuffer(states.get(),    0, 0 );
            compute_encoder->setBuffer(reservoir.get(), 0, 1 );
            compute_encoder->setBytes(&chunks_per_grid, sizeof(size_t), 2 );
            
            MTL::Size th_per_tg (threads_per_threadgroup,1,1);
            MTL::Size tg_per_gr (threads_per_device / threads_per_threadgroup,1,1);
            
            // Encode the compute command.
            compute_encoder->dispatchThreadgroups(tg_per_gr,th_per_tg);
            
            // Signal that we have encoded all we want.
            compute_encoder->endEncoding();
            
            // Encode synchronization of return buffers.
            MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
            assert( blit_command_encoder != nullptr );
            
            blit_command_encoder->synchronizeResource(states.get());
            blit_command_encoder->synchronizeResource(reservoir.get());
            blit_command_encoder->endEncoding();
            
            // Execute the command buffer.
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
        }
        
    public:
        
        virtual std::string ClassName() const
        {
            return "Randomizor_Metal";
        }
        
    };
}
