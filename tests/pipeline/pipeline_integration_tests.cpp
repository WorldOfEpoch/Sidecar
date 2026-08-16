#include "sidecar/pipeline/physics.hpp"
#include "sidecar/storage/physics.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void Expect(bool value,const char* message){if(!value)throw std::runtime_error(message);}
}

int main(){
    try{
        auto provider=sidecar::pipeline::CreateNativePipelineProvider();
        if(!provider->SupportsCuda()){std::cout<<"[SKIP] CUDA disabled\n";return 0;}
        sidecar::pipeline::HostCopyConfiguration hc;hc.bytes=1ULL<<20U;hc.warmups=1;hc.repetitions=1;
        const auto copy=provider->RunHostCopy(hc);Expect(copy.status==sidecar::pipeline::PipelineStatus::Success,"small pageable->pinned copy");
        auto storage=sidecar::storage::CreateNativeStorageProvider();auto target=storage->ResolveTarget(std::nullopt,std::nullopt);
        for(const auto path:{sidecar::pipeline::PipelineType::NvmePageablePinnedH2D,sidecar::pipeline::PipelineType::NvmePinnedH2D}){
            sidecar::pipeline::PipelineConfiguration c;c.pipeline_type=path;c.aggregate_bytes=4ULL<<20U;c.chunk_bytes=2ULL<<20U;
            c.buffer_depth=2;c.compute_window_us=1000;c.deadline_ns=16'000'000;c.repetitions=1;c.phase=sidecar::pipeline::PipelinePhase::Validate;
            c.contention_test=path==sidecar::pipeline::PipelineType::NvmePageablePinnedH2D?
                sidecar::pipeline::ContentionTest::I:sidecar::pipeline::ContentionTest::J;
            const auto result=provider->Run(target,c);Expect(result.status==sidecar::pipeline::PipelineStatus::Success,"small native pipeline");
            Expect(result.samples.size()==1&&result.samples[0].verified,"end-to-end pipeline verification");
            Expect(result.slots.size()==2&&result.stages.size()>=4,"double-buffered stage evidence");
        }
        sidecar::pipeline::PipelineConfiguration cancelled;cancelled.aggregate_bytes=4ULL<<20U;cancelled.chunk_bytes=2ULL<<20U;
        cancelled.buffer_depth=2;cancelled.compute_window_us=1000;cancelled.repetitions=1;cancelled.cancel_after_blocks=0;
        const auto cancellation=provider->Run(target,cancelled);Expect(cancellation.status==sidecar::pipeline::PipelineStatus::Cancelled,"pipeline cancellation");
        std::cout<<"[PASS] WU8 native correctness integration\n";return 0;
    }catch(const std::exception&e){std::cerr<<"[FAIL] "<<e.what()<<'\n';return 1;}
}
