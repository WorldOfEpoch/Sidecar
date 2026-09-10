add_executable(sidecar_tests unit/sidecar_tests.cpp)
target_link_libraries(sidecar_tests PRIVATE sidecar::core)
target_include_directories(sidecar_tests PRIVATE "${CMAKE_BINARY_DIR}/generated")

if(MSVC)
    target_compile_options(sidecar_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME sidecar.unit COMMAND sidecar_tests)

add_executable(sidecar_trace_tests trace/trace_tests.cpp)
target_link_libraries(sidecar_trace_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_trace_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_trace_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.trace COMMAND sidecar_trace_tests)

add_executable(sidecar_memory_tests memory/memory_tests.cpp)
target_link_libraries(sidecar_memory_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_memory_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_memory_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.memory COMMAND sidecar_memory_tests)

add_executable(sidecar_transfer_tests cuda/transfer_tests.cpp)
target_link_libraries(sidecar_transfer_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_transfer_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_transfer_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.cuda_transfer COMMAND sidecar_transfer_tests)

add_executable(sidecar_overlap_tests cuda/overlap_tests.cpp)
target_link_libraries(sidecar_overlap_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_overlap_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_overlap_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.cuda_overlap COMMAND sidecar_overlap_tests)

add_executable(sidecar_storage_tests storage/storage_tests.cpp)
target_link_libraries(sidecar_storage_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_storage_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_storage_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.storage COMMAND sidecar_storage_tests)

add_executable(sidecar_pipeline_tests pipeline/pipeline_tests.cpp)
target_link_libraries(sidecar_pipeline_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_pipeline_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.pipeline COMMAND sidecar_pipeline_tests)

add_executable(sidecar_llama_tests llama/llama_tests.cpp)
target_link_libraries(sidecar_llama_tests PRIVATE sidecar::core)
if(MSVC)
    target_compile_options(sidecar_llama_tests PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(sidecar_llama_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sidecar.llama COMMAND sidecar_llama_tests)

if(SIDECAR_ENABLE_LLAMA)
    add_custom_command(TARGET sidecar_llama_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SIDECAR_LLAMA_BIN_DIR}/llama.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll"
            "${SIDECAR_LLAMA_BIN_DIR}/ggml-cpu.dll"
            "$<TARGET_FILE_DIR:sidecar_llama_tests>")
    if(SIDECAR_ENABLE_CUDA)
        add_executable(sidecar_llama_integration_tests llama/llama_integration_tests.cpp)
        target_link_libraries(sidecar_llama_integration_tests PRIVATE sidecar::core)
        if(MSVC)
            target_compile_options(sidecar_llama_integration_tests PRIVATE /W4 /permissive- /EHsc)
        endif()
        add_custom_command(TARGET sidecar_llama_integration_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SIDECAR_LLAMA_BIN_DIR}/llama.dll"
                "${SIDECAR_LLAMA_BIN_DIR}/ggml.dll"
                "${SIDECAR_LLAMA_BIN_DIR}/ggml-base.dll"
                "${SIDECAR_LLAMA_BIN_DIR}/ggml-cpu.dll"
                "${SIDECAR_LLAMA_BIN_DIR}/ggml-cuda.dll"
                "$<TARGET_FILE_DIR:sidecar_llama_integration_tests>")
        add_test(NAME sidecar.llama.integration COMMAND sidecar_llama_integration_tests)
    endif()
endif()

if(SIDECAR_ENABLE_CUDA)
    add_executable(sidecar_transfer_integration_tests cuda/transfer_integration_tests.cpp)
    target_link_libraries(sidecar_transfer_integration_tests PRIVATE sidecar::core)
    if(MSVC)
        target_compile_options(sidecar_transfer_integration_tests PRIVATE /W4 /permissive- /EHsc)
    endif()
    add_test(NAME sidecar.cuda_transfer.integration COMMAND sidecar_transfer_integration_tests)

    add_executable(sidecar_overlap_integration_tests cuda/overlap_integration_tests.cpp)
    target_link_libraries(sidecar_overlap_integration_tests PRIVATE sidecar::core)
    if(MSVC)
        target_compile_options(sidecar_overlap_integration_tests PRIVATE /W4 /permissive- /EHsc)
    endif()
    add_test(NAME sidecar.cuda_overlap.integration COMMAND sidecar_overlap_integration_tests)

    add_executable(sidecar_pipeline_integration_tests pipeline/pipeline_integration_tests.cpp)
    target_link_libraries(sidecar_pipeline_integration_tests PRIVATE sidecar::core)
    if(MSVC)
        target_compile_options(sidecar_pipeline_integration_tests PRIVATE /W4 /permissive- /EHsc)
    endif()
    add_test(NAME sidecar.pipeline.integration COMMAND sidecar_pipeline_integration_tests)
endif()

add_test(NAME sidecar.cli.version COMMAND sidecar-lab version)
add_test(NAME sidecar.cli.trace_selftest
         COMMAND sidecar-lab trace selftest --record-size both)

set(SIDECAR_CLI_TEST_TRACE "${CMAKE_CURRENT_BINARY_DIR}/sidecar-cli-test.sidecartrace")
add_test(NAME sidecar.cli.trace_generate
         COMMAND sidecar-lab trace generate-test "${SIDECAR_CLI_TEST_TRACE}")
add_test(NAME sidecar.cli.trace_validate
         COMMAND sidecar-lab trace validate "${SIDECAR_CLI_TEST_TRACE}")
add_test(NAME sidecar.cli.trace_summary
         COMMAND sidecar-lab trace summary "${SIDECAR_CLI_TEST_TRACE}")
add_test(NAME sidecar.cli.trace_summary_json
         COMMAND sidecar-lab trace summary "${SIDECAR_CLI_TEST_TRACE}" --json)
add_test(NAME sidecar.cli.trace_inspect
         COMMAND sidecar-lab trace inspect "${SIDECAR_CLI_TEST_TRACE}")
set_tests_properties(sidecar.cli.trace_generate PROPERTIES FIXTURES_SETUP sidecar_cli_trace)
set_tests_properties(sidecar.cli.trace_validate sidecar.cli.trace_summary
    sidecar.cli.trace_summary_json sidecar.cli.trace_inspect
    PROPERTIES FIXTURES_REQUIRED sidecar_cli_trace)

set(SIDECAR_CLI_TEST_DATABASE
    "${CMAKE_CURRENT_BINARY_DIR}/sidecar-cli-test-wu6-terminal-status.db")
add_test(NAME sidecar.cli.db_init
         COMMAND sidecar-lab db init "${SIDECAR_CLI_TEST_DATABASE}")
add_test(NAME sidecar.cli.db_status
         COMMAND sidecar-lab db status "${SIDECAR_CLI_TEST_DATABASE}")
set_tests_properties(sidecar.cli.db_init PROPERTIES FIXTURES_SETUP sidecar_cli_database)
set_tests_properties(sidecar.cli.db_status PROPERTIES FIXTURES_REQUIRED sidecar_cli_database)
if(SIDECAR_ENABLE_CUDA)
    add_test(NAME sidecar.cli.cuda_overlap_validate_persistence
             COMMAND sidecar-lab cuda overlap validate --repetitions 1
                     --database "${SIDECAR_CLI_TEST_DATABASE}" --json)
    set_tests_properties(sidecar.cli.cuda_overlap_validate_persistence
        PROPERTIES FIXTURES_REQUIRED sidecar_cli_database)
endif()

add_test(NAME sidecar.cli.info_human COMMAND sidecar-lab info --no-persist)
add_test(NAME sidecar.cli.info_json COMMAND sidecar-lab info --json --no-persist)
add_test(NAME sidecar.cli.topology_human COMMAND sidecar-lab topology --no-persist)
add_test(NAME sidecar.cli.topology_json COMMAND sidecar-lab topology --json --no-persist)

add_test(NAME sidecar.cli.memory_info COMMAND sidecar-lab memory info)
add_test(NAME sidecar.cli.memory_info_json COMMAND sidecar-lab memory info --json)
add_test(NAME sidecar.cli.memory_plan
         COMMAND sidecar-lab memory plan --dry-run --method hostalloc)
add_test(NAME sidecar.cli.memory_plan_json
         COMMAND sidecar-lab memory plan --dry-run --method cudaHostAlloc --json)
add_test(NAME sidecar.cli.memory_pageable_small
         COMMAND sidecar-lab memory lifecycle --method pageable --sizes 4M
                 --repetitions 1 --no-persist)
add_test(NAME sidecar.cli.cuda_transfer_info COMMAND sidecar-lab cuda transfer info --json)
add_test(NAME sidecar.cli.cuda_transfer_plan
         COMMAND sidecar-lab cuda transfer plan --dry-run --size 4M --json)
add_test(NAME sidecar.cli.cuda_transfer_report
         COMMAND sidecar-lab cuda transfer report --database
                 "${SIDECAR_CLI_TEST_DATABASE}" --json)
set_tests_properties(sidecar.cli.cuda_transfer_report
    PROPERTIES FIXTURES_REQUIRED sidecar_cli_database)
add_test(NAME sidecar.cli.cuda_overlap_info
         COMMAND sidecar-lab cuda overlap info --json)
add_test(NAME sidecar.cli.cuda_overlap_plan
         COMMAND sidecar-lab cuda overlap plan --dry-run --compute gemm
                 --compute-window 4ms --direction h2d --size 64M --json)
add_test(NAME sidecar.cli.cuda_overlap_report
         COMMAND sidecar-lab cuda overlap report --database
                 "${SIDECAR_CLI_TEST_DATABASE}" --json)
set_tests_properties(sidecar.cli.cuda_overlap_report
    PROPERTIES FIXTURES_REQUIRED sidecar_cli_database)
add_test(NAME sidecar.cli.storage_info
         COMMAND sidecar-lab storage info --json)
add_test(NAME sidecar.cli.storage_plan
         COMMAND sidecar-lab storage plan --dry-run --backend unbuffered
                 --pattern sequential --block 1M --qd 1 --json)
add_test(NAME sidecar.cli.pipeline_info COMMAND sidecar-lab pipeline info --json)
add_test(NAME sidecar.cli.llama_info COMMAND sidecar-lab llama info --json)
add_test(NAME sidecar.cli.llama_authoritative_tensor_guard
         COMMAND sidecar-lab llama baseline --model
                 "C:/Sidecar/test-fixtures/not-a-real-model.gguf"
                 --authoritative --no-tensor-checks --json)
set_tests_properties(sidecar.cli.llama_authoritative_tensor_guard
    PROPERTIES WILL_FAIL TRUE)
add_test(NAME sidecar.cli.llama_shadow COMMAND sidecar-lab llama shadow
         --causal-lead-ns 10 --oracle-lead-ns 100 --pipeline-ns 50 --json)
add_test(NAME sidecar.cli.pipeline_plan
         COMMAND sidecar-lab pipeline plan --dry-run --repetitions 1 --json)
if(SIDECAR_ENABLE_CUDA)
    add_test(NAME sidecar.cli.pipeline_validate
             COMMAND sidecar-lab pipeline validate --path pinned --compute alu
                     --compute-window 1ms --deadline 16ms --no-persist --json)
endif()
if(NOT SIDECAR_ENABLE_CUDA)
    add_test(NAME sidecar.cli.cuda_transfer_unsupported
             COMMAND sidecar-lab cuda transfer run --size 4M --repetitions 1
                     --no-persist --json)
    add_test(NAME sidecar.cli.cuda_overlap_unsupported
             COMMAND sidecar-lab cuda overlap validate --no-persist --json)
    add_test(NAME sidecar.cli.pipeline_unsupported
             COMMAND sidecar-lab pipeline validate --no-persist --json)
endif()
