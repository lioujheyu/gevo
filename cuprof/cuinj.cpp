/*
 * Copyright 2021 NVIDIA Corporation. All rights reserved.
 *
 * CUPTI based tracing injection to trace any CUDA application.
 * This sample demonstrates how to use activity
 * and callback APIs in the injection code.
 * Refer to the README.txt file for usage.
 *
 * Workflow in brief:
 *
 *  After the initialization routine returns, the application resumes running,
 *  with the registered callbacks triggering as expected.
 *  Subscribed to ProfilerStart and ProfilerStop callbacks. These callbacks
 *  control the collection of profiling data.
 *
 *  ProfilerStart callback:
 *      Start the collection by enabling activities. Also enable callback for
 *      the API cudaDeviceReset to flush activity buffers.
 *
 *  ProfilerStop callback:
 *      Get all the activity buffers which have all the activity records completed
 *      by using cuptiActivityFlushAll() API and then disable cudaDeviceReset callback
 *      and all the activities to stop collection.
 *
 *  atExitHandler:
 *      Register to the atexit handler to get all the activity buffers including the ones
 *      which have incomplete activity records by using force flush API
 *      cuptiActivityFlushAll(1).
 */
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>

#include <cxxabi.h>
#include <unistd.h>

#include <cuda.h>
#include <cupti.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include "detours.h"
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

// Variable related to initialize injection .
std::mutex initializeInjectionMutex;

// value order in the vector: Calls, Total_time, Min_time, Max_time
std::map<std::string, std::vector<uint64_t>> kernels;

// Macros
#define IS_ACTIVITY_SELECTED(activitySelect, activityKind)                               \
    (activitySelect & (1LL << activityKind))

#define SELECT_ACTIVITY(activitySelect, activityKind)                                    \
    (activitySelect |= (1LL << activityKind))

#define CUPTI_CALL(call)                                                                 \
    do {                                                                                 \
        CUptiResult _status = call;                                                      \
        if (_status != CUPTI_SUCCESS) {                                                  \
            const char *errstr;                                                          \
            cuptiGetResultString(_status, &errstr);                                      \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",         \
                    __FILE__, __LINE__, #call, errstr);                                  \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

#define BUF_SIZE (8 * 1024 * 1024) // 8MB
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                      \
    (((uintptr_t)(buffer) & ((align)-1))                                                 \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))                    \
         : (buffer))
#define CLASS_NAME(somePointer) ((const char *) cppDemangle(somePointer).get() )

// Global Structure
typedef struct {
    volatile uint32_t initialized;
    CUpti_SubscriberHandle subscriber;
    int tracingEnabled;
    uint64_t profileMode;
} injGlobalControl;

injGlobalControl globalControl;

// Function Declarations

std::shared_ptr<char> cppDemangle(const char *abiName);
static CUptiResult cuptiInitialize(void);
static void atExitHandler(void);
void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, void *cbInfo);

// Function Definitions

std::shared_ptr<char> cppDemangle(const char *abiName)
{
  int status;    
  char *ret = abi::__cxa_demangle(abiName, 0, 0, &status);  

  /* NOTE: must free() the returned char when done with it! */
  std::shared_ptr<char> retval;
  retval.reset( (char *)ret, [](char *mem) { if (mem) free((void*)mem); } );
  return retval;
}

static void globalControlInit(void) {
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.tracingEnabled = 0;
    globalControl.profileMode = 0;
}

void registerAtExitHandler(void) {
    atexit(&atExitHandler);
}

static void atExitHandler(void) {
    CUPTI_CALL(cuptiGetLastError());

    // Force flush
    if (globalControl.tracingEnabled) {
        CUPTI_CALL(cuptiActivityFlushAll(1));
    }

    fprintf(stderr, "\n=== [cuprof result] ===\n");
    
    if (kernels.size() == 0) {
        printf("No kernel is profiled.");
        exit(-1);
    }

    /**
     * Accumulate kernel execution time
     */
    uint64_t total_kernel_time = 0;
    std::vector<std::pair<std::string, std::vector<uint64_t>>> kernel_vec;
    for (auto const& kernel : kernels) {
        total_kernel_time += kernel.second[1];
        kernel_vec.push_back(std::make_pair(CLASS_NAME(kernel.first.c_str()), kernel.second));
    }

    // sorted from the longest kernel to the shortest
    std::sort(kernel_vec.begin(), kernel_vec.end(), 
              [=](std::pair<std::string, std::vector<uint64_t>> a, 
                  std::pair<std::string, std::vector<uint64_t>> b){
                    return a.second[1] > b.second[1];
    });

    for (auto const& kernel : kernel_vec) {
        // mimic nvprof's default print behavior in csv format and micro second (us)): 
        //   "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
        fprintf(stderr, "\"GPU activities\",%f,%f,%llu,%f,%f,%f,\"%s\"\n",
            (float)kernel.second[1]/total_kernel_time*100,
            (float)kernel.second[1]/1000,
            (long long unsigned int)kernel.second[0],
            (float)kernel.second[1]/1000/kernel.second[0],
            (float)kernel.second[2]/1000,
            (float)kernel.second[3]/1000,
            kernel.first.c_str());
    }
}

static CUptiResult unsubscribeAllCallbacks(void) {
    if (globalControl.subscriber) {
        CUPTI_CALL(cuptiEnableAllDomains(0, globalControl.subscriber));
        CUPTI_CALL(cuptiUnsubscribe(globalControl.subscriber));
        globalControl.subscriber = NULL;
    }
    return CUPTI_SUCCESS;
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                     size_t *maxNumRecords) {
    uint8_t *rawBuffer;

    *size = BUF_SIZE;
    rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL) {
        printf("Error: Out of memory.\n");
        exit(-1);
    }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                                     size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;
    size_t dropped;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            CUpti_ActivityKind kind = record->kind;

            switch (kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
                uint64_t new_kernel_time = kernel->end - kernel->start;
                if (kernels.count(kernel->name) == 0) {
                    kernels[kernel->name] = std::vector<uint64_t>{
                        (uint64_t)1, new_kernel_time, new_kernel_time, new_kernel_time};
                }
                else {
                    uint64_t min_time = kernels[kernel->name][2];
                    uint64_t max_time = kernels[kernel->name][3];
                    kernels[kernel->name][0]++; // count
                    kernels[kernel->name][1] += new_kernel_time; // total time
                    if (min_time > new_kernel_time)
                        kernels[kernel->name][2] = new_kernel_time;
                    if (max_time < new_kernel_time)
                        kernels[kernel->name][3] = new_kernel_time;
                }
                break;
            }
            default:
                break;
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            CUPTI_CALL(status);
        }
    } while (1);

    // Report any records dropped from the queue
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
        printf("Dropped %u activity records.\n", (unsigned int)dropped);
    }
    free(buffer);
}

static CUptiResult selectActivities() {
    SELECT_ACTIVITY(globalControl.profileMode, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    return CUPTI_SUCCESS;
}

static CUptiResult enableCuptiActivity(CUcontext ctx) {
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));
    CUPTI_CALL(selectActivities());

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i) {
        if (IS_ACTIVITY_SELECTED(globalControl.profileMode, i)) {
            // If context is NULL activities are being enabled after CUDA initialization
            // else the activities are being enabled on cudaProfilerStart API
            if (ctx == NULL) {
                CUPTI_CALL(cuptiActivityEnable((CUpti_ActivityKind)i));
            } else {
                // Since some activities are not supported at context mode, enable them in
                // global mode if context mode fails
                result = cuptiActivityEnableContext(ctx, (CUpti_ActivityKind)i);

                if (result == CUPTI_ERROR_INVALID_KIND) {
                    cuptiGetLastError();
                    result = cuptiActivityEnable((CUpti_ActivityKind)i);
                } else if (result != CUPTI_SUCCESS) {
                    CUPTI_CALL(result);
                }
            }
        }
    }

    return result;
}

static CUptiResult cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber,
                              (CUpti_CallbackFunc)callbackHandler, NULL));

    // Subscribe Driver  callback to call onProfilerStartstop function
    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(cuptiEnableCallback(1, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuProfilerStop));

    // Enable CUPTI activities
    CUPTI_CALL(enableCuptiActivity(NULL));

    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    return status;
}

static CUptiResult onCudaDeviceReset(void) {
    // Flush all queues
    CUPTI_CALL(cuptiActivityFlushAll(0));

    return CUPTI_SUCCESS;
}

static CUptiResult onProfilerStart(CUcontext context) {
    if (context == NULL) {
        // Don't do anything if context is NULL
        return CUPTI_SUCCESS;
    }

    CUPTI_CALL(enableCuptiActivity(context));

    return CUPTI_SUCCESS;
}

static CUptiResult disableCuptiActivity(CUcontext ctx) {
    CUptiResult result = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiEnableCallback(0, globalControl.subscriber,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));

    for (int i = 0; i < CUPTI_ACTIVITY_KIND_COUNT; ++i) {
        if (IS_ACTIVITY_SELECTED(globalControl.profileMode, i)) {
            // Since some activities are not supported at context mode, disable them in
            // global mode if context mode fails
            result = cuptiActivityDisableContext(ctx, (CUpti_ActivityKind)i);

            if (result == CUPTI_ERROR_INVALID_KIND) {
                cuptiGetLastError();
                CUPTI_CALL(cuptiActivityDisable((CUpti_ActivityKind)i));
            } else if (result != CUPTI_SUCCESS) {
                CUPTI_CALL(result);
            }
        }
    }

    return CUPTI_SUCCESS;
}

static CUptiResult onProfilerStop(CUcontext context) {
    if (context == NULL) {
        // Don't do anything if context is NULL
        return CUPTI_SUCCESS;
    }

    CUPTI_CALL(cuptiActivityFlushAll(0));
    CUPTI_CALL(disableCuptiActivity(context));

    return CUPTI_SUCCESS;
}

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, void *cbdata) {
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;

    // Check last error
    CUPTI_CALL(cuptiGetLastError());

    switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API: {
        switch (cbid) {
        // case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
        //     /* We start profiling collection on exit of the API. */
        //     if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        //         onProfilerStart(cbInfo->context);
        //     }
        //     break;
        // }
        // case CUPTI_DRIVER_TRACE_CBID_cuProfilerStop: {
        //     /* We stop profiling collection on entry of the API. */
        //     if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        //         onProfilerStop(cbInfo->context);
        //     }
        //     break;
        // }
        default:
            break;
        }
        break;
    }
    case CUPTI_CB_DOMAIN_RUNTIME_API: {
        switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020: {
            if (cbInfo->callbackSite == CUPTI_API_ENTER) {
                CUPTI_CALL(onCudaDeviceReset());
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

#ifdef _WIN32
extern "C" __declspec(dllexport) int InitializeInjection(void)
#else
extern "C" int InitializeInjection(void)
#endif
{
    if (globalControl.initialized) {
        return 1;
    }

    initializeInjectionMutex.lock();

    // Init globalControl
    globalControlInit();

    registerAtExitHandler();

    // Initialize CUPTI
    if (cuptiInitialize() != CUPTI_SUCCESS) {
        printf("Error: Cupti Initilization failed.\n");
        unsubscribeAllCallbacks();
        exit(EXIT_FAILURE);
    }
    globalControl.tracingEnabled = 1;
    globalControl.initialized = 1;
    initializeInjectionMutex.unlock();
    return 1;
}
