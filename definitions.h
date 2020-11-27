#ifndef DEMOTRACK_CUDA_DEFINITIONS_H__
#define DEMOTRACK_CUDA_DEFINITIONS_H__

#include <cuda_runtime.h>

#if defined( __APPLE__ ) && __APPLE__
    #include <TargetConditionals.h>
#endif /* defined( __APPLE__ ) */

#if !defined( DEMOTRACK_DEVICE_FN )
    #define DEMOTRACK_DEVICE_FN __device__
#endif /* !defined( DEMOTRACK_DEVICE_FN ) */

#if !defined( DEMOTRACK_HOST_FN )
    #define DEMOTRACK_HOST_FN __host__
#endif /* !defined( DEMOTRACK_HOST_FN ) */

#if !defined( DEMOTRACK_FN )
    #define DEMOTRACK_FN __host__ __device__
#endif /* !defined( DEMOTRACK_FN ) */

namespace demotrack
{
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        using int32_type = long int ;
        using uint32_type = unsigned long int;
        using int64_type = long long int ;
        using uint64_type = unsigned long long int ;
    #elif __APPLE__ && defined( TARGET_OS_MAC )
        using int32_type = long int ;
        using uint32_type = unsigned long int;

        using int64_type = long long int ;
        using uint64_type = unsigned long long int ;
    #elif __linux__
        using int32_type = int ;
        using uint32_type = unsigned int;

        using int64_type = long long int ;
        using uint64_type = unsigned long long int ;
    #else
        #error "Unknown platform"
    #endif

    using size_type = uint64_type;
}

#endif /* DEMOTRACK_CUDA_DEFINITIONS_H__ */
