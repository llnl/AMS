#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#define CALIPER(stmt) stmt
#else
#define CALIPER(stmt)
#endif

#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#define MPI_CALL(stmt)                                                         \
  if (stmt != MPI_SUCCESS) {                                                   \
    fprintf(stderr, "Error in MPI-Call (File: %s, %d)\n", __FILE__, __LINE__); \
  }
#else
typedef void *MPI_Comm;
#define MPI_CALL(stm)
#endif

#ifdef __AMS_ENABLE_PERFFLOWASPECT__
#define PERFFASPECT() __attribute__((annotate("@critical_path()")))
#else
#define PERFFASPECT()
#endif
