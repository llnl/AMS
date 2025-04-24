#pragma once

namespace ams
{
typedef enum { AMS_SINGLE = 0, AMS_DOUBLE, AMS_UNKNOWN_TYPE } AMSDType;

typedef enum {
  AMS_UNKNOWN = -1,
  AMS_HOST = 0,
  AMS_DEVICE = 1,
  AMS_PINNED = 2,
  AMS_RSEND
} AMSResourceType;

typedef enum { AMS_UBALANCED = 0, AMS_BALANCED } AMSExecPolicy;

typedef enum { AMS_NONE = 0, AMS_HDF5, AMS_RMQ } AMSDBType;

enum struct AMSUQPolicy {
  AMS_UQ_BEGIN = 0,
  AMS_DELTAUQ_MEAN,
  AMS_DELTAUQ_MAX,
  AMS_RANDOM,
  AMS_UQ_END
};

}  // namespace ams
