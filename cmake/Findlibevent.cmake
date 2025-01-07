# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# LIBEVENT_FOUND - System has LibEvent
# LIBEVENT_INCLUDE_DIR - the LibEvent include directory
# LIBEVENT_LIBRARIES 0 The libraries needed to use LibEvent
find_path     (LIBEVENT_INCLUDE_DIR NAMES event.h HINTS ${AMS_LIBEVENT_HINTS})
find_library  (LIBEVENT_LIBRARY     NAMES event HINTS ${AMS_LIBEVENT_HINTS} NO_CMAKE_SYSTEM_PATH)

if(NOT LIBEVENT_LIBRARY)
    find_library(LIBEVENT_LIBRARY NAMES event)
endif()

find_library  (LIBEVENT_CORE NAMES event_core HINTS ${AMS_LIBEVENT_HINTS} NO_CMAKE_SYSTEM_PATH)
if(NOT LIBEVENT_CORE)
  find_library(LIBEVENT_CORE NAMES event_core)
endif()


find_library  (LIBEVENT_EXTRA       NAMES event_extra HINTS ${AMS_LIBEVENT_HINTS} NO_CMAKE_SYSTEM_PATH)
if(NOT LIBEVENT_EXTRA)
  find_library(LIBEVENT_EXTRA NAMES event_extra)
endif()

if (NOT EVHTP_DISABLE_EVTHR)
  find_library (LIBEVENT_THREAD NAMES event_pthreads HINTS ${AMS_LIBEVENT_HINTS} NO_CMAKE_SYSTEM_PATH)
  if(NOT LIBEVENT_THREAD)
    find_library(LIBEVENT_THREAD NAMES event_pthreads)
  endif()
endif()

if (NOT EVHTP_DISABLE_SSL)
  find_library (LIBEVENT_SSL  NAMES event_openssl HINTS ${AMS_LIBEVENT_HINTS} NO_CMAKE_SYSTEM_PATH)
  if(NOT LIBEVENT_SSL)
    find_library(LIBEVENT_SSL NAMES event_openssl)
  endif()
endif()

include (FindPackageHandleStandardArgs)
set (LIBEVENT_INCLUDE_DIRS ${LIBEVENT_INCLUDE_DIR})
set (LIBEVENT_LIBRARIES
        ${LIBEVENT_LIBRARY}
        ${LIBEVENT_SSL}
        ${LIBEVENT_CORE}
        ${LIBEVENT_EXTRA}
        ${LIBEVENT_THREAD}
        ${LIBEVENT_EXTRA})
    find_package_handle_standard_args (libevent DEFAULT_MSG LIBEVENT_LIBRARIES LIBEVENT_INCLUDE_DIR)
mark_as_advanced(LIBEVENT_INCLUDE_DIRS LIBEVENT_LIBRARIES)

