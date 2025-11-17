include(FetchContent)
function(FetchAndAddFmt USE_FETCH)
  # Normalize input to uppercase (ON/OFF)
  string(TOUPPER "${USE_FETCH}" USE_FETCH_UC)

  if(USE_FETCH_UC STREQUAL "ON")
    message(STATUS "[AMS] Fetching vendored fmt…")
    set(FMT_INSTALL ON CACHE BOOL "" FORCE)
    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF CACHE BOOL "" FORCE)
    include(FetchContent)

    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG        12.1.0
    )

    # Actually download + add the subproject
    FetchContent_MakeAvailable(fmt)

    # fmt creates a target called fmt::fmt
    # We export that as-is
    message(STATUS "[AMS] Using vendored fmt::fmt")

  else()
    message(STATUS "[AMS] Looking for system fmt…")

    find_package(fmt QUIET)

    if(NOT fmt_FOUND)
      message(FATAL_ERROR
        "[AMS] fmt was not found in the system! "
        "Pass ON to FetchAndAddFmt to fetch it automatically."
      )
    endif()

    message(STATUS "[AMS] Using system fmt::fmt")
  endif()

endfunction()
