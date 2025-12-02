include(FetchContent)

function(FetchAndAddTlExpected FETCH_TL_EXPECTED)
  string(TOUPPER "${FETCH_TL_EXPECTED}" FETCH_TL_EXPECTED_UC)

  if(FETCH_TL_EXPECTED_UC STREQUAL "ON")
    message(STATUS "AMS: Fetching tl-expected via FetchContent")
    # --- IMPORTANT: disable tl-expected's own tests & packaging ---
    set(EXPECTED_BUILD_TESTS   OFF CACHE BOOL "Disable TlExpected tests"    FORCE)
    set(EXPECTED_BUILD_PACKAGE OFF CACHE BOOL "Disable TlExpected package"  FORCE)
    FetchContent_Declare(
      tl-expected
      GIT_REPOSITORY https://github.com/TartanLlama/expected.git
      GIT_TAG        v1.3.1
    )

    FetchContent_MakeAvailable(tl-expected)
  else()
    message(STATUS "AMS: Using system-provided tl-expected")
    find_package(tl-expected REQUIRED)
  endif()
endfunction()
