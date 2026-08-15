include(FetchContent)

function(FetchAndAddCatch2 FETCH_CATCH2)
  string(TOUPPER "${FETCH_CATCH2}" FETCH_CATCH2_UC)

  if(FETCH_CATCH2_UC STREQUAL "ON")
    message(STATUS "AMS: Fetching Catch2 via FetchContent")
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "Disable Catch2 tests" FORCE)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Disable Catch2 docs" FORCE)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.11.0
    )

    FetchContent_MakeAvailable(Catch2)
  else()
    message(STATUS "AMS: Using system-provided Catch2")
    find_package(Catch2 CONFIG REQUIRED)
  endif()
endfunction()
