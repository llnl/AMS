include(FetchContent)
function(FetchAndAddCatch FETCH_CATCH)
  string(TOUPPER "${FETCH_CATCH}" USE_CATCH_UC)
  if(USE_CATCH_UC STREQUAL "ON")
    message(STATUS "[AMS] Fetching Catch")
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.6.0
    )
    FetchContent_MakeAvailable(Catch2)
    set(CMAKE_MODULE_PATH
        "${CMAKE_MODULE_PATH};${Catch2_SOURCE_DIR}/extras"
        PARENT_SCOPE)
    include(Catch)
  endif()
endfunction()
