# Automatically configure git hooks when the project is configured.
# Place in cmake/setup-git-hooks.cmake and include from the top-level CMakeLists.txt.

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Only set if not already configured (respects developer overrides)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} config --get core.hooksPath
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE _current_hooks_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
  )

  if(NOT _rc EQUAL 0 OR NOT _current_hooks_path STREQUAL ".githooks")
    message(STATUS "Setting git hooks path to .githooks/")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} config core.hooksPath .githooks
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif()
endif()
