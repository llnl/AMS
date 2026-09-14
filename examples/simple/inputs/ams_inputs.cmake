include_guard(GLOBAL)

function(ams_download_dataset)
  set(options)
  set(oneValueArgs NAME URL SHA256 OUT_FILE)
  cmake_parse_arguments(ADS "${options}" "${oneValueArgs}" "" ${ARGN})

  foreach(req NAME URL SHA256 OUT_FILE)
    if(NOT ADS_${req})
      message(FATAL_ERROR "ams_download_dataset: missing required argument: ${req}")
    endif()
  endforeach()

  get_filename_component(_out_dir "${ADS_OUT_FILE}" DIRECTORY)

  if(EXISTS "${ADS_OUT_FILE}")
    file(SHA256 "${ADS_OUT_FILE}" _have_sha256)
    string(TOLOWER "${_have_sha256}" _have_sha256)
    string(TOLOWER "${ADS_SHA256}" _want_sha256)
    if(_have_sha256 STREQUAL _want_sha256)
      message(STATUS "[inputs] ${ADS_NAME}: present, SHA256 OK -> ${ADS_OUT_FILE}")
      return()
    endif()
    message(FATAL_ERROR
      "[inputs] ${ADS_NAME}: SHA256 mismatch for existing file.\n"
      "  Path:   ${ADS_OUT_FILE}\n"
      "  Have:   ${_have_sha256}\n"
      "  Expect: ${_want_sha256}\n"
      "Delete the file or fix the URL/SHA."
    )
  endif()

  file(MAKE_DIRECTORY "${_out_dir}")
  set(_tmp "${ADS_OUT_FILE}.part")

  message(STATUS "[inputs] ${ADS_NAME}: downloading -> ${ADS_OUT_FILE}")
  file(DOWNLOAD
    "${ADS_URL}"
    "${_tmp}"
    SHOW_PROGRESS
    TLS_VERIFY ON
    EXPECTED_HASH "SHA256=${ADS_SHA256}"
    STATUS _dl_status
    LOG _dl_log
  )
  list(GET _dl_status 0 _dl_code)
  list(GET _dl_status 1 _dl_msg)
  if(NOT _dl_code EQUAL 0)
    if(EXISTS "${_tmp}")
      file(REMOVE "${_tmp}")
    endif()
    message(FATAL_ERROR "[inputs] ${ADS_NAME}: download failed (${_dl_code}): ${_dl_msg}\n${_dl_log}")
  endif()

  file(RENAME "${_tmp}" "${ADS_OUT_FILE}")
  message(STATUS "[inputs] ${ADS_NAME}: download complete (hash verified)")
endfunction()

# ---- Decompression helper (build-time) ----
function(ams_add_zstd_decompress_target)
  set(options)
  set(oneValueArgs NAME ZST_FILE OUT_FILE)
  cmake_parse_arguments(DEC "${options}" "${oneValueArgs}" "" ${ARGN})

  foreach(req NAME ZST_FILE OUT_FILE)
    if(NOT DEC_${req})
      message(FATAL_ERROR "ams_add_zstd_decompress_target: missing required argument: ${req}")
    endif()
  endforeach()

  # Find zstd executable on PATH
  find_program(AMS_ZSTD_EXECUTABLE zstd)
  if(NOT AMS_ZSTD_EXECUTABLE)
    message(FATAL_ERROR
      "[inputs] zstd not found. Please install it (e.g., 'brew install zstd' on macOS) "
      "or put 'zstd' on PATH."
    )
  endif()

  get_filename_component(_out_dir "${DEC_OUT_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${_out_dir}")

  # Build-time rule: OUT_FILE depends on ZST_FILE
  add_custom_command(
    OUTPUT  "${DEC_OUT_FILE}"
    DEPENDS "${DEC_ZST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
    # Write to temp then rename for atomicity
    COMMAND "${AMS_ZSTD_EXECUTABLE}" -d -f --no-progress -o "${DEC_OUT_FILE}.part" "${DEC_ZST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E rename "${DEC_OUT_FILE}.part" "${DEC_OUT_FILE}"
    COMMENT "[inputs] Decompressing ${DEC_ZST_FILE} -> ${DEC_OUT_FILE}"
    VERBATIM
  )

  add_custom_target("${DEC_NAME}" DEPENDS "${DEC_OUT_FILE}")
endfunction()

