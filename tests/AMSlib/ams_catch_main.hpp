#ifndef AMS_TEST_CATCH_MAIN_HPP
#define AMS_TEST_CATCH_MAIN_HPP

#include <catch2/catch_session.hpp>
#include <cstdio>
#include <cstdlib>

#include "AMS.h"

namespace ams::test
{
inline int runCatchSession(int argc, char** argv, bool finalizeAMS = true)
{
  Catch::Session session;
  if (int rc = session.applyCommandLine(argc, argv)) return rc;

  int rc = session.run();
  if (finalizeAMS) AMSFinalize();

  std::fflush(stdout);
  std::fflush(stderr);

  // HIP/Torch finalizers can abort after successful tests; AMS is cleaned above.
  const int normalizedRC = (rc == 0 || rc == 4) ? EXIT_SUCCESS : rc;
  std::_Exit(normalizedRC);
}
}  // namespace ams::test

#endif  // AMS_TEST_CATCH_MAIN_HPP
