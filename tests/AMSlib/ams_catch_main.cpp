#include "ams_catch_main.hpp"


// Shared Catch2 entry point for tests that need AMS-controlled shutdown.
int main(int argc, char** argv)
{
  return ams::test::runCatchSession(argc, argv);
}
