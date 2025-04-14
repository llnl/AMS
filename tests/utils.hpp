/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

 #ifndef __TEST_UTILS__
 #define __TEST_UTILS__
 
#include <csignal>
#include <unistd.h>
#include <execinfo.h>

// Signal handler to print the stack trace
void signalHandler(int signum) {
    const char* msg = "[signalHandler] Caught signal\n";
    write(STDERR_FILENO, msg, sizeof(msg));

    // Obtain the backtrace
    const int maxFrames = 128;
    void *addrlist[maxFrames];

    // Get void*'s for all entries on the stack
    int addrlen = backtrace(addrlist, maxFrames);

    if (addrlen == 0) {
        const char* no_stack = "No stack trace available\n";
        write(STDERR_FILENO, no_stack, sizeof(no_stack));
        _exit(1); // _exit() Cannot be trap, interrupted
    }

    // Print out all the frames to stderr
    backtrace_symbols_fd(addrlist, addrlen, STDERR_FILENO);
    _exit(1);
}


void installSignals() {
    std::signal(SIGSEGV, signalHandler); // segmentation fault
    std::signal(SIGABRT, signalHandler); // abort()
    std::signal(SIGFPE, signalHandler);  // floating-point exception
    std::signal(SIGILL, signalHandler);  // illegal instruction
    std::signal(SIGINT, signalHandler);  // interrupt (e.g., Ctrl+C)
    std::signal(SIGTERM, signalHandler); // termination request
    std::signal(SIGPIPE, signalHandler); // broken pipe  
}

#endif