# Build and Link with AMS

AMS is a standard cmake package providing a `AMS-config.cmake` file and the AMS target.
We provide an example C++ code and the respective `cmake` file to build and link with AMS. 

## Include and link with AMS

To use AMS we need to `find_package(AMS REQUIRED)` ([here](./CMakeLists.txt#L22)) and then use the `AMS::AMS` target in the `cmake` function `target_link_libraries`([here](./CMakeLists.txt#L36)).

## Example Code 

The [example](./ex0.cpp) code is very simple. The user provides in the *cli* the length of 2 vectors, the example initializes the `input` vector from *0-length-1* values and then assigns these values to the [output vector](./ex0.cpp#L52) finally it computes the sum of all elements in the output vector and prints the sum at the terminal with the expected output value.

## Configure and link

To configure the example please provide these commands:

```
mkdir build
cd build
cmake  ../
```

If AMS is not installed in a default cmake directory please also provide the path to the AMS installation by passing the option 
`-DAMS_DIR=<path-to-install>` to the cmake command. 

## Execute example

To execute the example please provide the following *cli*:

```
./EX0 -l 10
```

And the expected output should be:

```
[Example] Expected output is 45 and computed 45
```

## Enable Logger

The container's AMS version is linked and working with the AMS logger to enable it you can execute:

```
AMS_LOG_LEVEL=debug ./EX0 -l 10
```

The output should look like this:

```
[AMS:DEBUG:ResourceManager] Initialization of allocators
[AMS:DEBUG:ResourceManager] Set Allocator [0] to pool with name : HOST
[Example] Expected output is 45 and computed 45
[AMS:DEBUG:AMS] Finalization of AMS
[AMS:DEBUG:AMSDefaultDeviceAllocator] Destroying default host allocator
```

There are the following log levels, debug, info, warning.
