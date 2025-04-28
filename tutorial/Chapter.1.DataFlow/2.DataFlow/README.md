# AMS DataFlow
AMS (partially) replaces some arbitary function with Machine Learning (Torch) model. The `AMSTensor` provides information to the AMS runtime regarding the memory access pattern of a single memory blob. 

However, an arbitary function can access multiple memory blobs with different intentions. AMS categorizes intention in 3 categories:

1. Memory locations that are being written by the underlying computation and are considered a result of the function. In the example this would be `out`.
2. Memory locations that are being read by the underlying computation and is "necessary" for the mathematical formulation of the underlying result. Intermediate inputs or temporal variables can be ignored. In the example this would be `in`.
3. Memory locations that are being read AND written  by the underlying computation and is "necessary" for the mathematical formulation of the underlying result. Intermediate inputs or temporal variables can be ignored. 

## The SmallVector

Multiple memory blobs of the same intention can be packed together in a vector. AMS instead of using the `std::vector` uses ams::SmallVector a lightweight C++ vector abstraction (originated from the LLVM project) that can be allocated in the stack and is more efficient. 

### Create a C++ lambda that takes 3 input parameters (1 for each memory category), each of Smallvector type storing the AMSTensors. 



