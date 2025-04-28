# AMS DataFlow
AMS (partially) replaces some arbitary function with Machine Learning (Torch) model. 

## The AMSTensor

The AMSTensor is a C++ abstraction that associates a contineous memory blob with some access pattern and reshaping. In other words, it represents the memory as a tensor. The AMSTensor is a shim lay on top of the torch tensor representation and currently only isolates the binary linkage of the example/application code to the torch librarry.


### Use AMSTensors in the example code 



