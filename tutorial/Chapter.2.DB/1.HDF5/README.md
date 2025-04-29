# Data collection through the hdf5 interface

AMS provides file system databases and in-situ data processing capabilities. File system databases are created in an existing directory and files follow the hdf5 file format.

Within the hdf5 files there will exist 2 *datasets* the inputs and the outputs.
1. Inputs will contain all input tensors and inout tensors as defined by the application executor. 
2. Outputs contain both output tensors and inout tensors as defined by the application executor.

Please extend the example code to perform data collection through the AMS interface. 

## Extend the solution cli to accept a directory to store all data to and the file name prefix.

You will need to instantiate a file system database and assign a database label to the compound model. The database label will act as the prefix of the filename.  

