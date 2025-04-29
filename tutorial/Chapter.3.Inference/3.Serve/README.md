# Model inference through AMS interface.

Please modify the example code to create a AMSCompoundModel with the model trained on the previous example. Since we have already defined compounding and the matching of the executors with the underlying function. The changes should be minimal.

## Threshold and Uncertainty.

In AMS every compound model is accompanied by some UQ method. When models do not define their own uncertainty approach, we will define by default *random* uncertainty. Although the implementation of the ML UQ mechanism will be different the implementation remains the same. 


### Random Uncertainty

In the case of random uncertainty *AMS* computes a random value in the interval [0,1] for every computed sample. When the uncertainty (the randomly generated value) is higher than a user provided threshold, AMS considers this as a highly uncertain inference and only computes the underlying function for that specific sample.

In the provided [solution](./sol6.cpp) there is an extra cli parameter that defines the uncertainty threshold for the compound model. Use the parameter and explore both the reported values and how the size of the database increases.
