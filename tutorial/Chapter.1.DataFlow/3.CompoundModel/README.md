# AMS Core Concepts 


## AMS Compound Model
AMS (partially) replaces some arbitary function with a surrogate ML (torch) model. However, AMS does more than that, it is reponsible to perform:
1. Data collection for that specific function
2. Uncertainty quantification
3. Model inference.

To model all these 3 entities AMS provides a simple mapping between the original computation (the function to be replaced) with an *AMSCompoundModel*. The *AMSCompoundModel* enapuslates all these 3 capabilities and the AMS API associates the application function (referred in AMS terminology as domain) with the respective compound model. *In AMS API we represent a function with a user provided string, this is necessary to refer to the same function with a user provided string*.

The AMS Compound Model is a persistent data structure that will be persist inside the application during application execution time and does not define any computation.


### Create an empty Compound Model and associate it with a domain.

Use the AMS API to create a AMS Compound model and register it with AMS.

#### Discussion about solution


In the solution we are using the following API call:

```CPP
  AMSCAbstrModel model_descr = 
    AMSRegisterAbstractModel(
      "compute", ams::AMSUQPolicy::AMS_RANDOM, -1.0, "", "compute");
```


Most of the parameters passed to the `AMSRegisterAbstractModel` are place holders and in the next examples we will fill in accordingly. 

## AMS Executor 

Besides the compound model, AMS provides the notion of the `AMSExecutor`, the executor is reponsible to perform all necessary actions described in the compound model and when uncertain it will fall back to the existing solution.
Essentially the executor associates the domain name, domain function with a compound model and performs the necessary actions. 
