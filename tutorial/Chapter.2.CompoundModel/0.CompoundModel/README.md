# AMS Compound Model 

AMS (partially) replaces some arbitary function with a surrogate ML (torch) model. However, AMS does more than that, it is reponsible to perform:
1. Data collection for that specific function
2. Uncertainty quantification
3. Model inference.

To model all these 3 entities AMS provides a simple mapping between the original computation (the function to be replaced) with an *AMSCompoundModel*. The *AMSCompoundModel* enapuslates all these 3 capabilities and the AMS API associates the application function (referred in AMS terminology as domain) with the respective compound model. *In AMS API we represent a function with a user provided string, this is necessary to refer to the same function with a user provided string*.

## Create an empty Compound Model and associate it with a domain.

Use the AMS API to create a AMS Compound model and register it with AMS.


