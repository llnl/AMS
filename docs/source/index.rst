Autonomous MultiScale Library
=============================

.. toctree::
   :hidden:

   installation
   api/library_root


.. image:: https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg
   :target: https://github.com/LLNL/AMS/blob/develop/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/LLNL/AMS
   :target: https://github.com/LLNL/AMS
   :alt: GitHub Stars

Autonomous MultiScale (AMS) is a framework designed to simplify the integration of machine learning (ML) surrogate models 
in multiphysics high-performance computing (HPC) codes.

Overview
--------

AMS provides the end-to-end infrastructure to automate all steps in the process 
from testing and deploying ML surrogate models in
scientific applications. With simple code modifications, developers can integrate 
AMS into their scientific workflows to make multiphysics codes:

* **Faster** - by replacing expensive evaluations with reliable surrogate models 
  backed by verified fallbacks.
* **More Accurate** - by increasing the effective fidelity of subscale models 
  beyond what is currently feasible.
* **Portable** - by providing a general framework applicable to a wide range of 
  use cases.

Key Features
------------

* **Automated Workflow**: Automation of ML surrogate models deployment and testing.
* **HPC Integration**: Designed for supercomputing environments.
* **Multiple Backend Support**: CPU, or GPU (CUDA and HIP).
* **Database Integration**: Support for HDF5 and RabbitMQ.
* **Surrogate Model Support**: PyTorch.
* **Performance Monitoring**: Built-in Caliper support.

Quick Links
-----------

* **GitHub Repository**: https://github.com/LLNL/AMS
* **Issue Tracker**: https://github.com/LLNL/AMS/issues

Citation
--------

If you use this software, please cite it as:

.. code-block:: bibtex

   @software{ams2023,
     author = {Bhatia, Harsh and Patki, Tapasya A. and Brink, Stephanie and 
               Pottier, Loïc and Stitt, Thomas M. and Parasyris, Konstantinos and 
               Milroy, Daniel J. and Laney, Daniel E. and Blake, Robert C. and 
               Yeom, Jae-Seung and Bremer, Peer-Timo and Doutriaux, Charles},
     title = {Autonomous MultiScale Library},
     url = {https://github.com/LLNL/AMS},
     year = {2023},
     doi = {10.11578/dc.20230721.1}
   }
