Implementation of the van Herk/Gil-Werman algorithm for fast morphological dilation/erosion exploiting shared memory for maximum GPU parallelization.

The repository contains a Python wrapper around the CUDA code based on cupy RawKernels.

The current implementation is generally 5-10x time faster than the one contained in Scipy when using an Nvidia GTX 1060 against an overclocked Intel i5-6600k.

As i'm new to CUDA develop i'm also quite sure this can go even faster!

This repo is heavily inspired by "Parallel van Herk/Gil-Werman image morphology on GPUs using CUDA" Luke Domanski, Pascal Vallotton, Dadong Wang (2016) and of course "A fast algorithm for local minimum and maximum filters on rectangular and octagonal kernels" Van Herk, M. (1992). 
