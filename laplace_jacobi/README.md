# `openacc-unified-memory-jacobi` Sample

| Area              | Description
|:---                   |:---
| What you will learn              | Migrate the sample from OpenACC to OpenMP and optimize
| Time to complete              | 15 minutes
| Category                      | Code Optimization

## Purpose

The sample shows the migration of convolutionSeperable from CUDA to SYCL using SYCLomatic tool and optimizing the migrated sycl code further to achieve good results.


## Prerequisites

| Optimized for              | Description
|:---                   |:---
| OS                    | Ubuntu* 22.04
| Hardware              | Intel® Gen9 <br> Intel® Gen11 <br> Intel® Xeon CPU <br> Intel® Data Center GPU Max
| Software                | intel-application-migration-tool-for-openacc-to-openmp <br> Intel® oneAPI Base Toolkit version 2024.1.0

For more information on how to install intel-application-migration-tool-for-openacc-to-openmp visit [Migrate from OpenACC* to OpenMP*](https://github.com/intel/intel-application-migration-tool-for-openacc-to-openmp.git) 

## Key Implementation Details

This sample demonstrates the migration of the following OpenACC pragmas: 

- #pragma acc kernels
- #pragma acc loop independent

### OpenACC source code evaluation

This sample is migrated from the NVIDIA OpenACC sample. See the sample [openacc-unified-memory-jacobi](https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/openacc-unified-memory-jacobi) in the NVIDIA-developer-blog GitHub.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `openacc-unified-memory-jacobi` Sample

### Migrate the Code using `intel-application-migration-tool-for-openacc-to-openmp`

For this sample, the tool automatically migrates 100% of the CUDA runtime API's to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA-developer-blog/code-samples.git
   ```
2. Change to the convolutionSeparable sample directory.
   ```
   cd posts/openacc-unified-memory-jacobi
   ```
3. Migrate the code using the tool
   ```
   intel-application-migration-tool-for-openacc-to-openmp/src/intel-application-migration-tool-for-openacc-to-openmp laplace2d.c 
   ```
   
### Manual Workaround
The kernels construct (`#pragma acc kernels`) is a prescriptive construct for the OpenACC compiler to automatically extract the parallelism from the code enclosed by the construct and it has no direct OpenMP counter-part. 
In order to get the functional correctness we need to map the (variables Anew, A & error) data to the target. Replace the `#pragma omp target` line from the code as follows
```
#pragma omp target map(to: Anew[0:n*m]) map(tofrom: A[0:n*m], error)   
```

### Optimizations

The migrated code can be optimized by using profiling tools which helps in identifying the hotspots (in this case convolutionRowsKernel() and convolutionColumnsKernel()).
 
If we observe the migrated SYCL code, especially in the above-mentioned function calls we see many ‘for’ loops that are being unrolled.
Although loop unrolling exposes opportunities for instruction scheduling optimization by the compiler and thus can improve performance, sometimes it may increase pressure on register allocation and cause register spilling. 

So, it is always a good idea to compare the performance with and without loop unrolling along with different times of unrolls to decide if a loop should be unrolled or how many times to unroll it.

In this case, by implementing the above technique, we can decrease the execution time by avoiding loop unrolling at the innermost “for-loop” of the computation part in convolutionRowsKernel function (line 120) and avoiding loop unrolling at the outer loop of the computation part in convolutionColumnsKernel function (line 242) of the file convolutionSeparable.dp.cpp.

Also, we can still decrease the execution time by avoiding the repetitive loading of c_Kernel[] array (as it is independent of `i` for-loop in convolutionSeparable.dp.cpp file). 

  ```
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
      float sum = 0;
  for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
     sum += c_Kernel[KERNEL_RADIUS - j] *s_Data[item_ct1.get_local_id(1)][item_ct1.get_local_id(2) + i * ROWS_BLOCKDIM_X + j];}
  ```

We can separate the array and load it into another new array and use it in place of c_Kernel

  ```
  float a[2*KERNEL_RADIUS + 1];
  for(int i=0; i<= 2*KERNEL_RADIUS; i++)
  a[i]=c_Kernel[i]; 
  ```
>**Note**: These optimization techniques also work with the larger input image sizes.

## Build the `convolutionSeparable` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   icx -fiopenmp -fopenmp-targets=spir64   laplace2d_translated.c -DOPENACC2OPENMP_ORIGINAL_OPENMP=1
   ```

3. Run the code

      ```
      ./a.out
      ```
