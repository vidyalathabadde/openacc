#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <dpct/rng_utils.hpp>

#include <dpct/dpl_utils.hpp>

// Fill d_buffer with num random numbers
extern "C" void fill_rand(float *d_buffer, int num, void *stream) try {
  dpct::rng::host_rng_ptr gen;
  int status;

  // Create generator
  status = DPCT_CHECK_ERROR(
      gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59));

  // Set CUDA stream
  status |= DPCT_CHECK_ERROR(gen->set_queue((dpct::queue_ptr)stream));

  // Set seed
  status |= DPCT_CHECK_ERROR(gen->set_seed(1234ULL));

  // Generate num random numbers
  status |= DPCT_CHECK_ERROR(gen->generate_uniform(d_buffer, num));

  // Cleanup generator
  status |= DPCT_CHECK_ERROR(gen.reset());

  if (status != 0) {
      printf ("curand failure!\n");
      exit (EXIT_FAILURE);
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Sort key value pairs
extern "C" void sort(int *keys, int *values, int num, void *stream)
{
    try {
        // Sort keys AND values array by key
        dpct::sort(oneapi::dpl::execution::make_device_policy(
                       *((dpct::queue_ptr)stream)),
                   keys, keys + num, values);
    }
    catch (std::system_error &e) {
        std::cerr << "Error sorting with Thrust: " << e.what() << std::endl;
        exit (EXIT_FAILURE);
    }
}
