// Assignment 1: ParallelSine
// CSCI 415: Networking and Parallel Computation
// Spring 2017
// Name(s): 
//
// Sine implementation derived from slides here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch


// standard imports
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// problem size (vector length) N
static const int N = 12345678; //#of threads?

// Number of terms to use when approximating sine
static const int TERMS = 6; //# of blocks

// kernel function (CPU - Do not modify)
void sine_serial(float *input, float *output)
{
  int i;

  for (i=0; i<N; i++) {
      float value = input[i]; //0.1f * i ;i=(0-N)
      float numer = input[i] * input[i] * input[i]; //input^3
      int denom = 6; // 3! 
      int sign = -1; 
      //std::cout << input[i] << std::endl;
      for (int j=1; j<=TERMS;j++) 
      { 
         value += sign * numer / denom; 
         numer *= input[i] * input[i]; //(input^2 * input^3)*blockIdx.x
         denom *= (2*j+2) * (2*j+3); 
         sign *= -1; 
      } 
      output[i] = value;
      //std::cout << output[i] << std::endl;
    }
}


// kernel function (CUDA device)
// TODO: Implement your graphics kernel here. See assignment instructions for method information
__global__ void paralellSine(float *input, float *output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //Proper indexing of elements.
	float value = input[idx];
	float numer = input[idx] * input[idx] * input[idx];
	int denom = 6;
	int sign = -1;

	for (int j=1; j<=TERMS; j++)
	{
		value += sign * numer/denom;
		numer *= input[idx] * input[idx];
		denom *= (2 * j + 2) * (2 * j + 3);
		sign *= -1;
	}
	output[idx] = value;


}

// BEGIN: timing and error checking routines (do not modify)

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, std::string name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << std::setprecision(5);	
	std::cout << name << ": " << ((float) (end_time - start_time)) / (1000 * 1000) << " sec\n";
	return end_time - start_time;
}

void checkErrors(const char label[])
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

// END: timing and error checking routines (do not modify)



int main (int argc, char **argv)
{
  //BEGIN: CPU implementation (do not modify)
  float *d_input;
  float *d_output;
  int size = N * sizeof(float);
  //Initialize data on CPU
  float *h_input = (float*)malloc(N*sizeof(float));
  int i;
  for (i=0; i<N; i++)
  {
    h_input[i] = 0.1f * i;
  }


  //Execute and time the CPU version
  long long CPU_start_time = start_timer();
  float *h_cpu_result = (float*)malloc(N*sizeof(float));
  sine_serial(h_input, h_cpu_result);
  long long CPU_time = stop_timer(CPU_start_time, "\nCPU Run Time");
  //END: CPU implementation (do not modify)


  long long GPU_Total_start = start_timer();

  long long GPU_Malloc_Start = start_timer();
  float *h_gpu_result = (float*)malloc(N*sizeof(float));
  cudaMalloc((void **) &d_input, size);
  cudaMalloc((void **) &d_output, size);
  long long GPU_Malloc = stop_timer(GPU_Malloc_Start, "\nGPU Memory Allocation");

  long long GPU_Memcpy_start = start_timer();
  cudaMemcpy(d_input, h_input,size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output,h_gpu_result, size, cudaMemcpyHostToDevice);
  long long GPU_Memcpy = stop_timer(GPU_Memcpy_start, "GPU Memory Copy to Device");

  long long GPU_start_time = start_timer();
  paralellSine <<< 12057,1024 >>> (d_input, d_output); //Blocks must be < # CUDA cores
  cudaThreadSynchronize();
  long long GPU_time = stop_timer(GPU_start_time, "GPU Kernel Run Time");

  long long GPU_MemcpytoHost_start = start_timer();
  cudaMemcpy(h_gpu_result, d_output, size, cudaMemcpyDeviceToHost);
  long long GPU_MemcpytoHost = stop_timer(GPU_MemcpytoHost_start, "GPU Memory Copy to Host");

  long long GPU_Time = stop_timer(GPU_Total_start, "GPU Total run time");

  // Checking to make sure the CPU and GPU results match - Do not modify
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_cpu_result[i]-h_gpu_result[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("\nResult comparison failed.\n");
  else
    printf("\nResult comparison passed.\n");

  // Cleaning up memory
  free(h_input);
  free(h_cpu_result);
  free(h_gpu_result);

  cudaFree(d_input);

  return 0;
}






