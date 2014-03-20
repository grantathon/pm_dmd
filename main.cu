// ###
// ###
// ### Depth Map Denoising of Kinect Depth Images
// ### 
// ###
// ### Technical University of Munich
// ###
// ### 
// ### Grant Bartel, grant.bartel@tum.de
// ### Faisal Caeiro, faisal.caeiro@tum.de
// ### Ayman Saleem, ayman.saleem@tum.de
// ###
// ###

// Uncomment to use the live Kinect Camera
//#define KINECT
#include "aux.h"	// Helping functions for CUDA GPU Programming
#include <iostream>	// For standard IO on console
#include "constant.cuh"

#ifdef KINECT
#include "libfreenect_sync.h"	// Free Kinect Lib
#else
#include <fstream>	// For reading raw binary depth file
#endif

using namespace std;

uint16_t *depth = new uint16_t[KINECT_SIZE_X*KINECT_SIZE_Y];
float *fInDepth = new float[KINECT_SIZE_X*KINECT_SIZE_Y];

void normalizeDepth(uint16_t *input, float *output, bool inverse = false)
{
	uint16_t maxValue = 0.0f;

	// Find the maximum value
	for (size_t y = 0; y < KINECT_SIZE_Y; y++)
	{
		for (size_t x = 0; x < KINECT_SIZE_X; x++)
		{
			size_t idx = x + y * KINECT_SIZE_X;
			if (maxValue < input[idx]) maxValue = input[idx];
		}
	}
	// Normalize it to [0,1]
	for (size_t y = 0; y < KINECT_SIZE_Y; y++)
		for (size_t x = 0; x < KINECT_SIZE_X; x++)
		{
			size_t idx = x + y * KINECT_SIZE_X;
			if (isnan(input[idx])) output[idx] = 1.0f;
			else output[idx] = (inverse) ? 1.0f - (float) input[idx] / (float) maxValue : (float) input[idx] / (float) maxValue;
		}
}

__host__ __device__ float DiamondDotProduct(float *p, int w, int h, int x, int y)
{
    size_t offset = (size_t)h*w;
    float pp = p[0];
    float a1 = 0.0f;    float a2 = 0.0f;    float a3 = 0.0f;
    float b1 = 0.0f;    float b2 = 0.0f;
    float c1 = 0.0f;    float c2 = 0.0f;    float c3 = 0.0f;
    float d1 = 0.0f;    float d2 = 0.0f;
                                            float e3 = 0.0f;

    if(x!=0)            { a1 = p[-1]; a2 = p[offset-1];     a3 = p[2*offset-1]; }
    if((x+1)!=w)        { b1 = p[1];  b2 = p[offset+1]; }
    if(y!=0)            { c1 = p[-w]; c2 = p[offset-w];     c3 = p[2*offset-w]; }
    if((y+1)!=h)        { d1 = p[w];  d2 = p[offset+w]; }
    if(y!=0 && x!=0)    {                                   e3  = p[2*offset-w-1]; }

    return  sqrtf(1.0f/3.0f)*( a1 + b1 + c1 + d1 - 4*pp )
          + sqrtf(2.0f/3.0f)*( c2 + d2 - a2 - b2 )
          + sqrtf(8.0f/3.0f)*( pp + e3 - a3 - c3 );
}

__host__ __device__ void DiamondOperator(float *u, float* dd, int w, int h, int x, int y)
{
    size_t offset = (size_t)h*w;
    float uu = u[0];
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;
    float e = 0.0f;

    if(x!=0)                    { a = u[-1]; }
    if((x+1)!=w)                { b = u[1]; }
    if(y!=0)                    { c = u[-w]; }
    if((y+1)!=h)                { d = u[w]; }
    if((y+1)!=h && (x+1)!=w)    { e = u[w+1]; }

    dd[0]           = sqrtf(1.0f/3.0f)*( a + b + c + d - 4*uu );
    dd[offset]      = sqrtf(2.0f/3.0f)*( c + d - a - b );
    dd[2*offset]    = sqrtf(8.0f/3.0f)*( uu + e - b - d );
}

__global__ void ComputeImageUpdate(float *v, float *d, float *p, float *u, int w, int h, float tau, float theta)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;
        size_t offset = (size_t)h*w;

        u[idx] = v[idx] - theta*DiamondDotProduct(&p[idx], w, h, x, y);
        
        DiamondOperator(&u[idx], &d[idx], w, h, x, y);
        float p1 = p[idx]           + (tau/theta)*d[idx];
        float p2 = p[idx+offset]    + (tau/theta)*d[idx+offset];
        float p3 = p[idx+2*offset]  + (tau/theta)*d[idx+2*offset];
        float maxDenom = fmax(1, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
        
        p[idx]          = p1/maxDenom;
        p[idx+offset]   = p2/maxDenom;
        p[idx+2*offset] = p3/maxDenom;
    }
}

int main(int argc, char **argv)
{
	// Before the GPU can process the kernels, call Device Synchronize for devise initialization
	cudaDeviceSynchronize(); CUDA_CHECK;

#ifdef KINECT
#else
	// Raw File input is a must
	string rawfile = "";
	bool ret = getParam("i", rawfile, argc, argv);
	if (!ret) cerr << "ERROR; no input raw file specified" << endl;
	if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -blockX -blockY -blockZ -theta -tau -decay -N" << endl; return 1;}
#endif

	// Default setting for block sizes
	size_t blockX = 32, blockY = 8, blockZ = 1;
	getParam("blockX", blockX, argc, argv);
	getParam("blockY", blockY, argc, argv);
	getParam("blockZ", blockZ, argc, argv);
	cout << "blocksize: " << blockX << "x" << blockY << "x" << blockZ << endl;

	// Default setting for optimization parameter theta
    float theta = 500.0f;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

	// Default setting for time step
    float tau = 0.005f;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

	// Default setting for theta decay
    float decay = 0.98f;
    getParam("decay", decay, argc, argv);
    cout << "decay: " << decay << endl;

	// Default setting for total GPU iterations
    int N = 200;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

#ifdef KINECT
	while (cv::waitKey(30) < 0)
	{
		void *data;
		unsigned int timestamp;
        freenect_sync_get_depth((void**)(&data), &timestamp, 0, FREENECT_DEPTH_11BIT);
        depth = (uint16_t*)data;
      
#else
    // Load the raw file (Size must be KINECT_SIZE_X x KINECT_SIZE_Y) i.e. 640x480
	ifstream file_buf(rawfile.c_str(), ios_base::binary);
	file_buf.read((char*) depth, KINECT_SIZE_X*KINECT_SIZE_Y*sizeof(uint16_t));
	file_buf.close();
#endif

	normalizeDepth(depth, fInDepth);

	// Setup input image and save
	cv::Mat mInDepth(KINECT_SIZE_Y,KINECT_SIZE_X,CV_32FC1);
	convert_layered_to_mat(mInDepth, fInDepth);
	showImage("Input Depth Image", mInDepth, 100, 100);
    cv::imwrite("image_input.png",mInDepth*255.f);

    // Setup output image
    float *fOutDepth = new float[(size_t)KINECT_SIZE_Y*KINECT_SIZE_X];
	cv::Mat mOutDepth(KINECT_SIZE_Y,KINECT_SIZE_X,CV_32FC1);
	
    // Start the timer for the GPU process
    Timer timer;
    timer.start();

    // Allocate memory on the GPU and copy data
    float *dU, *dV, *dP, *dD;
    cudaMalloc(&dU, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dV, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dP, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dD, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(dU, fInDepth, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(dV, dU, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
    cudaMemset(dP, 0, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMemset(dD, 0, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(blockX, blockY, blockZ);
    dim3 grid = dim3((KINECT_SIZE_X+block.x-1)/block.x, (KINECT_SIZE_Y+block.y-1)/block.y, 1);

    // Iterate through main computation
    for(int n=0; n<N; n++)
    {
        theta *= decay;
        ComputeImageUpdate<<<grid, block>>>(dV, dD, dP, dU, KINECT_SIZE_X, KINECT_SIZE_Y, tau, theta);
        cudaDeviceSynchronize();
    }

    // Compute final output image
    theta *= decay;
    ComputeImageUpdate<<<grid, block>>>(dV, dD, dP, dU, KINECT_SIZE_X, KINECT_SIZE_Y, tau, theta);
    
    // Copy data back to CPU
    cudaMemcpy(fOutDepth, dU, (size_t)KINECT_SIZE_X*KINECT_SIZE_Y*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    // Deallocate memory on the GPU
    cudaFree(dU); CUDA_CHECK;
    cudaFree(dV); CUDA_CHECK;
    cudaFree(dP); CUDA_CHECK;
    cudaFree(dD); CUDA_CHECK;

    // Display output image and save
    convert_layered_to_mat(mOutDepth, fOutDepth);
	showImage("Output Depth Image", mOutDepth, 100+KINECT_SIZE_X, 100);
    cv::imwrite("image_output.png",mOutDepth*255.f);

    // End the timer for the GPU process
    timer.end();
    float t = timer.get();  // Time in seconds
    cout << "GPU time: " << t*1000 << " ms" << endl;

#ifdef KINECT
	}
#else
	// wait for key input to quit
	cv::waitKey(0);
#endif

	// free golbal allocated arrays
	delete[] fInDepth;
	delete[] fOutDepth;
	delete[] depth;
	
	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}

