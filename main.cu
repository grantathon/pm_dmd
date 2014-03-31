// ###
// ###
// ### Depth Map Denoising of Kinect Depth Images
// ### 
// ###
// ### Technical University of Munich
// ###
// ### 
// ### Group 6
// ### 
// ### Project Phase: Denoising Kinect Depth Images
// ###
// ### Grant Bartel, grant.bartel@tum.de, p051
// ### Faisal Caeiro, faisal.caeiro@tum.de, p079
// ### Ayman Saleem, ayman.saleem@tum.de, p050
// ###
// ###

// Uncomment to use the live Kinect Camera
//#define KINECT

#include "aux.h"	// Helping functions for CUDA GPU Programming
#include <iostream>	// For standard IO on console
#include <sstream>
#include <iomanip>
#include "constant.cuh"

#ifdef KINECT
#include "libfreenect_sync.h"	// Free Kinect Lib
#else
#include <fstream>	// For reading raw binary depth file
#endif

using namespace std;

uint16_t maxValue = 0;
uint16_t *depth = new uint16_t[KINECT_SIZE_X*KINECT_SIZE_Y];
float *fInDepth = new float[KINECT_SIZE_X*KINECT_SIZE_Y];

texture<float, 2, cudaReadModeElementType> vTexRef;

uint16_t normalizeDepth(uint16_t *input, float *output, bool inverse = false)
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

    return maxValue;
}

__global__ void InpaintingMask(bool *m, int w, int h, float thresh)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h) { 
		size_t idx = x + (size_t)y*w;
		
        // Find th valid pixels for the GPU algorithm
	    bool mask = true;
	    for (int dy=-1; dy<=1; ++dy) {
		    for (int dx=-1; dx<=1; ++dx) {
			    if (x+dx>=0 && y+dy>=0 && x+dx < w && y+dy < h) {
				    if (tex2D(vTexRef, x+0.5+dx, y+0.5+dy) >= thresh) {
					    mask = false;
				    }
			    }
		    }
	    }
	    
        if(mask==false) { m[idx] = false; }  // Since m is initialized to true...
	}
}

__host__ __device__ float DiamondDotProduct(float *p, int w, int h, int x, int y)
{
    // Init indexing variables
    size_t idx = x + (size_t)y*w;
    size_t offset = (size_t)w*h;
    
    float p_iMinus1_1, p_iMinus1_2, p_iMinus1_3, p_iPlus1_1, p_iPlus1_2, p_jMinus1_1, p_jMinus1_2, p_jMinus1_3, p_jPlus1_1, p_jPlus1_2, p_ijMinus1_3;
    float dyPlus_1, dyMinus_1, dxPlus_1, dxMinus_1, dyPlus_2, dyMinus_2, dxPlus_2, dxMinus_2, d2dxdy;

    // Set the values necessary for the diamond operator components
    p_iMinus1_1 = p[idx-w];	                    // (i - 1,j,1);   
    p_iMinus1_2 = p[idx-w + offset];            // (i - 1,j,2);
    p_iMinus1_3 = p[idx-w + (size_t)2*offset];  // (i - 1,j,3);   
    dyMinus_1 = p_iMinus1_1 - p[idx];       
    dyMinus_2 = p_iMinus1_2 - p[idx+offset];

    p_iPlus1_1 = p[idx+w];	                    // (i + 1,j,1);    
    p_iPlus1_2 = p[idx+w + offset];	            // (i + 1,j,2); 
    dyPlus_1 = p_iPlus1_1 - p[idx];
    dyPlus_2 = p_iPlus1_2 - p[idx+offset];

    p_jMinus1_1 = p[idx-1];	                    // (i,j - 1,1);   
    p_jMinus1_2 = p[idx-1 + offset];	        // (i,j - 1,2); 
    p_jMinus1_3 = p[idx-1 + (size_t)2*offset];	// (i,j - 1,3); 
    dxMinus_1 = p_jMinus1_1 -  p[idx];
    dxMinus_2 = p_jMinus1_2 -  p[idx+offset];

    p_jPlus1_1 = p[idx+1];	                    // (i,j + 1,1);    
    p_jPlus1_2 = p[idx+1 + offset];	            // (i,j + 1,2);
    dxPlus_1 =  p_jPlus1_1 - p[idx];
    dxPlus_2 =  p_jPlus1_2 - p[idx+offset];

    p_ijMinus1_3 = p[idx-w - 1 + (size_t)2*offset];	// (i - 1,j - 1,3);
    d2dxdy = p[idx+(size_t)2*offset] + p_ijMinus1_3 - p_jMinus1_3 - p_iMinus1_3;

    // Compute the diamond operator components
    float p1 = sqrtf(1.0f/3.0f) * ((dxPlus_1 + dxMinus_1) + (dyPlus_1 + dyMinus_1));
    float p2 = sqrtf(2.0f/3.0f) * ((dxPlus_2 + dxMinus_2) - (dyPlus_2 + dyMinus_2));
    float p3 = sqrtf(8.0f/3.0f) * (d2dxdy);

    // Sum the diamond operator components
    return	p1 + p2 + p3;
}

__global__ void UpdateImageAndDualVariable(float *u, float *p, bool *m, int w, int h, float tau, float theta)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        // Initialize indexing variables
        size_t idx = x + (size_t)y*w;
        size_t offset = (size_t)h*w;
        int xSM = threadIdx.x;
        int ySM = threadIdx.y;
        size_t idxSM = xSM + (size_t)ySM*blockDim.x;

        // Update shared memory U and copy to global memory
        extern __shared__ float uSM[];
        uSM[idxSM] = tex2D(vTexRef, x+0.5, y+0.5);
        __syncthreads();

        if(m[idx] && (x+1)<w && (y+1)<h && x>0 && y>0)
        {
            // Update U using the diamond dot product of P
            uSM[idxSM] -= theta*DiamondDotProduct(p, w, h, x, y);
            u[idx] = uSM[idxSM];
            __syncthreads();
    
            // Perform the diamond operator and update P
	        float u_iMinus1, u_iPlus1, u_jMinus1, u_jPlus1, u_ijPlus1, dyPlus, dyMinus, dxPlus, dxMinus, d2dxdy;
	        
	        u_iMinus1 = (ySM != 0 ? uSM[idxSM - blockDim.x] : u[idx - w]);	
			dyMinus = u_iMinus1 - uSM[idxSM];

			u_iPlus1 	= ((ySM +  1)<blockDim.y ? uSM[idxSM + blockDim.x] : u[idx + w]);
			dyPlus = u_iPlus1 - uSM[idxSM];

			u_jMinus1 = (xSM != 0 ? uSM[idxSM - 1] : u[idx - 1]);
			dxMinus = u_jMinus1 -  uSM[idxSM];

			u_jPlus1 	= ((xSM +  1)<blockDim.x ? uSM[idxSM + 1] : u[idx + 1]);
			dxPlus = u_jPlus1 - uSM[idxSM];

			u_ijPlus1 = u[idx + w + 1];
			d2dxdy = uSM[idxSM] + u_ijPlus1 - u_jPlus1 - u_iPlus1;

		    float p1 = p[idx]                   + (tau/theta)*sqrtf(1.0f/3.0f) * ((dxPlus + dxMinus) + (dyPlus + dyMinus));
		    float p2 = p[idx+offset]            + (tau/theta)*sqrtf(2.0f/3.0f) * ((dxPlus + dxMinus) - (dyPlus + dyMinus));
		    float p3 = p[idx+(size_t)2*offset]  + (tau/theta)*sqrtf(8.0f/3.0f) * (d2dxdy);
            float maxDenom = fmax(1.0f, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
            
            // Update the normalized components of P
            p[idx]                  = p1/maxDenom;
            p[idx+offset]           = p2/maxDenom;
            p[idx+(size_t)2*offset] = p3/maxDenom;
        }
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
	size_t blockX = 64, blockY = 4, blockZ = 1;
	getParam("blockX", blockX, argc, argv);
	getParam("blockY", blockY, argc, argv);
	getParam("blockZ", blockZ, argc, argv);
	cout << "blocksize: " << blockX << "x" << blockY << "x" << blockZ << endl;

	// Default setting for optimization parameter theta
    float theta = 0.01f;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

	// Default setting for time step
    float tau = 0.01f;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

	// Default setting for theta decay
    float decay = 1.0f;
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

	maxValue = normalizeDepth(depth, fInDepth);

	// Setup input image and save
	cv::Mat mInDepth(KINECT_SIZE_Y,KINECT_SIZE_X,CV_32FC1);
	convert_layered_to_mat(mInDepth, fInDepth);
	showImage("Input Depth Image", mInDepth, 100, 100);
    cv::imwrite("image_input.png",mInDepth*255.f);

    // Setup output image
    float *fOutDepth = new float[(size_t)KINECT_SIZE_Y*KINECT_SIZE_X];
	cv::Mat mOutDepth(KINECT_SIZE_Y,KINECT_SIZE_X,CV_32FC1);
	
    // Allocate memory on the GPU and copy data
    float *dU, *dV, *dP;
    bool *dM;
    
    cudaMalloc(&dM, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(bool)); CUDA_CHECK;
    cudaMalloc(&dU, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dV, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dP, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    
    cudaMemcpy(dU, fInDepth, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(dV, dU, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
    cudaMemset(dP, 0, (size_t)3*KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(float)); CUDA_CHECK;
    cudaMemset(dM, true, (size_t)KINECT_SIZE_Y*KINECT_SIZE_X*sizeof(bool)); CUDA_CHECK;

    // Start the timer
    Timer timer;
    timer.start();

    // Setup texture reference for V
    vTexRef.addressMode[0] = cudaAddressModeClamp;
    vTexRef.addressMode[1] = cudaAddressModeClamp;
    vTexRef.filterMode = cudaFilterModeLinear;
    vTexRef.normalized = false;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &vTexRef, dV, &desc, KINECT_SIZE_X, KINECT_SIZE_Y, KINECT_SIZE_X*sizeof(float));

    // Init block, grid, and shared memory size
    dim3 block = dim3(blockX, blockY, blockZ);
    dim3 grid = dim3((KINECT_SIZE_X+block.x-1)/block.x, (KINECT_SIZE_Y+block.y-1)/block.y, 1);
    size_t smBytes = (size_t)block.x*block.y*block.z*sizeof(float);

    // Check which pixels should be ignored in the main computation
    InpaintingMask<<<grid, block>>>(dM, KINECT_SIZE_X, KINECT_SIZE_Y, 1.0f);

    // Iterate through main computation
    for(int n=0; n<N; n++)
    {
        theta *= decay;
        cudaDeviceSynchronize();
        UpdateImageAndDualVariable<<<grid, block, smBytes>>>(dU, dP, dM, KINECT_SIZE_X, KINECT_SIZE_Y, tau, theta); CUDA_CHECK;
    }
    
    // Copy data back to CPU
    cudaMemcpy(fOutDepth, dU, (size_t)KINECT_SIZE_X*KINECT_SIZE_Y*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // Display output image and save
    convert_layered_to_mat(mOutDepth, fOutDepth);
    showImage("Output Depth Image", mOutDepth, 100+KINECT_SIZE_X, 100);
    std::stringstream filename;
    
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
    
    // Free and unbind memory on the GPU
    cudaFree(dU); CUDA_CHECK;
    cudaFree(dV); CUDA_CHECK;
    cudaFree(dP); CUDA_CHECK;
    cudaFree(dM); CUDA_CHECK;
    cudaUnbindTexture(vTexRef);

	// free golbal allocated arrays
	delete[] fInDepth;
	delete[] fOutDepth;
	delete[] depth;
	
	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}

