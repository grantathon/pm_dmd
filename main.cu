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


#include "aux.h"	// Helping functions for CUDA GPU Programming
#include <iostream>	// For standard IO on console
#include <fstream>	// For reading raw depth file

using namespace std;

// Uncomment to use the live Kinect Camera
//#define KINECT

__global__ void DiamondDotProduct(float *p, float *d, int w, int h)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;

        float pp = p[idx];
        float a1 = 0.0f;    float a2 = 0.0f;    float a3 = 0.0f;
        float b1 = 0.0f;    float b2 = 0.0f;    float b3 = 0.0f;
        float c1 = 0.0f;    float c2 = 0.0f;    float c3 = 0.0f;
        float d1 = 0.0f;    float d2 = 0.0f;    float d3 = 0.0f;
                                                float e3 = 0.0f;

        if(x!=0)            { a1 = p[idx-1]; a2 = p[idx+(size_t)h*w-1]; a3 = p[idx+(size_t)2*h*w-1]; }
        if((x+1)!=w)        { b1 = p[idx+1]; b2 = p[idx+(size_t)h*w+1]; b3 = p[idx+(size_t)2*h*w+1]; }
        if(y!=0)            { c1 = p[idx-w]; c2 = p[idx+(size_t)h*w-w]; c3 = p[idx+(size_t)2*h*w-w]; }
        if((y+1)!=h)        { d1 = p[idx+w]; d2 = p[idx+(size_t)h*w+w]; d3 = p[idx+(size_t)2*h*w+w]; }
        if(y!=0 && x!=0)    {                                           e3  = p[idx+(size_t)2*h*w-w-1]; }
        //if((y+1)!=h && (x+1)!=h)    {                                   e3  = p[idx+(size_t)2*h*w+w+1]; }

        d[idx] = sqrtf(1.0f/3.0f)*( a1 + b1 + c1 + d1 - 4*pp )
               + sqrtf(2.0f/3.0f)*( c2 + d2 - a2 - b2 )
               + sqrtf(8.0f/3.0f)*( pp + e3 - a3 - c3 );
               //+ sqrtf(8.0f/3.0f)*( pp + e3 - b3 - d3 );
    }
}


__global__ void ComputeU(float *v, float *d, float *u, int w, int h, float theta)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;
        
        u[idx] = v[idx] - theta*d[idx];
    }
}


__global__ void DiamondOperator(float *u, float *dd, int w, int h)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;

        float a = 0.0f;
        float b = 0.0f;
        float c = 0.0f;
        float d = 0.0f;
        float e = 0.0f;

        if(x!=0)                    { a = u[idx-1]; }
        if((x+1)!=w)                { b = u[idx+1]; }
        if(y!=0)                    { c = u[idx-w]; }
        if((y+1)!=h)                { d = u[idx+w]; }
        if((y+1)!=h && (x+1)!=w)    { e = u[idx+w+1]; }
        //if(y!=0 && x!=0)            { e = u[idx-w-1]; }

        dd[idx]               = sqrtf(1.0f/3.0f)*( a + b + c + d - 4*u[idx] );
        dd[idx+(size_t)h*w]   = sqrtf(2.0f/3.0f)*( c + d - a - b );
        dd[idx+(size_t)2*h*w] = sqrtf(8.0f/3.0f)*( u[idx] + e - b - d );
        //dd[idx+(size_t)2*h*w] = sqrtf(8.0f/3.0f)*( u[idx] + e - a - c );
    }
}


__global__ void ComputeDualVariable(float *dd, float *p, int w, int h, float theta, float tau)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;

        float d = dd[idx];
        float p1 = p[idx] + (tau/theta)*d;
        float p2 = p[idx+(size_t)h*w] + (tau/theta)*d;
        float p3 = p[idx+(size_t)2*h*w] + (tau/theta)*d;
        float maxDenom = fmax(1, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
        
        p[idx]                = p1/maxDenom;
        p[idx+(size_t)h*w]    = p2/maxDenom;
        p[idx+(size_t)2*h*w]  = p3/maxDenom;
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
	if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <rawfile> [-blockX <blockX>] [-blockY <blockY>] [-blockZ <blockZ>]" << endl; return 1;}
#endif

	// Default setting for block sizes
	size_t blockX = 32, blockY = 8, blockZ = 1;
	getParam("blockX", blockX, argc, argv);
	getParam("blockY", blockY, argc, argv);
	getParam("blockZ", blockZ, argc, argv);
	cout << "blocksize: " << blockX << "x" << blockY << "x" << blockZ << endl;

    int width = 640;
    getParam("width", width, argc, argv);
    cout << "width: " << width << endl;

    int height = 480;
    getParam("height", height, argc, argv);
    cout << "height: " << height << endl;

    float theta = 100.0f;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

    float tau = 0.005f;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    float decay = 0.975f;
    getParam("decay", decay, argc, argv);
    cout << "decay: " << decay << endl;

    int N = 2500;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

#ifdef KINECT
// Codes to read from Kinect

#else
    // Load the raw file (Size must be 640x480 == IMG_WIDTH*IMG_HEIGHT)
	uint16_t *depth = new uint16_t[width*height];
	ifstream file_buf(rawfile.c_str(), ios_base::binary);
	file_buf.read((char*) depth, width*height*sizeof(uint64_t));
    file_buf.close();

	// Find Maximum and convert it to float
	float *fInDepth = new float[width*height];
	uint16_t maxValue = 0;
	for (int y = 0; y < height; y++)
    {
        size_t offset = (size_t)y*width;
		for (int x = 0; x < width; x++)
			if (maxValue < depth[x + offset]) maxValue = depth[x + offset];
    }

    // Normalize the input data
	for (int y = 0; y < height; y++)
    {
        size_t offset = (size_t)y*width;
		for (int x = 0; x < width; x++)
			fInDepth[x + offset] = (float)depth[x + offset] / (float)maxValue;
    }

    // Setup input image
	cv::Mat mInDepth(height,width,CV_32FC1);
	convert_layered_to_mat(mInDepth, (const float*) fInDepth);

    // Setup output image
    float *fOutDepth = new float[(size_t)width*height];
	cv::Mat mOutDepth(height,width,CV_32FC1);
    
#endif

    // Start the timer for the GPU process
    Timer timer;
    timer.start();

    // Allocate memory on the GPU and copy data
    float *dU, *dV, *dP, *dD, *dDiaD;
    cudaMalloc(&dU, (size_t)width*height*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dV, (size_t)width*height*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dD, (size_t)width*height*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dP, (size_t)3*width*height*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dDiaD, (size_t)3*width*height*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(dU, fInDepth, (size_t)width*height*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(dV, dU, (size_t)width*height*sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
    cudaMemset(dD, 0, (size_t)width*height*sizeof(float)); CUDA_CHECK;
    cudaMemset(dP, 0, (size_t)3*width*height*sizeof(float)); CUDA_CHECK;
    cudaMemset(dDiaD, 0, (size_t)3*width*height*sizeof(float)); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(blockX, blockY, blockZ);
    dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);

    for(int n=0; n<N; n++)
    {
        DiamondDotProduct<<<grid, block>>>(dP, dD, width, height);
        cudaDeviceSynchronize();

        theta *= decay;
        ComputeU<<<grid, block>>>(dV, dD, dU, width, height, theta);
        cudaDeviceSynchronize();
        
        DiamondOperator<<<grid, block>>>(dU, dDiaD, width, height);
        cudaDeviceSynchronize();

        ComputeDualVariable<<<grid, block>>>(dDiaD, dP, width, height, theta, tau);
        cudaDeviceSynchronize();
    }

    DiamondDotProduct<<<grid, block>>>(dP, dD, width, height);
    cudaDeviceSynchronize();

    //theta *= decay;
    ComputeU<<<grid, block>>>(dV, dD, dU, width, height, theta);
    
    // Copy data back to CPU
    cudaMemcpy(fOutDepth, dU, (size_t)width*height*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    // Deallocate memory on the GPU
    cudaFree(dU); CUDA_CHECK;
    cudaFree(dV); CUDA_CHECK;
    cudaFree(dP); CUDA_CHECK;
    cudaFree(dD); CUDA_CHECK;
    cudaFree(dDiaD); CUDA_CHECK;

    // Display images
	showImage("Input Depth Image", mInDepth, 40, 100);
    convert_layered_to_mat(mOutDepth, fOutDepth);
	showImage("Output Depth Image", mOutDepth, 100+width, 100);

    // Save images
    cv::imwrite("image_input.png",mInDepth*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_output.png",mOutDepth*255.f);

    // End the timer for the GPU process
    timer.end();
    float t = timer.get();  // Time in seconds
    cout << "GPU time: " << t*1000 << " ms" << endl;

#ifdef KINECT
#else
	// wait for key input to quit
	cv::waitKey(0);
#endif

	// free allocated arrays
	delete[] depth;
	delete[] fInDepth;
	delete[] fOutDepth;
	
	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}

