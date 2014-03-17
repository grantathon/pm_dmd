// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ### 
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###
// ### Project Phase: Boundary and Edge Detection using Kinect Depth Images
// ###
// ### Group 8
// ### 
// ### Xiao HUANG, xiao.huang@tum.de, p071
// ### Xiao, XUE, xuexiao1989@gmail.com, p072
// ### Sing Chun, LEE, leesingchun@gmail.com, p077
// ###
// ###


#include "aux.h"	// Helping functions for CUDA GPU Programming
#include <iostream>	// For standard IO on console
#include <fstream>	// For reading raw depth file

using namespace std;

// Uncomment to use the live Kinect Camera
//#define KINECT
#define IMG_WIDTH 640
#define IMG_HEIGHT 480

__global__ void DiamondDotProduct(float *p, float *d, int w, int h)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;
        float pp = p[idx];
        float a1 = 0.0f;    float a2 = 0.0f;    float a3 = 0.0f;
        float b1 = 0.0f;    float b2 = 0.0f;
        float c1 = 0.0f;    float c2 = 0.0f;    float c3 = 0.0f;
        float d1 = 0.0f;    float d2 = 0.0f;
                                                float e3 = 0.0f;

        if(x!=0)            { a1 = p[idx-1]; a2 = p[idx+(size_t)h*w-1]; a3 = p[idx+(size_t)2*h*w-1]; }
        if((x+1)!=w)        { b1 = p[idx+1]; b2 = p[idx+(size_t)h*w+1]; }
        if(y!=0)            { c1 = p[idx-w]; c2 = p[idx+(size_t)h*w-w]; c3 = p[idx+(size_t)2*h*w-w]; }
        if((y+1)!=h)        { d1 = p[idx+w]; d2 = p[idx+(size_t)h*w+w]; }
        if(y!=0 && x!=0)    {                                           e3  = p[idx+(size_t)2*h*w-w-1]; }

        d[idx] = sqrtf(1.0f/3.0f)*( a1 + b1 + c1 + d1 - 4*pp )
               + sqrtf(2.0f/3.0f)*( c2 + d2 - a2 - b2 )
               + sqrtf(8.0f/3.0f)*( pp + e3 - a3 - c3 );

    }
}


__global__ void ComputeU(float *u, float *v, float *d, int w, int h, float theta)
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

        dd[idx]               = sqrtf(1.0f/3.0f)*( a + b + c + d - 4*u[idx] );
        dd[idx+(size_t)h*w]   = sqrtf(2.0f/3.0f)*( c + d - a - b );
        dd[idx+(size_t)2*h*w] = sqrtf(8.0f/3.0f)*( u[idx] + e - b - d );
    }
}


__global__ void ComputeDualVariable(float *dd, float *p, int w, int h, float theta, float tau)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;
        float c = tau/theta;
        float d = dd[idx];
        
        for(int j=0; j<3; j++)
        {
            size_t idxOffset = idx + (size_t)j*h*w;

            float p1 = p[idxOffset] + c*d;
            float p2 = p[idxOffset+(size_t)h*w] + c*d;
            float p3 = p[idxOffset+(size_t)2*h*w] + c*d;
            
            p[idxOffset]                = p1/fmax(1, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
            p[idxOffset+(size_t)h*w]    = p2/fmax(1, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
            p[idxOffset+(size_t)2*h*w]  = p3/fmax(1, sqrtf(powf(p1, 2) + powf(p2, 2) + powf(p3, 2)));
            /*
            p[idxOffset]                = p1/fmax(1, fabs(p1));
            p[idxOffset+(size_t)h*w]    = p2/fmax(1, fabs(p2));
            p[idxOffset+(size_t)2*h*w]  = p3/fmax(1, fabs(p3));
            */
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
	if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <rawfile> [-blockX <blockX>] [-blockY <blockY>] [-blockZ <blockZ>]" << endl; return 1;}
#endif

	// Default setting for block sizes
	size_t blockX = 32, blockY = 8, blockZ = 1;
	getParam("blockX", blockX, argc, argv);
	getParam("blockY", blockY, argc, argv);
	getParam("blockZ", blockZ, argc, argv);
	cout << "blocksize: " << blockX << "x" << blockY << "x" << blockZ << endl;

    int theta = 50;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

    float tau = 0.02f;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    float decay = 0.95f;
    getParam("decay", decay, argc, argv);
    cout << "decay: " << decay << endl;

    int N1 = 20;
    getParam("N1", N1, argc, argv);
    cout << "N1: " << N1 << endl;

    int N2 = 10;
    getParam("N2", N2, argc, argv);
    cout << "N2: " << N2 << endl;

#ifdef KINECT
// Codes to read from Kinect

#else
    // Load the raw file (Size must be 640x480 == IMG_WIDTH*IMG_HEIGHT)
	uint16_t *depth = new uint16_t[IMG_WIDTH*IMG_HEIGHT];
	ifstream file_buf(rawfile.c_str(), ios_base::binary);
	file_buf.read((char*) depth, IMG_WIDTH*IMG_HEIGHT*sizeof(uint64_t));
    file_buf.close();

	// Find Maximum and convert it to float
	float *fInDepth = new float[IMG_WIDTH*IMG_HEIGHT];
	uint16_t maxValue = 0;
	for (size_t y = 0; y < IMG_HEIGHT; y++)
    {
        size_t offset = y*IMG_WIDTH;
		for (size_t x = 0; x < IMG_WIDTH; x++)
			if (maxValue < depth[x + offset]) maxValue = depth[x + offset];
    }

    // Normalize the input data
	for (size_t y = 0; y < IMG_HEIGHT; y++)
    {
        size_t offset = y*IMG_WIDTH;
		for (size_t x = 0; x < IMG_WIDTH; x++)
			fInDepth[x + offset] = (float)depth[x + offset] / (float)maxValue;
    }

    // Setup input image
	cv::Mat mInDepth(IMG_HEIGHT,IMG_WIDTH,CV_32FC1);
	convert_layered_to_mat(mInDepth, (const float*) fInDepth);

    // Setup output image
    float *fOutDepth = new float[(size_t)IMG_WIDTH*IMG_HEIGHT];
	cv::Mat mOutDepth(IMG_HEIGHT,IMG_WIDTH,CV_32FC1);
    
#endif

    // Start the timer for the GPU process
    Timer timer;
    timer.start();

    // Allocate memory on the GPU and copy data
    float *dU, *dV, *dP, *dD, *dDiaD;
    cudaMalloc(&dU, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dV, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dD, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dP, (size_t)3*IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dDiaD, (size_t)3*IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(dU, fInDepth, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(dV, dU, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
    cudaMemset(dD, 0, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMemset(dP, 0, (size_t)3*IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMemset(dDiaD, 0, (size_t)3*IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(blockX, blockY, blockZ);
    dim3 grid = dim3((IMG_WIDTH+block.x-1)/block.x, (IMG_HEIGHT+block.y-1)/block.y, 1);

    for(int n1=0; n1<N1; n1++)
    {
        for(int n2=0; n2<N2; n2++)
        {
            DiamondDotProduct<<<grid, block>>>(dP, dD, IMG_WIDTH, IMG_HEIGHT);
            cudaDeviceSynchronize();

            ComputeU<<<grid, block>>>(dU, dV, dD, IMG_WIDTH, IMG_HEIGHT, theta);
            cudaDeviceSynchronize();
            
            DiamondOperator<<<grid, block>>>(dU, dDiaD, IMG_WIDTH, IMG_HEIGHT);
            cudaDeviceSynchronize();

            ComputeDualVariable<<<grid, block>>>(dDiaD, dP, IMG_WIDTH, IMG_HEIGHT, theta, tau);
            cudaDeviceSynchronize();
        }

        theta *= decay;
    }

    DiamondDotProduct<<<grid, block>>>(dP, dD, IMG_WIDTH, IMG_HEIGHT);
    cudaDeviceSynchronize();

    ComputeU<<<grid, block>>>(dU, dV, dD, IMG_WIDTH, IMG_HEIGHT, theta);
    
    // Copy data back to CPU
    cudaMemcpy(fOutDepth, dU, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    // Deallocate memory on the GPU
    cudaFree(dU); CUDA_CHECK;
    cudaFree(dV); CUDA_CHECK;
    cudaFree(dP); CUDA_CHECK;
    cudaFree(dD); CUDA_CHECK;
    cudaFree(dDiaD); CUDA_CHECK;

    // Display images
	showImage("Input Depth Image", mInDepth, 40, 100);
    convert_layered_to_mat(mOutDepth, fOutDepth);
	showImage("Output Depth Image", mOutDepth, 100+IMG_WIDTH, 100);

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

