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

__global__ void Boobs(float *u_in, float *u_out, int w, int h)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<w && y<h)
    {
        size_t idx = x + (size_t)y*w;
        
        u_out[idx] = u_in[idx];
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
    float *dImgIn, *dImgOut;
    cudaMalloc(&dImgIn, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&dImgOut, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(dImgIn, fInDepth, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(dImgOut, 0, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float)); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(blockX, blockY, blockZ);
    dim3 grid = dim3((IMG_WIDTH+block.x-1)/block.x, (IMG_HEIGHT+block.y-1)/block.y, 1);

    // GPU computation
    Boobs<<<grid, block>>>(dImgIn, dImgOut, IMG_WIDTH, IMG_HEIGHT);
    
    // Copy data back to CPU
    cudaMemcpy(fOutDepth, dImgOut, (size_t)IMG_WIDTH*IMG_HEIGHT*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    // Deallocate memory on the GPU
    cudaFree(dImgIn); CUDA_CHECK;
    cudaFree(dImgOut); CUDA_CHECK;

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

