
//
// Voronoi example
//

#include <cstdio>
#include <iostream>
#include <random>
#include <exception>
#include <malloc.h>
#include <CL\opencl.h>
#include "setup_cl.h"
#include <FreeImage\FreeImagePlus.h>
#include <sstream>


using namespace std;


#pragma region Supporting structures to reflect vector types in OpenCL

struct float2 {
	float x, y;
	float2(const float _x, const float _y) : x(_x), y(_y) {}
};

struct float3 {
	float x, y, z;
	float3(const float _x, const float _y, const float _z) : x(_x), y(_y), z(_z) {}
};

struct float4 {
	float x, y, z, w;
	float4(const float _x, const float _y, const float _z, const float _w) : x(_x), y(_y), z(_z), w(_w) {}
};

#pragma endregion

//=======================================================================
__declspec(align(16)) struct julia_vars
{
	__declspec(align(16)) float2	component;
	__declspec(align(16)) float2	zRange;
	__declspec(align(16)) float2	threshhold;
};
//juliaset variables
//kernel void juliaSet(=====juliavars=====, write_only image2d_t outputImage, const int numIterations, const float threshhold)
cl_float	componentX = -0.805f;
cl_float	componentY = 0.156f;
cl_float	zRangeX = -1.77;
cl_float	zRangeY = 1.77;
cl_float	threshhold = 2.0f;
cl_int		power = 2;//-1 for power use
const cl_int iterations = 250;

julia_vars			*input = nullptr;
cl_mem				courseworkBuffer = 0;

int imageWidth = 1024;
int imageHeight = 1024;
//=======================================================================

int main(int argc, char **argv) {
	cl_int				err = 0;
	cl_context			context = 0;
	cl_program			program = 0;
	cl_kernel			voronoiKernel = 0;
	cl_mem				regionBuffer = 0;
	cl_mem				outputImage = 0;
	cl_device_id		device = 0;
	cl_command_queue	commandQueue = 0;
	fipImage			result;

	//=======================================================================
	float				accPi = 3.1415;
	float				frequency = 2.0f;
	float				start = 0.0f;
	float				end = 2.0 * accPi;
	cl_mem				tutorialBuffer = 0;
	//=======================================================================

	try
	{
		// Setup random number engine
		random_device rd;
		mt19937 mt(rd());
		auto D = uniform_real_distribution<float>(0.0f, 1.0f);

		//=======================================================================
		input = (julia_vars*)_aligned_malloc(sizeof(julia_vars), 16);
		if (!input) throw exception("cannot create input array in host memory");
		input[0].component.x = componentX; 
		input[0].component.y = componentY; 
		input[0].zRange.x = zRangeX; 
		input[0].zRange.y = zRangeY; 
		input[0].threshhold.x = threshhold; 
		//=======================================================================		

		// Create and validate the OpenCL context
		context = createContext();
		if (!context)throw exception("Cannot create OpenCL context");
		// Get the first device associated with the context - should be the GPU
		device = getDeviceForContext(context);
		if (!device)throw exception("Cannot obtain valid device ID");
		// Create the command queue
		commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, 0);
		if (!commandQueue)throw exception("Cannot create command queue");
		// Create the program object based on voronoi.cl
		cl_program program = createProgram(context, device, "Resources\\Kernels\\voronoi.cl");
		if (!program)throw exception("Cannot create program object");

		//=======================================================================
		// Get the kernel from program object created above
		voronoiKernel = clCreateKernel(program, "juliaSet", 0);
		if (!voronoiKernel)throw exception("Could not create kernel");
		courseworkBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(julia_vars), input, 0);
		if (!courseworkBuffer) throw exception("Cannot create coursework buffer");
		//=======================================================================

		// Setup output image
		cl_image_format outputFormat;
		outputFormat.image_channel_order = CL_BGRA;
		outputFormat.image_channel_data_type = CL_UNORM_INT8;
		outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &outputFormat, imageWidth, imageHeight, 0, 0, &err);
		if (!outputImage)throw exception("Cannot create output image object");

		//setup timer vars
		cl_ulong startTime = (cl_ulong)0;
		cl_ulong endTime = (cl_ulong)0;
		// Setup event (for profiling)
		cl_event voronoiEvent;

		// Setup memory object -> kernel parameter bindings
		//=======================================================================
		clSetKernelArg(voronoiKernel, 0, sizeof(cl_mem), &courseworkBuffer);
		clSetKernelArg(voronoiKernel, 1, sizeof(cl_mem), &outputImage);
		clSetKernelArg(voronoiKernel, 2, sizeof(cl_int), &iterations);
		clSetKernelArg(voronoiKernel, 3, sizeof(cl_float), &componentX);
		clSetKernelArg(voronoiKernel, 4, sizeof(cl_float), &componentY);
		clSetKernelArg(voronoiKernel, 5, sizeof(cl_float), &zRangeX);
		clSetKernelArg(voronoiKernel, 6, sizeof(cl_float), &zRangeY);
		clSetKernelArg(voronoiKernel, 7, sizeof(cl_float), &threshhold);
		clSetKernelArg(voronoiKernel, 8, sizeof(cl_int), &power);
		//=======================================================================
		// Setup worksize arrays
		size_t globalWorkSize[2] = { imageWidth, imageHeight };
		//set local worksize?
		size_t localWorkSize[2] = {1,1};
		// Enqueue kernel
		err = clEnqueueNDRangeKernel(commandQueue, voronoiKernel, 2, 0, globalWorkSize, localWorkSize, 0, 0, &voronoiEvent);
		if (err)throw exception("Error enqueueing kernel");
		
		//get local group size
		size_t sizeInfo;
		clGetKernelWorkGroupInfo(voronoiKernel, device, CL_KERNEL_WORK_GROUP_SIZE, 0, 0, &sizeInfo);
		int *infoSize = (int*)_aligned_malloc(sizeof(sizeInfo), 16);
		clGetKernelWorkGroupInfo(voronoiKernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeInfo, infoSize, 0);

		// Block until voronoi kernel finishes and report time taken to run the kernel
		clWaitForEvents(1, &voronoiEvent);
		clGetEventProfilingInfo(voronoiEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, 0);
		clGetEventProfilingInfo(voronoiEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, 0);
		double tdelta = (double)(endTime - startTime);
		cout << "Dim (" << imageWidth << "x" << imageHeight << "), Z (" << zRangeX << "," << zRangeY << ") ^ " << power << " + C (" << componentX << "," << componentY << ")";
		cout << "LGS: " << *infoSize << endl;
		cout << "Time taken(s): " << (tdelta * 1.0e-9) << endl;
		// Extract the resulting voronoi diagram image from OpenCL
		result = fipImage(FREE_IMAGE_TYPE::FIT_BITMAP, imageWidth, imageHeight, 32);
		if (!result.isValid())throw exception("Cannot create the output image");
		size_t origin[3] = { 0, 0, 0 };
		size_t region[3] = { imageWidth, imageHeight, 1 };
		err = clEnqueueReadImage(commandQueue, outputImage, CL_TRUE, origin, region, 0, 0, result.accessPixels(), 0, 0, 0);
		result.convertTo24Bits();
		BOOL saved = result.save("TEST.jpg");
		if (!saved)throw exception("Cannot save voronoi diagram"); 


		// Dispose of resources
		clReleaseMemObject(regionBuffer);
		clReleaseMemObject(courseworkBuffer);//=====added
		clReleaseMemObject(outputImage);
		clReleaseKernel(voronoiKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
		if (input)_aligned_free(input);//=====added
		return 0;
	}
	catch (exception& err)
	{
		// Output the exception message to the console
		cout << err.what() << endl;
		// Dispose of resources
		clReleaseMemObject(regionBuffer);
		clReleaseMemObject(courseworkBuffer);//=====added
		clReleaseMemObject(outputImage);
		clReleaseKernel(voronoiKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
		if (input)_aligned_free(input);//=====added
		// Done - report error
		return 1;
	}
}
