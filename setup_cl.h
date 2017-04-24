
#pragma once

#include <CL\opencl.h>

cl_context createContext();

cl_device_id getDeviceForContext(cl_context context, int deviceIndex = 0);

cl_command_queue createCommandQueue(cl_context context, cl_device_id *device);

cl_program createProgram(cl_context context, cl_device_id device, const char* fileName);
