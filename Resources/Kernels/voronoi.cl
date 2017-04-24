
// Voronoi kernels

// Voronoi region struct.  This reflects the C/C++ struct that adheres to log-base2 type alignment outlined in section 6.1.5 of the OpenCL 1.1 specification document.

typedef struct _voronoi_region {
	
	float2		pos; // Position of region (in normalised image coordinates [0, 1]) - float4 alignment
	float3		colour; // Colour of region - float4 alignment
} voronoi_region;

typedef struct _parameters
{
	float		frequency;
	float		startTheta;
	float		endTheta;

} parameters;

typedef struct _julia_vars {

	float2		component;
	float2		zRange;
	float2		threshhold;
} julia_vars;

kernel void voronoi(const global voronoi_region *R, write_only image2d_t outputImage, const int numRegions)
{
	// Get id of element in array
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	
	// Calculate normalised coordinates of current pixel in [0, 1] range
	float4 P;
	P.x = (float)(x) / (float)(w-1);
	P.y = (float)(y) / (float)(h-1);
	P.z = 0.0f;
	P.w = 0.0f;
	
	int s = 0;
	
	float minDist = length(R[s].pos.xy - P.xy); // Euclidean distance metric used here
	//float minDist = fabs(R[s].pos.x - P.x) + fabs(R[s].pos.y - P.y); // "Manhattan distance"
	
	for (int i=1; i<numRegions; i++) {

		float dist = length(R[i].pos.xy - P.xy); // Euclidean distance metric used here
		//float dist = fabs(R[i].pos.x - P.x) + fabs(R[i].pos.y - P.y); // "Manhattan distance"
		
		if (dist < minDist) {
			s = i;
			minDist = dist;
		}
	}

	float4 C;

	C.x = R[s].colour.x;
	C.y = R[s].colour.y;
	C.z = R[s].colour.z;
	C.w = 1.0;

	write_imagef(outputImage, (int2)(x, y), C);
}

kernel void tutorial(const global parameters *I, write_only image2d_t outputImage2)
{
	// Get id of element in array
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	float range = (float)(I[0].endTheta);// -I[0].startTheta);
	float ttlPixels = (w*h);

	// Calculate normalised coordinates of current pixel in [0, 1] range
	float4 P;
	P.x = (float)(x) / (float)(w-1);
	P.y = (float)(y) / (float)(h-1);
	P.z = 0.0f;
	P.w = 0.0f;

	//value = cos(theta * frequency) * sin(phi * frequency)
	
	float theta = (P.x) * (6.28-0)+0;
	float phi = (P.y) * (6.28-0)+0;

	float4 result;
	result.x = cos(theta * I[0].frequency) *sin(phi * I[0].frequency);
	result.y = cos(theta * I[0].frequency) *sin(phi * I[0].frequency);
	result.z = cos(theta * I[0].frequency) *sin(phi * I[0].frequency);
	result.w = 1.0;

	write_imagef(outputImage2, (int2)(x, y), result);
}

kernel void juliaSet(const global julia_vars *R, write_only image2d_t outputImage, const int numIterations, const float CX, const float CY, const float RX, const float RY, const float T, const int P)
{
	// Get id of element in array
	int x = get_global_id(0);
	int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);

	//zk+1 = zk^2 + c

	//local variables
	float4 result; //holds the output colour of the pixel
	//set up initial z value in range given
	//float2 Z = {RX * (-w / 2 + x)/(w * 0.5) , RY * (-h / 2 + y)/(h * 0.5)};
	float2 Z = {(((float)(x) / (float)(w - 1)) * (RY - RX) + RX), (((float)(y) / (float)(h - 1)) * (RY - RX) + RX)};
	//set up const c value
	float2 C = { CX, CY};//-0.805, 0.156

	int currentIteration = 0;

	while (currentIteration < numIterations)
	{
		//mult complex = (a.x*b.x - a.y*b.y), (a.x*b.y + a.y*b.x)
		float2 ZxZ = (float2)((Z.x*Z.x - Z.y*Z.y), (Z.x*Z.y + Z.y*Z.x));
		//we have z^2 so power again every time after 2
		for (int power = 2; power < P; power++)//runs pow-2 times 3 = 1, 4 = 2 etc.
		{
			//mult complex = (a.x*b.x - a.y*b.y), (a.x*b.y + a.y*b.x)
			ZxZ = (float2)((ZxZ.x * Z.x - ZxZ.y*Z.y), (ZxZ.x*Z.y + ZxZ.y*Z.x));
		}

		//add complex = (Z.x+C.x),(Z.y+C.y)
		float2 ZxZpC = (float2)(ZxZ.x + C.x, ZxZ.y + C.y);
		Z.x = ZxZpC.x;
		Z.y = ZxZpC.y;
		currentIteration ++;
		//length of complex = sqrt((Z.x*Z.x) + (Z.y*Z.y))
		float tmpLength = sqrt((Z.x*Z.x) + (Z.y*Z.y));
		if (tmpLength > T)//  2.0 
		{
			break;
		}
	}
	if (currentIteration == numIterations)
	{
		//set colour to black
		result.x = 0.0f;
		result.y = 0.0f;
		result.z = 0.0f;
		result.w = 0.0f;
	}
	else
	{
		//set colour to given colour of the 6 given:
		int tmpColour = (int)(currentIteration % 6);
		//r
		if (tmpColour == 0)
		{
			result.x = 1.0f;
			result.y = 0.0f;
			result.z = 0.0f;
		}
		//o
		else if (tmpColour == 1)
		{
			result.x = 1.0f;
			result.y = 0.5f;
			result.z = 0.0f;
		}
		//y
		else if (tmpColour == 2)
		{
			result.x = 1.0f;
			result.y = 1.0f;
			result.z = 0.0f;
		}
		//g
		else if (tmpColour == 3)
		{
			result.x = 0.0f;
			result.y = 1.0f;
			result.z = 0.0f;
		}
		//b
		else if (tmpColour == 4)
		{
			result.x = 0.0f;
			result.y = 0.5f;
			result.z = 1.0f;
		}
		//p
		else if (tmpColour == 5)
		{
			result.x = 0.5f;
			result.y = 0.0f;
			result.z = 1.0f;
		}
		result.w = 1.0f;
	}

	write_imagef(outputImage, (int2)(x, y), result);
}