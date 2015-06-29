#ifndef _COMMON_INCLUDE_H_
#define _COMMON_INCLUDE_H_

#include <cuda.h>

#define PATH_PREFIX	"C:\\Users\\ying\\Desktop\\Dynamic-Refractive-Relighting\\windows\\"

#define MOUSE_SPEED 2.0*0.0001f
#define ZOOM_SPEED 8
#define MIDDLE_SPEED 12

#define USE_CUDA_RASTERIZER 0
#define OCTREE 0


#define W_WIDTH (1024)
#define W_HEIGHT (576)


#define PHOTON_LOWER_LIMIT (1.0/1000)

#define QUADRATIC_N
//#define CONSTANT_N

#define ABSORPTION 0.1
#define SCATTERING 0.1

#define COLOR

#endif