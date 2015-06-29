#if 0
#ifndef PHOTON_H
#define PHOTON_H


#include <glm/glm.hpp>
#include <GL/glut.h>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "sceneStructs.h"
#include "timingUtils.h"
#include "commoninclude.h"

using namespace std;


struct Photon{
	float position[3];
	float direction[3]; // also serves as v
	float residual_radiance;
};

struct Node{
	// min and max are normed to [0,1]
	float min; // serve as original value in level 0
	float max; // at first is same as min, later serve as extinction coefficient in level 0
	float level;
};


//__host__ int GeneratePhotons(GLuint renderedTexture, glm::vec3& lightpos, Photon** out, 
	//glm::vec3 meshBoundingMin, glm::vec3 meshBoundingMax);

#endif ///VOXELIZATION_H
#endif