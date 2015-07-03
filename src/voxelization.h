#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <iostream>
#include <glm/glm.hpp>
//#include <GL/glut.h>
#include <GL/glew.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "sceneStructs.h"
#include "timingUtils.h"
#include "commoninclude.h"

using namespace std;

#define THREADS_PER_BLOCK (512)

#define envmap_grid_size (64)
#define envmap_w (4 * envmap_grid_size)
#define envmap_h (3 * envmap_grid_size)
#define envmap_color_grid_size (64)
#define envmapcolor_w (4 * envmap_color_grid_size)
#define envmapcolor_h (3 * envmap_color_grid_size)

// Voxelization resolution
const int log_N = 6;
const int log_T = 3; ///Can only be > 3 when in BIT_MODE
const int N = 1 << log_N; //N is the total number of voxels (per dimension)
const int M = 1 << (log_N - log_T); //M is the total number of tiles (per dimension)
const int T = 1 << log_T; //T is the tile size - voxels per tile (per dimension)

//1/2 edge length of the world cube
static float world_size = 2.5f;
//Compute the 1/2 edge length for the resulting voxelization
const float vox_size = world_size / float(N);
const float CUBE_MESH_SCALE = 0.1f;



// voxelization
__host__ int voxelizeMesh(Mesh &m_in, bmp_texture* h_tex, int* d_voxels, int* d_values);

__host__ void extractCubesFromVoxelGrid(int* d_voxels, int numVoxels, int* d_values, Mesh &m_cube, Mesh &m_out);

__host__ void voxelizeToCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out, vector<glm::vec3> selectedInternal);


// photon.h
struct Photon{
	float position[3];
	float direction[3]; // also serves as v
	float residual_radiance[3]; // rgb radiance. this also serves as screen coordinate when marching view rays
	//float is_dead; // deprecated, just exit the while loop if it is dead
//#define DEBUG_PATH
#ifdef DEBUG_PATH
	float iteration_count;
#endif
};

struct Node{
	// min and max are normed to [0,1]
	float min; // serve as absorption_coefficient in level 0
	float max; // serve as scattering_coefficient in level 0
	float level;
};

struct Voxel{
	float color[3];
	float radiance[3]; // rgb radiance
	float direction[3]; // light path direction in this voxel, this is not normed to [0,1] because of accumulation
	float gradient[3];
	float absorption_coefficient;
	float scattering_coefficient;
	float is_occupied;
};



__host__ int CreateMipmap(float threshold);

__host__ int GeneratePhotons(bool is_light_photon, GLuint renderedTexture, glm::vec3& lightpos, glm::vec3& light_radiance, int& step_len
#ifdef DEBUG_PATH
	, vector<glm::vec3> *photons_debug
#endif
	);


__host__ int MarchPhotons(bool is_light_photon, int photon_num, int step_len, int minimum_march_grid_count, int maximum_march_grid_count
#ifdef DEBUG_PATH
	, vector<glm::vec3> *paths
#endif
	);

__host__ void SetCudaSideParams(glm::vec3 meshBoundingBoxMin_in, glm::vec3 meshBoundingBoxMax_in);

__host__ GLuint GetEnvironmentMapTex(bool is_update);
__host__ GLuint GetViewMapTex(bool is_update);
__host__ GLuint GetViewMapColorTex(bool is_update);


__host__ void OnStarting_Voxel();
__host__ void OnExiting_Voxel();
__host__ void initCudaSideTexture(GLuint background_tex);

__host__ void static printLastError();


#endif ///VOXELIZATION_H
