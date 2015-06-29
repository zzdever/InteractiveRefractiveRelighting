
#include "voxelization.h"
#include <glm/glm.hpp>
//#include <GL/glut.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <voxelpipe/voxelpipe.h>

#include "timingUtils.h"


#define VOXEL_CUBE_COLOR ((116) + (233 << 8) + (229 << 16) + (127 << 24))
// Cuda side global data
// remember to free this on program exiting
Node *mipmap;
Photon* photon_list;
Voxel* mesh_voxels;

float* environment_map; // device side
struct cudaGraphicsResource* envmap_shared;
float* environment_map_color; // device side
GLfloat* envmap_texture; // host side
GLuint envmap_texture_id;
float* view_map; // _RGB, device side
float* view_map_color; // _RGB color, device side
GLfloat* view_map_texture;
GLfloat* view_map_color_texture;
GLuint view_map_texture_id, view_map_color_texture_id;


float grid_size[3];
int maximum_march_grid_count;
int minimum_march_grid_count;


voxelpipe::FRContext<log_N, log_T>*  context;
bool first_time = true;


//Create bounding box to perform voxelization within
glm::vec3 meshBoundingBoxMin, meshBoundingBoxMax;
float3 mbbmin, mbbmax;


//Compute tile/grid sizes
const float3 t_d = make_float3((mbbmax.x - mbbmin.x) / float(M),
	(mbbmax.y - mbbmin.y) / float(M),
	(mbbmax.z - mbbmin.z) / float(M));

const float3 p_d = make_float3(t_d.x / float(T),
	t_d.y / float(T), t_d.z / float(T));



__device__ inline float3 getCenterFromIndex(int idx, int M, int T, float3 mbbmin, float3 t_d, float3 p_d) {
	int T3 = T*T*T;
	int tile_num = idx / T3;
	int pix_num = idx % T3;
	float3 cent;
	int tx = (tile_num / (M*M)) % M;
	int px = (pix_num / (T*T)) % T;
	int ty = (tile_num / M) % M;
	int py = (pix_num / T) % T;
	int tz = tile_num % M;
	int pz = pix_num % T;
	cent.z = mbbmin.x + tx*t_d.x + px*p_d.x + p_d.x / 2.0f;
	cent.y = mbbmin.y + ty*t_d.y + py*p_d.y + p_d.y / 2.0f;
	cent.x = mbbmin.z + tz*t_d.z + pz*p_d.z + p_d.z / 2.0f;
	return cent;
}


struct ColorShader
{
	glm::vec3* texture;
	int tex_width;
	int tex_height;
	float* texcoord;
	int texcoord_size;

	__device__ float shade(
		const int tri_id,
		const float4 v0,
		const float4 v1,
		const float4 v2,
		const float3 n,
		const float  bary0,
		const float  bary1,
		const int3   xyz) const
	{
		//If there is no texture, just return green
		if (tex_width == 0) {
			return __int_as_float(VOXEL_CUBE_COLOR);
		}

		//If there are no texcoordinates, just return the first value in the texture
		if (texcoord_size == 0) {
			int r = (int)(texture[0].r * 255.0);
			int g = (int)(texture[0].g * 255.0);
			int b = (int)(texture[0].b * 255.0);
			return __int_as_float(r + (g << 8) + (b << 16) + (127 << 24));
		}

		//Get the texture coordinates from the triangle id
		int t1_x = texcoord[6 * tri_id] * tex_width;
		int t1_y = texcoord[6 * tri_id + 1] * tex_height;
		int t2_x = texcoord[6 * tri_id + 2] * tex_width;
		int t2_y = texcoord[6 * tri_id + 3] * tex_height;
		int t3_x = texcoord[6 * tri_id + 4] * tex_width;
		int t3_y = texcoord[6 * tri_id + 5] * tex_height;

		//Get the colors from the texture at these vertices
		glm::vec3 c1 = texture[t1_y * tex_width + t1_x];
		glm::vec3 c2 = texture[t2_y * tex_width + t2_x];
		glm::vec3 c3 = texture[t3_y * tex_width + t3_x];

		//TODO: Interpolate using barycentric coordinates
		glm::vec3 color = c1;

		//Compute rgb components
		int r = (int)(clamp(color.r, 0.0f, 1.0f) * 255.0f);
		int g = (int)(clamp(color.g, 0.0f, 1.0f) * 255.0f);
		int b = (int)(clamp(color.b, 0.0f, 1.0f) * 255.0f);

		//Compact
		int val = r + (g << 8) + (b << 16) + (127 << 24);

		return __int_as_float(val);
	}
};

// getOccupiedVoxels<< <N*N*N, 256 >> >
// (thrust::raw_pointer_cast(&d_fb.front()), M, T, d_vox);
__global__ void getOccupiedVoxels(void* fb, int M, int T, int* voxels, int* selected) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < M*M*M*T*T*T){
		int ii = 0;
		while (selected[ii] >= 0){
			if (index == selected[ii]) {
				voxels[index] = 2;
				return;
			}
			ii++;
		}
		

		int alpha = __float_as_int(*((float*)fb + index)) >> 24;
		bool is_occupied = alpha > 0;

		if (is_occupied) {
			//voxels[index] = index;
			voxels[index] = 1;
		}
		else {
			voxels[index] = -1;
		}
	}


	/*

  int T3 = T*T*T;
  int M3 = M*M*M;

  int pix_num = (blockIdx.x * THREADS_PER_BLOCK % T3) + threadIdx.x;
  int tile_num = blockIdx.x * THREADS_PER_BLOCK / T3;

  if (pix_num < T3 && tile_num < M3) {
  //TODO: Is there any benefit in making this shared?
  float* tile;

  bool is_occupied;
  tile = (float*)fb + tile_num*T3;
  int alpha = __float_as_int(tile[pix_num]) >> 24;
  is_occupied = alpha > 0;

  if (is_occupied) {
  voxels[tile_num*T3 + pix_num] = tile_num*T3 + pix_num;
  } else {
  voxels[tile_num*T3 + pix_num] = -1;
  }
  }
  */
}

#define FILL_DEBUG_CONDITION (0)

__global__ void FillVoxels(int* voxels, int N, int* total)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < N*N*N){
		if (voxels[index] > 0)
			return;

		int T3 = 1 << ((log_T << 1) + log_T);

		int x, y, z;
		int gx, gy, gz;

		x = (((index >> ((log_T << 1) + log_T)) & (M - 1)) << log_T) + ((index & (T3 - 1))&(T - 1));
		y = ((((index >> ((log_T << 1) + log_T)) >> (log_N - log_T)) & (M - 1)) << log_T) + (((index & (T3 - 1)) >> log_T)&(T - 1));
		z = ((((index >> ((log_T << 1) + log_T)) >> ((log_N - log_T)<<1)) & (M - 1)) << log_T) + (((index & (T3 - 1)) >> (log_T<<1))&(T - 1));

		for (int dx = -1; dx <= 1; dx++){
			for (int dy = -1; dy <= 1; dy++){
				for (int dz = -1; dz <= 1; dz++){
					gx = x + dx;
					gy = y + dy;
					gz = z + dz;

					if (   gx < 0 || gx >= N
						|| gy < 0 || gy >= N
						|| gz < 0 || gz >= N) continue; // out of boundary


					int testing_offset = (((gx >> log_T) + ((gy >> log_T) << (log_N - log_T)) + ((gz >> log_T) << ((log_N - log_T) << 1))) << ((log_T << 1) + log_T))
						+ (gx&(T - 1)) + ((gy&(T - 1)) << log_T) + ((gz&(T - 1)) << (log_T << 1));
					
					if (voxels[testing_offset] == 2){
						voxels[index] = 2;
						atomicAdd(total, 1);
						return;
					}
				}
			}
		}
	}

}

__global__ void SetOccupiedVoxelsIndex(int* voxels, int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
									 
	if (index < N*N*N){
		if (voxels[index] > 0)
			voxels[index] = index;
	}
}

/*
__global__ void SetMeshOccupiedVoxels(void* fb, int N, Voxel* mesh_voxels, int selected)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < N*N*N){
		Voxel* v = mesh_voxels + index;

		if (index == selected) {
			v->is_occupied = 2.0;
			return;
		}

		int alpha = __float_as_int(*((float*)fb + index)) >> 24;
		bool is_occupied = (alpha > 0);
		if (is_occupied) {
			v->is_occupied = 1.0;
		}
		else {
			v->is_occupied = -1.0;
		}

		// TODO set other params here?
	}
}
*/


//Thrust predicate for removal of empty voxels
struct check_voxel {
	__host__ __device__
	bool operator() (const int& c) {
		return (c != -1);
	}
};


__global__ void extractValues(void* fb, int* voxels, int num_voxels, int* values) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < num_voxels) {
		//TODO: Make this support other storage_type's besides int32
		//float* tile = (float*)fb;
		//values[index] =  __float_as_int(tile[voxels[index]]);
		values[index] = VOXEL_CUBE_COLOR; 
	}
}


__global__ void createCubeMesh(int* voxels, int* values, int M, int T, float3 mbbmin, float3 t_d, float3 p_d, float scale_factor, int num_voxels, float* cube_vbo,
	int cube_vbosize, int* cube_ibo, int cube_ibosize, float* cube_nbo, float* out_vbo, int* out_ibo, float* out_nbo, float* out_cbo) {

	//Get the index for the thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < num_voxels) {

		int vbo_offset = idx * cube_vbosize;
		int ibo_offset = idx * cube_ibosize;
		float3 center = getCenterFromIndex(voxels[idx], M, T, mbbmin, t_d, p_d);

		int color = values[idx];

		for (int i = 0; i < cube_vbosize; i++) {
			if (i % 3 == 0) {
				out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.x;
				out_cbo[vbo_offset + i] = (float)((color & 0xFF) / 255.0);
			}
			else if (i % 3 == 1) {
				out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.y;
				out_cbo[vbo_offset + i] = (float)(((color >> 8) & 0xFF) / 255.0);
			}
			else {
				out_vbo[vbo_offset + i] = cube_vbo[i] * scale_factor + center.z;
				out_cbo[vbo_offset + i] = (float)(((color >> 16) & 0xFF) / 255.0);
			}
			out_nbo[vbo_offset + i] = cube_nbo[i];
		}

		for (int i = 0; i < cube_ibosize; i++) {
			out_ibo[ibo_offset + i] = cube_ibo[i] + ibo_offset;
		}

	}

}


__global__ void GetOccupiedVoxelsFilled(Voxel* mesh_voxels, int* d_voxels, int N, int* total)
{
	int numVoxels = 0;
	for (int index = 0; index < N*N*N; index++){
		if ((mesh_voxels + index)->is_occupied > 0) {
			d_voxels[numVoxels] = index;
			numVoxels++;
		}
	}
	(*total) = numVoxels;
}

__global__ void BuildMeshVoxels(Voxel* mesh_voxels, int* voxels, int M, int T, int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index < N*N*N) {
		int T3 = T*T*T;
		int x = (((index >> ((log_T << 1) + log_T)) & (M - 1)) << log_T) + ((index & (T3 - 1))&(T - 1));
		int y = ((((index >> ((log_T << 1) + log_T)) >> (log_N - log_T)) & (M - 1)) << log_T) + (((index & (T3 - 1)) >> log_T)&(T - 1));
		int z = ((((index >> ((log_T << 1) + log_T)) >> ((log_N - log_T) << 1)) & (M - 1)) << log_T) + (((index & (T3 - 1)) >> (log_T << 1))&(T - 1));

		Voxel* vv = mesh_voxels + x*N*N + y*N + z;

		if (y > N / 4){
			vv->color[0] = 0.0314;
			vv->color[1] = 0.1059;
			vv->color[2] = 0.6941;
		}
		else{
			vv->color[0] = 0.8549;
			vv->color[1] = 0.3137;
			vv->color[2] = 0.0784;
		}
		
		vv->radiance[0] = 0;
		vv->radiance[1] = 0;
		vv->radiance[2] = 0;
		vv->direction[0] = 0;
		vv->direction[1] = 0;
		vv->direction[2] = 0;
		vv->absorption_coefficient = ABSORPTION;
		vv->scattering_coefficient = SCATTERING;
		
		if (voxels[index] > 0)
			vv->is_occupied = 1.0;
		else
			vv->is_occupied = 0.0;
	}
}


__host__ int voxelizeMesh(Mesh &m_in, bmp_texture* h_tex, int* d_voxels, int* d_values, int* selectedInternal) 
{
	//Initialize sizes
	const int n_triangles = m_in.ibosize / 3;
	const int n_vertices = m_in.vbosize / 3;

	startTiming();
	//Create host vectors
	thrust::host_vector<int4> h_triangles(n_triangles);
	thrust::host_vector<float4> h_vertices(n_vertices);

	std::cout << "Create host vectors Time: " << stopTiming() << std::endl;
	startTiming();

	//Fill in the data
	for (int i = 0; i < n_vertices; i++) {
		h_vertices[i].x = m_in.vbo[i * 3 + 0];
		h_vertices[i].y = m_in.vbo[i * 3 + 1];
		h_vertices[i].z = m_in.vbo[i * 3 + 2];
	}
	for (int i = 0; i < n_triangles; i++) {
		h_triangles[i].x = m_in.ibo[i * 3 + 0];
		h_triangles[i].y = m_in.ibo[i * 3 + 1];
		h_triangles[i].z = m_in.ibo[i * 3 + 2];
	}

	std::cout << "Fill in the data Time: " << stopTiming() << std::endl;
	startTiming();

	//Copy to device vectors
	thrust::device_vector<int4> d_triangles(h_triangles);
	thrust::device_vector<float4> d_vertices(h_vertices);

	std::cout << "Copy to device vectors Time: " << stopTiming() << std::endl;
	startTiming();

	if (first_time) {
		//Create the voxelpipe context
		context = new voxelpipe::FRContext<log_N, log_T>();

		//Reserve data for voxelpipe
		context->reserve(n_triangles, 1024u * 1024u * 16u);

		first_time = false;
	}

	std::cout << "Context Time: " << stopTiming() << std::endl;
	startTiming();

	//Initialize the result data on the device
	thrust::device_vector<float>  d_fb(M*M*M * T*T*T);

	std::cout << "Initialize the result data on the device Time: " << stopTiming() << std::endl;

	// texture
	//{
	startTiming();
	//Copy the texture to the device
	glm::vec3 *device_tex = NULL;
	cudaMalloc((void**)&device_tex, h_tex->width * h_tex->height *sizeof(glm::vec3));
	cudaMemcpy(device_tex, h_tex->data, h_tex->width * h_tex->height *sizeof(glm::vec3), cudaMemcpyHostToDevice);

	std::cout << "Copy texture to device Time: " << stopTiming() << std::endl;
	startTiming();

	//Copy the texture coordinates to the device
	float* device_texcoord = NULL;
	cudaMalloc((void**)&device_texcoord, m_in.tbosize * sizeof(float));
	cudaMemcpy(device_texcoord, m_in.tbo, m_in.tbosize *sizeof(float), cudaMemcpyHostToDevice);

	std::cout << "Copy coord to device Time: " << stopTiming() << std::endl;
	//}

	startTiming();

	//Create the shader to be used that will write texture colors to voxels
	ColorShader my_shader;
	my_shader.texture = device_tex;
	my_shader.tex_height = h_tex->height;
	my_shader.tex_width = h_tex->width;
	my_shader.texcoord = device_texcoord;
	my_shader.texcoord_size = m_in.tbosize;

	//Perform coarse and fine voxelization
	context->coarse_raster(n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), mbbmin, mbbmax);

	context->fine_raster< voxelpipe::Float, voxelpipe::FP32S_FORMAT, voxelpipe::CONSERVATIVE_RASTER, voxelpipe::NO_BLENDING, ColorShader >(
		n_triangles, n_vertices, thrust::raw_pointer_cast(&d_triangles.front()), thrust::raw_pointer_cast(&d_vertices.front()), mbbmin, mbbmax, thrust::raw_pointer_cast(&d_fb.front()), my_shader);

	std::cout << "True voxel Time: " << stopTiming() << std::endl;
	startTiming();

	cudaFree(device_tex);
	cudaFree(device_texcoord);

	std::cout << "cudaFree Time: " << stopTiming() << std::endl;
	startTiming();


	//Get occupied voxels
	int numVoxels = N*N*N;
	int* d_vox;
	cudaMalloc((void**)&d_vox, numVoxels*sizeof(int));

	std::cout << "Get voxel centers cudaMalloc Time: " << stopTiming() << std::endl;
	startTiming();

	getOccupiedVoxels << <(N*N*N / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(thrust::raw_pointer_cast(&d_fb.front()), M, T, d_vox, selectedInternal);

	std::cout << "Get getOccupiedVoxels Time: " << stopTiming() << std::endl;
	startTiming();

	cudaDeviceSynchronize();

	std::cout << "cudaDeviceSynchronize wait Time: " << stopTiming() << std::endl;
	startTiming();

#define FILL
#ifdef FILL

	//Fill the volumn
	//for(int inter = 0; inter<=N; inter++){
	int total = 1, last_total = 1;
	int *total_dev;
	cudaMalloc((void**)&total_dev, sizeof(int));
	cudaMemcpy(total_dev, &total, sizeof(int), cudaMemcpyHostToDevice);

	int loopc = 0;
	while (1){
		loopc++;
		FillVoxels << <(N*N*N / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(d_vox, N, total_dev);
		cudaDeviceSynchronize();

		cudaMemcpy(&total, total_dev, sizeof(int), cudaMemcpyDeviceToHost);
		if (last_total == total)
			break;
		else
			last_total = total;
	}
	cudaFree(total_dev);

	std::cout<< loopc << " loops, " << "FillVoxels Time: " << stopTiming() << std::endl;

	startTiming();

#endif

	//Set index for occupied voxels
	SetOccupiedVoxelsIndex << < (N*N*N / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(d_vox, N);


	std::cout << "SetOccupiedVoxelsIndex Time: " << stopTiming() << std::endl;
	startTiming();


	//Build up the mesh voxels
	BuildMeshVoxels << <(N*N*N / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(mesh_voxels, d_vox, M, T, N);

	std::cout << "BuildMeshVoxels Time: " << stopTiming() << std::endl;
	startTiming();


	
	//Stream Compact voxels to remove the empties
	numVoxels = thrust::copy_if(thrust::device_pointer_cast(d_vox), thrust::device_pointer_cast(d_vox) + numVoxels, 
		thrust::device_pointer_cast(d_voxels), check_voxel())
		- thrust::device_pointer_cast(d_voxels);

	std::cout << "Num Voxels: " << numVoxels << std::endl;
	std::cout << "Stream Compact voxels Time: " << stopTiming() << std::endl;
	startTiming();
	
	//Extract the values at these indices
	extractValues << <(numVoxels / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> >(thrust::raw_pointer_cast(&d_fb.front()), d_voxels, numVoxels, d_values);
	cudaDeviceSynchronize();

	std::cout << "extractValues Time: " << stopTiming() << std::endl;

	
	cudaFree(d_vox);

	return numVoxels;
}

// extractCubesFromVoxelGrid(d_voxels, numVoxels, d_values, m_cube, m_out);
__host__ void extractCubesFromVoxelGrid(int* d_voxels, int numVoxels, int* d_values, Mesh &m_cube, Mesh &m_out) {

	//Move cube data to GPU
	thrust::device_vector<float> d_vbo_cube(m_cube.vbo, m_cube.vbo + m_cube.vbosize);
	thrust::device_vector<int> d_ibo_cube(m_cube.ibo, m_cube.ibo + m_cube.ibosize);
	thrust::device_vector<float> d_nbo_cube(m_cube.nbo, m_cube.nbo + m_cube.nbosize);

	//Create output structs
	float* d_vbo_out;
	int* d_ibo_out;
	float* d_nbo_out;
	float* d_cbo_out;
	cudaMalloc((void**)&d_vbo_out, numVoxels * m_cube.vbosize * sizeof(float));
	cudaMalloc((void**)&d_ibo_out, numVoxels * m_cube.ibosize * sizeof(int));
	cudaMalloc((void**)&d_nbo_out, numVoxels * m_cube.nbosize * sizeof(float));
	cudaMalloc((void**)&d_cbo_out, numVoxels * m_cube.nbosize * sizeof(float));

	//Warn if vbo and nbo are not same size on cube
	if (m_cube.vbosize != m_cube.nbosize) {
		std::cout << "ERROR: cube vbo and nbo have different sizes." << std::endl;
		return;
	}

	//Create resulting cube-ized mesh
	float3 t_d = make_float3((mbbmax.x - mbbmin.x) / float(M),
		(mbbmax.y - mbbmin.y) / float(M),
		(mbbmax.z - mbbmin.z) / float(M));
	float3 p_d = make_float3(t_d.x / float(T),
		t_d.y / float(T), t_d.z / float(T));
	createCubeMesh << <(numVoxels / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> >(d_voxels, d_values, M, T, mbbmin, t_d, p_d, vox_size / CUBE_MESH_SCALE, numVoxels, thrust::raw_pointer_cast(&d_vbo_cube.front()),
		m_cube.vbosize, thrust::raw_pointer_cast(&d_ibo_cube.front()), m_cube.ibosize, thrust::raw_pointer_cast(&d_nbo_cube.front()), d_vbo_out, d_ibo_out, d_nbo_out, d_cbo_out);

	//Store output sizes
	m_out.vbosize = numVoxels * m_cube.vbosize;
	m_out.ibosize = numVoxels * m_cube.ibosize;
	m_out.nbosize = numVoxels * m_cube.nbosize;
	m_out.cbosize = m_out.nbosize;

	//Memory allocation for the outputs
	m_out.vbo = (float*)malloc(m_out.vbosize * sizeof(float));
	m_out.ibo = (int*)malloc(m_out.ibosize * sizeof(int));
	m_out.nbo = (float*)malloc(m_out.nbosize * sizeof(float));
	m_out.cbo = (float*)malloc(m_out.cbosize * sizeof(float));

	//Sync here after doing some CPU work
	cudaDeviceSynchronize();

	//Copy data back from GPU
	//TODO: Can we avoid this step by making everything run from device-side VBO/IBO/NBO/CBO?
	cudaMemcpy(m_out.vbo, d_vbo_out, m_out.vbosize*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_out.ibo, d_ibo_out, m_out.ibosize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_out.nbo, d_nbo_out, m_out.nbosize*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_out.cbo, d_cbo_out, m_out.cbosize*sizeof(float), cudaMemcpyDeviceToHost);

	///Free GPU memory
	cudaFree(d_vbo_out);
	cudaFree(d_ibo_out);
	cudaFree(d_nbo_out);
	cudaFree(d_cbo_out);

}



__host__ void voxelizeToCubes(Mesh &m_in, bmp_texture* tex, Mesh &m_cube, Mesh &m_out, vector<glm::vec3> selectedInternal)
{
	// TODO move positon for following codes
	cudaMemset(mesh_voxels, 0, N*N*N*sizeof(struct Voxel));
	cudaDeviceSynchronize();

	//Voxelize the mesh input
	int numVoxels = N*N*N;
	int* d_voxels;
	int* d_values;
	cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
	cudaMalloc((void**)&d_values, numVoxels*sizeof(int));

	int* select = (int*)malloc((selectedInternal.size() + 1)*sizeof(int));
	int* d_select;
	cudaMalloc((void**)&d_select, (selectedInternal.size() + 1)*sizeof(int));

	int gx, gy, gz;
	for (int i = 0; i < selectedInternal.size(); i++){
		gx = (int)(selectedInternal.at(i).x / grid_size[0]);
		gy = (int)(selectedInternal.at(i).y / grid_size[1]);
		gz = (int)(selectedInternal.at(i).z / grid_size[2]);

		select[i] = (((gx >> log_T) + ((gy >> log_T) << (log_N - log_T)) + ((gz >> log_T) << ((log_N - log_T) << 1))) << ((log_T << 1) + log_T))
			+ (gx&(T - 1)) + ((gy&(T - 1)) << log_T) + ((gz&(T - 1)) << (log_T << 1));

		cout << "selectedInternal set at " << select[i] << endl;
	}
	select[selectedInternal.size()] = -1;
	
	cudaMemcpy(d_select, select, (selectedInternal.size()+1)*sizeof(int), cudaMemcpyHostToDevice);
	numVoxels = voxelizeMesh(m_in, tex, d_voxels, d_values, d_select);
	// Note: in voxelpipe, x varies most fast.

	printLastError();

#if 0
	int* h_voxels = (int*)malloc(numVoxels*sizeof(int));
	int* h_values = (int*)malloc(numVoxels*sizeof(int));
	cudaMemcpy(h_voxels, d_voxels, numVoxels*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_values, d_values, numVoxels*sizeof(int), cudaMemcpyDeviceToHost);
	int a, r, g, b;
	for (int i = 0; i < numVoxels; i++){
		a = (h_values[i] >> 24) & 0xFF;
		r = (h_values[i] >> 16) & 0xFF;
		g = (h_values[i] >> 8) & 0xFF;
		b = (h_values[i]) & 0xFF;
		std::cout << i << " " << h_voxels[i] << " " << a<<" "<<r<<" "<<g<<" "<<b<< std::endl; 

		if (i % 20 == 0) system("pause");
	}
#endif


	//Extract Cubes from the Voxel Grid
	startTiming();
	extractCubesFromVoxelGrid(d_voxels, numVoxels, d_values, m_cube, m_out);
	std::cout << "Extraction Time: " << stopTiming() << std::endl;
	std::cout << "mout sizes: " << m_out.vbosize << " " << m_out.ibosize << " " << m_out.nbosize << " " << m_out.cbosize << std::endl;

	printLastError();

	cudaFree(d_voxels);
	cudaFree(d_values);
	free(select);
	cudaFree(d_select);
}





/*
 *
 * Implementations of photon related functions.
 * Use header file to both share the global variables
 * and in the meanwhile not make this file too long.
 *
 */
#include "photon_inline.h"
