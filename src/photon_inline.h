/*
*
* This file is included at the end of "voxelization.cpp" as an implementation file.
* DO NOT include this file again.
*
*/

#define PHOTON_NUM_LOWER_LIMIT 10000
#define PHOTON_NUM_UPPER_LIMIT 200000

__host__ void OnExiting_Voxel(){
	cudaFree(mipmap);
	cudaFree(photon_list);
	cudaFree(mesh_voxels);
	cudaFree(view_map);
	cudaFree(view_map_color);
	cudaFree(environment_map);
	cudaFree(environment_map_color);
	if (envmap_texture) free(envmap_texture);
	if (view_map_texture) free(view_map_texture);
	if (view_map_color_texture) free(view_map_color_texture);
}

__host__ void OnStarting_Voxel()
{
	int totalsize = sizeof(struct Node)*((1 << (3 * log_N + 3)) - 1) / 7;
	cudaMalloc((void**)&mipmap, totalsize);
	cudaMalloc((void**)&photon_list, 2 * PHOTON_NUM_UPPER_LIMIT*sizeof(struct Photon)); // here amount includes the interpolated ones.
	cudaMalloc((void**)&mesh_voxels, N*N*N*sizeof(struct Voxel));

	cudaMalloc((void**)&view_map, W_WIDTH*W_HEIGHT * 3 * sizeof(float));
	cudaMalloc((void**)&view_map_color, W_WIDTH*W_HEIGHT * 3 * sizeof(float));
	cudaMalloc((void**)&environment_map, envmap_w * envmap_h * 3 * sizeof(float));
	cudaMalloc((void**)&environment_map_color, envmap_w * envmap_h * 3 * sizeof(float));

}

__host__ void initCudaSideTexture(GLuint background_tex)
{
	// setup envmap texture
	envmap_texture = (GLfloat*)malloc(envmap_w * envmap_h * 3 * sizeof(GLfloat));
	if (envmap_texture == 0){
		cout << "envmap_texture allocate failed" << endl;
	}
	memset(envmap_texture, 0, envmap_w * envmap_h * 3 * sizeof(GLfloat));
	glGenTextures(1, &envmap_texture_id);
	if (envmap_texture_id == 0) {
		cout << "Failed to generate envmap_texture_id" << endl;
	}

	//cudaGraphicsGLRegisterImage(&envmap_shared, envmap_texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	//printLastError();


	// setup envmap color texture
	glBindTexture(GL_TEXTURE_2D, background_tex);

	GLint w, h;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

	GLfloat* tmp = (GLfloat*)malloc(w * h * 3 * sizeof(GLfloat));
	if (tmp == 0){
		cout << "envmap_texture allocate failed" << endl;
	}
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, tmp);


	GLfloat* tmp_device = (GLfloat*)malloc(envmap_w * envmap_h * 3 * sizeof(GLfloat));
	if (tmp_device == 0){
		cout << "envmap_texture allocate failed" << endl;
	}
	memset(tmp_device, 0, envmap_w * envmap_h * 3 * sizeof(GLfloat));

	gluScaleImage(GL_RGB, w, h, GL_FLOAT, tmp,
		envmap_w, envmap_h, GL_FLOAT, tmp_device);

	cudaMemcpy(environment_map_color, tmp_device, envmap_w * envmap_h * 3 * sizeof(float), cudaMemcpyHostToDevice);

	free(tmp);
	free(tmp_device);


	// setup view map texture
	view_map_texture = (GLfloat*)malloc(W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	if (view_map_texture == 0){
		cout << "view_map_texture allocate failed" << endl;
	}
	memset(view_map_texture, 0, W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	glGenTextures(1, &view_map_texture_id);
	if (view_map_texture_id == 0) {
		cout << "Failed to generate view_map_texture_id" << endl;
	}

	// setup view map color texture
	view_map_color_texture = (GLfloat*)malloc(W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	if (view_map_color_texture == 0){
		cout << "view_map_texture allocate failed" << endl;
	}
	memset(view_map_color_texture, 0, W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	glGenTextures(1, &view_map_color_texture_id);
	if (view_map_color_texture_id == 0) {
		cout << "Failed to generate view_map_texture_id" << endl;
	}
}





#define N_COEFFICIENT (4.0)

/*
__host__ inline float CalculateRefractive(float x, float y, float z)
{
#ifdef LINEAR
return fabs(y);
#endif

#define CIRCLE
#ifdef CIRCLE
return (x*x + y*y) / N_COEFFICIENT + 1.0;
#endif
}  */

#ifdef QUADRATIC_N
#define CalculateRefractive(x,y,z) (-((x)*(x) + (y)*(y)) / N_COEFFICIENT + 5.0)
/*
#define CalculateRefractive(x,y,z) \
 ((((x)-4.45665)*((x)-4.45665) + ((y)-3.9382) * ((y)-3.9382) + ((z)-4.57365) * ((z)-4.57365)) <  0.9*0.9) ?	\
	 ((((x)-4.45665)*((x)-4.45665) + ((y)-3.9382) * ((y)-3.9382) + ((z)-4.57365) * ((z)-4.57365)) / N_COEFFICIENT + 1.0) : \
	 ((((x)-5.57505)*((x)-5.57505) + ((y)-2.1220) * ((y)-2.1220) + ((z)-5.33565) * ((z)-5.33565)) / N_COEFFICIENT + 1.0)
//(((x)-5.57505)*((x)-5.57505) + ((y)-2.122) * ((y)-2.122) + ((z)-5.33565) * ((z)-5.33565)) <  0.60*0.60) ?
*/
#endif

#ifdef CONSTANT_N
#define CalculateRefractive(x,y,z) (3.0)
#endif

#define GetRefractive(mesh_voxels, offset, grid_x, grid_y, grid_z, grid_size_w, grid_size_l, grid_size_h) \
	((((mesh_voxels) + (id))->is_occupied > 0) ? CalculateRefractive((grid_size_w)* (grid_x), (grid_size_l)* (grid_y), 0) : (1.0))




__host__ inline int GetLevelOriginOffset(int level){
	if (level == 0)
		return 0;
	else if (level > log_N)
		return -1;
	else
		return ((1 << (3 * log_N + 3)) - (1 << (3 * log_N - 3 * level + 3))) / 7;
}






__global__ void CreateMipmapLevel0(int threads, Voxel* mesh_voxels, Node* origin, int dim, float grid_size_w, float grid_size_l, float grid_size_h)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < threads){
		float n;
		n = GetRefractive(mesh_voxels, id, (id / dim / dim) % dim, (id / dim) % dim, id % dim, grid_size_w, grid_size_l, grid_size_h);
		(origin + id)->min = n;
		(origin + id)->max = (origin + id)->min;
		(origin + id)->level = 0;
	}

}


__global__ void ReadInExtinction(int threads, Node* origin, int dim, float grid_size_w, float grid_size_l, float grid_size_h)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < threads){
		(origin + id)->min = 0.1;
		(origin + id)->max = 0.1;
	}

}

//#define DEBUG_GRADIENT
#define DEBUG_GRADIENT_CONDITION (id==2333)

__global__ void StoreGradientInMipmap(int threads, Voxel* mesh_voxels, Node* origin, int dim, float grid_size_w, float grid_size_l, float grid_size_h)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < threads){

		int x = (id / dim / dim) % dim;
		int y = (id / dim) % dim;
		int z = id % dim;

		int tdx=0, tdy=0, tdz=0;
		int max_gradient = 0;
		float n, tn;
		n = (origin + id)->min;

		for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++)
		for (int dz = -1; dz <= 1; dz++)
		{
			if (dx == 0 && dy == 0 && dz == 0) continue;
			if (x + dx < 0 || x + dx >= dim || y + dy < 0 || y + dy >= dim || z + dz<0 || z + dz >= dim) continue;

			//tn = GetRefractive(mesh_voxels, id + dx*dim*dim + dy*dim + dz, x+dx, y+dy, z+dz, grid_size_w, grid_size_l, grid_size_h);
			tn = (origin + id + dx*dim*dim + dy*dim + dz)->min;
			
#ifdef DEBUG_GRADIENT
			if (tn>1.0) 
				printf("[%d] dxdydz:%d,%d,%d, n: %f\n", id, dx, dy, dz, tn);
#endif

			if (abs(tn - n) > abs(max_gradient)){
				tdx = dx;
				tdy = dy;
				tdz = dz;
				max_gradient = tn - n;
			}
		}

		if (tdx == 0 && tdy == 0 && tdz == 0){
			(mesh_voxels + id)->gradient[0] = 0;
			(mesh_voxels + id)->gradient[1] = 0;
			(mesh_voxels + id)->gradient[2] = 0;
		}
		else{
			float len = sqrt(1.0*tdx*tdx + 1.0*tdy*tdy + 1.0*tdz*tdz);
			(mesh_voxels + id)->gradient[0] = max_gradient * tdx / len;
			(mesh_voxels + id)->gradient[1] = max_gradient * tdy / len;
			(mesh_voxels + id)->gradient[2] = max_gradient * tdz / len;

#ifdef DEBUG_GRADIENT
			printf("[%d] gradient: %f,%f,%f\n", id, (mesh_voxels + id)->gradient[0], (mesh_voxels + id)->gradient[1], (mesh_voxels + id)->gradient[2]);
#endif
		}
		
		
	}

}



__global__ void CreateMipmapOtherLevels(int threads, Node* origin, Node* origin_pre, int dim, float grid_size_w, float grid_size_l, float grid_size_h)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < threads){
		int x = (id / dim / dim) % dim;
		int y = (id / dim) % dim;
		int z = id % dim;
		int base = 8 * x*dim*dim + 4 * y*dim + 2 * z;

		float tmin = 100000;
		float tmax = 0.0;
		Node *tmp;
		for (int dx = 0; dx <= 1; dx++){
			for (int dy = 0; dy <= 1; dy++){
				for (int dz = 0; dz <= 1; dz++){
					tmp = origin_pre + base + 4 * dx*dim*dim + 2 * dy*dim + dz;
					if (tmp->min < tmin) tmin = tmp->min;
					if (tmp->max > tmax) tmax = tmp->max;
				}
			}
		}
		(origin + id)->min = tmin;
		(origin + id)->max = tmax;
		(origin + id)->level = 0;
	}

}

__global__ void LabelLevel(int threads, Node* origin, Node* origin_pre, int dim, int level, float threshold)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < threads){
		if ((origin + id)->level < level) /*actually == 0*/
		{
			if ((origin + id)->max - (origin + id)->min < threshold){
				(origin + id)->level = level;
			}
		} // else keep (origin + id)->level = 0


		int x = (id / dim / dim) % dim;
		int y = (id / dim) % dim;
		int z = id % dim;
		int base = 8 * x*dim*dim + 4 * y*dim + 2 * z;
		float thislevel = (origin + id)->level;

		Node *tmp;
		for (int dx = 0; dx <= 1; dx++){
			for (int dy = 0; dy <= 1; dy++){
				for (int dz = 0; dz <= 1; dz++){
					tmp = origin_pre + base + 4 * dx*dim*dim + 2 * dy*dim + dz;
					tmp->level = thislevel;
				}
			}
		}
	}

}

__host__ int CreateMipmap(float threshold)
{
	int threads = 0;
	int grid_num_this_level = 0;


	// clear the mipmap storage
	int totalsize = sizeof(struct Node)*((1 << (3 * log_N + 3)) - 1) / 7;
	cudaMemset(mipmap, 0, totalsize);


	startTiming();
	// mipmap level 0
	threads = (1 << (3 * (log_N - 0)));
	grid_num_this_level = (1 << (log_N - 0));
	CreateMipmapLevel0 << < (threads / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >
		(threads, mesh_voxels, mipmap + GetLevelOriginOffset(0), grid_num_this_level, grid_size[0], grid_size[1], grid_size[2]);
	std::cout << "Create mipmap level 0 time: " << stopTiming() << std::endl;


	startTiming();
	// create other mipmap levels
	for (int level = 1; level <= log_N; level++){
		threads = (1 << (3 * (log_N - level)));
		grid_num_this_level = (1 << (log_N - level));

		CreateMipmapOtherLevels << < (threads / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >
			(threads, mipmap + GetLevelOriginOffset(level), mipmap + GetLevelOriginOffset(level - 1), grid_num_this_level,
			grid_size[0], grid_size[1], grid_size[2]);

		cudaDeviceSynchronize();
	}
	std::cout << "Create mipmap other levels time: " << stopTiming() << std::endl;



	startTiming();
	// label levels
	for (int level = log_N; level > 0; level--){
		threads = (1 << (3 * (log_N - level)));
		grid_num_this_level = (1 << (log_N - level));

		LabelLevel << < (threads / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >
			(threads, mipmap + GetLevelOriginOffset(level), mipmap + GetLevelOriginOffset(level - 1), grid_num_this_level, level, threshold);

		cudaDeviceSynchronize();
	}
	std::cout << "Label mipmap levels time: " << stopTiming() << std::endl;



#ifdef CONSTANT_N
	startTiming();
	// store refractive gradient in mesh_voxels
	threads = (1 << (3 * (log_N - 0)));
	grid_num_this_level = (1 << (log_N - 0));
	StoreGradientInMipmap << < (threads / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >
		(threads, mesh_voxels, mipmap + GetLevelOriginOffset(0), grid_num_this_level, grid_size[0], grid_size[1], grid_size[2]);
	std::cout << "StoreGradientInMipmap time: " << stopTiming() << std::endl;
#ifdef DEBUG_GRADIENT
	system("pause");
#endif
#endif

	/*

	startTiming();
	// read in extinction (absorption + scatterring)
	// @see <Interactive Relighting of Dynamic Refractive Objects>[ACM Trans. Graph. 27, 3, Article 35, 2008, page 2]
	threads = (1 << (3 * (log_N - 0)));
	grid_num_this_level = (1 << (log_N - 0));
	ReadInExtinction << < (threads / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >
		(threads, mipmap + GetLevelOriginOffset(0), grid_num_this_level, grid_size[0], grid_size[1], grid_size[2]);
	std::cout << "Read in extinction time: " << stopTiming() << std::endl;

	*/




	/*
	Node *tmp , *ttmp;
	tmp = (Node*)malloc(totalsize);
	cudaMemcpy(tmp, mipmap, (totalsize), cudaMemcpyDeviceToHost);



	for (int level = 0; level <= log_N; level++){
	ttmp = tmp + GetLevelOriginOffset(level);
	int num = 1 << (log_N - level);
	num = num*num*num;
	for (int i = 0; i < (num < 100 ? num : 100); i++){
	cout << "at level " << level << ": " << (ttmp + i)->min << " " << (ttmp + i)->max << " " << (ttmp + i)->level << endl;
	}
	system("PAUSE");

	}
	*/

	return 0;
}





#define FLT_THRESHOLD 0.0001

__device__ int MapIntersectionToEnvMap(float3 coord, float world_size, float3 world_offset)
{
	int u0, v0;
	float u, v;

	coord.x = coord.x - world_offset.x;
	coord.y = coord.y - world_offset.y;
	coord.z = coord.z - world_offset.z;

	// left
	if (fabs(coord.x - (-world_size)) < FLT_THRESHOLD){
		u = 1 - (coord.z + world_size) / world_size / 2;
		v = (coord.y + world_size) / world_size / 2;
		u0 = 0; v0 = 1;
	}
	// right
	else if (fabs(coord.x - world_size) < FLT_THRESHOLD){
		u = (coord.z + world_size) / world_size / 2;
		v = (coord.y + world_size) / world_size / 2;
		u0 = 2; v0 = 1;
	}
	// back
	else if (fabs(coord.z - (-world_size)) < FLT_THRESHOLD){
		u = (coord.x + world_size) / world_size / 2;
		v = (coord.y + world_size) / world_size / 2;
		u0 = 1; v0 = 1;
	}
	// front
	else if (fabs(coord.z - world_size) < FLT_THRESHOLD){
		u = 1 - (coord.x + world_size) / world_size / 2;
		v = (coord.y + world_size) / world_size / 2;
		u0 = 3; v0 = 1;
	}
	// top
	else if (fabs(coord.y - world_size) < FLT_THRESHOLD){
		u = (coord.x + world_size) / world_size / 2;
		v = (coord.z + world_size) / world_size / 2;
		u0 = 1; v0 = 2;
	}
	// bottom
	else if (fabs(coord.y - (-world_size)) < FLT_THRESHOLD){
		u = (coord.x + world_size) / world_size / 2;
		v = 1 - (coord.z + world_size) / world_size / 2;
		u0 = 1; v0 = 0;
	}

	return (u0*envmap_grid_size + (int)(u*envmap_grid_size)) * 3
		+ (v0*envmap_grid_size + (int)(v*envmap_grid_size))*envmap_w * 3;
}









#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)>(b)) ? (a) : (b))

#define Clamp(value, low, high) (((value)<(low)) ? (low) : (((value)>(high)) ? (high) : (value)))

#define GetVecLen(x,y,z) (sqrt((x)*(x) + (y)*(y) + (z)*(z)))

#define GetVoxelOffsetFromPointCoord(x, y, z, grid_size_w, grid_size_l, grid_size_h) \
	(((int)((x) / (grid_size_w)))*N*N + ((int)((y) / (grid_size_l)))*N + (int)((z) / (grid_size_h)))

#define Sign(v) ((v)>0 ? 1 : ((v)<0 ? -1 : 0))


#ifdef QUADRATIC_N

#define GetGradient(gradient, x, y, z) \
{\
	(gradient)[0] = -2 * (x) / N_COEFFICIENT; \
	(gradient)[1] = -2 * (y) / N_COEFFICIENT; \
	(gradient)[2] = 0; \
}

/*
#define GetGradient(gradient, x, y, z) \
if ((((x)-4.45665)*((x)-4.45665) + ((y)-3.9382) * ((y)-3.9382) + ((z)-4.57365) * ((z)-4.57365)) < 0.9*0.9) \
{\
	(gradient)[0] = 2 * ((x)-4.45665) / N_COEFFICIENT; \
	(gradient)[1] = 2 * ((x)-3.9382) / N_COEFFICIENT; \
	(gradient)[2] = 2 * ((x)-4.57365); \
}\
else\
{\
	(gradient)[0] = 2 * ((x)-5.57505) / N_COEFFICIENT; \
	(gradient)[1] = 2 * ((x)-2.1220) / N_COEFFICIENT; \
	(gradient)[2] = 2 * ((x)-5.33565); \
}
*/
//((((x)-4.45665)*((x)-4.45665) + ((y)-3.9382) * ((y)-3.9382) + ((z)-4.57365) * ((z)-4.57365)) / N_COEFFICIENT + 1.0) : \
//((((x)-5.57505)*((x)-5.57505) + ((y)-2.1220) * ((y)-2.1220) + ((z)-5.33565) * ((z)-5.33565)) / N_COEFFICIENT + 1.0)
#endif

#ifdef CONSTANT_N
#define GetGradient(gradient, x, y, z) (0)
#endif

#define GetVoxelCoordFromPointCoord(coord_in, grid_size) ((int)((coord_in) / (grid_size)))



/*
__device__ inline int3 GetVoxelCoordFromPointCoord(float x, float y, float z, float grid_size_w, float grid_size_l, float grid_size_h)
{
// TODO ensure in the volumn
int3 coord;
coord.x = (int)(x / grid_size_w);
coord.y = (int)(y / grid_size_l);
coord.z = (int)(z / grid_size_h);

return coord;
}
*/



#define DEBUG_PATH_CONDITION (0)
#define TIMING_CONDITION (0)


__global__ void MarchPhoton(bool is_light_photon, Voxel* mesh_voxels, Node* mipmap, Photon* photon_list, int minimum_march_grid_count, int maximum_march_grid_count,
int photon_num, int photon_num_lower_limit, int *photon_num_trace_dev,
float grid_size_w, float grid_size_l, float grid_size_h, float3 meshBoundingBoxMin, float3 meshBoundingBoxMax, float world_size,
float *environment_map, float* environment_map_color, float* view_map, float* view_map_color
#ifdef DEBUG_PATH
, float3 *debug_path
#endif
)
{
	// TODO will make photon_num_trace_dev shared improve performance?


	int id = blockIdx.x*blockDim.x + threadIdx.x;


	long long int start_time, stop_time;

	if (id < photon_num){

		start_time = clock64();

		float photonpos[3];
		photonpos[0] = (photon_list + id)->position[0];
		photonpos[1] = (photon_list + id)->position[1];
		photonpos[2] = (photon_list + id)->position[2];

		float direction[3];
		direction[0] = (photon_list + id)->direction[0];
		direction[1] = (photon_list + id)->direction[1];
		direction[2] = (photon_list + id)->direction[2];

		float radiance[3];
		radiance[0] = (photon_list + id)->residual_radiance[0];
		radiance[1] = (photon_list + id)->residual_radiance[1];
		radiance[2] = (photon_list + id)->residual_radiance[2];

		float photon_color[3] = { 1.0, 1.0, 1.0 };

		float* v = direction;
		int screen_x, screen_y;
		if (!is_light_photon){
			screen_x = (int)radiance[0];
			screen_y = (int)radiance[1];
			radiance[0] = 0.;
			radiance[1] = 0.;
			radiance[2] = 0.;
		}
		float grid_size[3] = { grid_size_w, grid_size_l, grid_size_h };
		bool is_last_intersection = false;

		Voxel *vv;
		Node* node;

		float global_origin[3] = { meshBoundingBoxMin.x, meshBoundingBoxMin.y, meshBoundingBoxMin.z };
		int region_grid_count;
		float region_size[3];
		float region_origin[3];

		float photonpos_next[3];
		int valid_count;

		float n;
		float len;
		float ds;

		float gradient[3] = { 0, 0, 0 };

		float3 world_offset = make_float3(0, meshBoundingBoxMin.y - (-world_size), 0);



		if (TIMING_CONDITION){
			stop_time = clock64();
			printf("initilization: %ld\n", stop_time - start_time);
			start_time = stop_time;
		}

#ifdef DEBUG_PATH
		(photon_list + id)->iteration_count = 0;
		if (DEBUG_PATH_CONDITION)
			printf("[%d]:\n pos: %f,%f,%f, dire:%f,%f,%f\n", id,
			photonpos[0], photonpos[1], photonpos[2], direction[0], direction[1], direction[2]);
#endif

		// march loop
		while (*photon_num_trace_dev > photon_num_lower_limit)
		{
			node = mipmap + GetVoxelOffsetFromPointCoord(
				Clamp(photonpos[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x) - meshBoundingBoxMin.x,
				Clamp(photonpos[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y) - meshBoundingBoxMin.y,
				Clamp(photonpos[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z) - meshBoundingBoxMin.z,
				grid_size_w, grid_size_l, grid_size_h);

			region_grid_count = node->level + 1;
			region_size[0] = grid_size[0] * region_grid_count;
			region_size[1] = grid_size[1] * region_grid_count;
			region_size[2] = grid_size[2] * region_grid_count;

			region_origin[0] = ((int)((photonpos[0] - global_origin[0]) / region_size[0])) * region_size[0] * 1.0f + global_origin[0];
			region_origin[1] = ((int)((photonpos[1] - global_origin[1]) / region_size[1])) * region_size[1] * 1.0f + global_origin[1];
			region_origin[2] = ((int)((photonpos[2] - global_origin[2]) / region_size[2])) * region_size[2] * 1.0f + global_origin[2];

			// >>>>>>>>>>>>>>>>>>>>>>>>>>>
		last_intersection_in:
			photonpos_next[0] = -1;
			photonpos_next[1] = -1;
			photonpos_next[2] = -1;
			valid_count = 0;

			if (DEBUG_PATH_CONDITION)
				printf("\nregion_grid_count:%d \ngrid_size:%f %f %f \nregion_size:%f\n", region_grid_count,
				grid_size[0], grid_size[1], grid_size[2],
				region_size[0], region_size[1], region_size[2]);
			if (DEBUG_PATH_CONDITION)
				printf("region_origin: %f,%f,%f\n", region_origin[0], region_origin[1], region_origin[2]);

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("while initilization: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

			// get the intersection point with the region faces
			for (int known_i = 0; known_i < 3; known_i++) {
				if (Sign(direction[known_i]) == 0) {
					// direction is parallel to this axis, so no intersection with this face
					continue;
				}
				else {
					photonpos_next[known_i] = region_origin[known_i] + (Sign(direction[known_i]) + 1) * region_grid_count * grid_size[known_i] / 2;
					valid_count = 1;

					if (DEBUG_PATH_CONDITION)
						printf("\nknown_i %d:%f\n", known_i, photonpos_next[known_i]);

					float tmp;
					for (int nki = 1; nki <= 2; nki++) {
						int not_known_i = (known_i + nki) % 3;
						tmp = (photonpos_next[known_i] - photonpos[known_i]) / direction[known_i] * direction[not_known_i] + photonpos[not_known_i];

						if (DEBUG_PATH_CONDITION)
							printf("not_known_i %d: %f\n", not_known_i, tmp);

						if (tmp < region_origin[not_known_i]
							|| tmp  > region_grid_count * grid_size[not_known_i] + region_origin[not_known_i]) {
							break;
						}
						else {
							photonpos_next[not_known_i] = tmp;
							valid_count++;
						}
					}

					// we get 3 valid points coord, stop
					if (valid_count == 3) {
						break;
					}
				}
			}
			if (is_last_intersection){
				goto last_intersection_out;
			}


			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("intersection: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

			// now we get the ds, note it is march length multiplied by 1.001
			n = (mesh_voxels + (int)(node - mipmap))->is_occupied > 0 ? CalculateRefractive(photonpos[0], photonpos[1], photonpos[2]) : 1.0;

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("get n: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

			len = GetVecLen(photonpos_next[0] - photonpos[0], photonpos_next[1] - photonpos[1], photonpos_next[2] - photonpos[2]);
			ds = 1.1 * MAX(len, MIN(grid_size_w, MIN(grid_size_l, grid_size_h)));

			if (DEBUG_PATH_CONDITION)
				printf("\nn:%f, ds:%f ", n, ds);

			ds = MIN(ds, maximum_march_grid_count*MIN(grid_size_w, MIN(grid_size_l, grid_size_h)));

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("get ds: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

			if (DEBUG_PATH_CONDITION)
				printf("adjusted ds: %f\n", ds);


#ifdef QUADRATIC_N
			if ((mesh_voxels + (int)(node - mipmap))->is_occupied > 0) {
				GetGradient(gradient, photonpos[0], photonpos[1], photonpos[2]);
			}
			else
			{
				gradient[0] = 0;
				gradient[1] = 0;
				gradient[2] = 0;
			}
#endif
#ifdef CONSTANT_N
			gradient[0] = (mesh_voxels + (int)(node - mipmap))->gradient[0];
			gradient[1] = (mesh_voxels + (int)(node - mipmap))->gradient[1];
			gradient[2] = (mesh_voxels + (int)(node - mipmap))->gradient[2];
#endif

			if (DEBUG_PATH_CONDITION)
				printf("gradient: %f,%f,%f\n", gradient[0], gradient[1], gradient[2]);

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("get gradient: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

#ifdef DEBUG_PATH
			if ((photon_list + id)->iteration_count < 100){
				(debug_path + 100 * id + (int)(photon_list + id)->iteration_count)->x = photonpos[0];
				(debug_path + 100 * id + (int)(photon_list + id)->iteration_count)->y = photonpos[1];
				(debug_path + 100 * id + (int)(photon_list + id)->iteration_count)->z = photonpos[2];
			}
			(photon_list + id)->iteration_count = (photon_list + id)->iteration_count + 1;
#endif


			// march
			// @see <Interactive Relighting of Dynamic Refractive Objects>[ACM Trans. Graph. 27, 3, Article 35, 2008, page 3,5]
			// update position
			photonpos_next[0] = photonpos[0] + ds / n*v[0];
			photonpos_next[1] = photonpos[1] + ds / n*v[1];
			photonpos_next[2] = photonpos[2] + ds / n*v[2];

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("march: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}

			// interpolate between passed voxels
			if (is_light_photon)
			{
				int3 start;
				start.x = GetVoxelCoordFromPointCoord(Clamp(photonpos[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x) - meshBoundingBoxMin.x, grid_size_w);
				start.y = GetVoxelCoordFromPointCoord(Clamp(photonpos[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y) - meshBoundingBoxMin.y, grid_size_l);
				start.z = GetVoxelCoordFromPointCoord(Clamp(photonpos[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z) - meshBoundingBoxMin.z, grid_size_h);

				int3 end;
				end.x = GetVoxelCoordFromPointCoord(Clamp(photonpos_next[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x) - meshBoundingBoxMin.x, grid_size_w);
				end.y = GetVoxelCoordFromPointCoord(Clamp(photonpos_next[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y) - meshBoundingBoxMin.y, grid_size_l);
				end.z = GetVoxelCoordFromPointCoord(Clamp(photonpos_next[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z) - meshBoundingBoxMin.z, grid_size_h);

				/*
				GetVoxelCoordFromPointCoord(start,
				Clamp(photonpos[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x) - meshBoundingBoxMin.x,
				Clamp(photonpos[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y) - meshBoundingBoxMin.y,
				Clamp(photonpos[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z) - meshBoundingBoxMin.z,
				grid_size_w, grid_size_l, grid_size_h);
				*/

				/*
				GetVoxelCoordFromPointCoord(end,
				Clamp(photonpos_next[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x) - meshBoundingBoxMin.x,
				Clamp(photonpos_next[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y) - meshBoundingBoxMin.y,
				Clamp(photonpos_next[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z) - meshBoundingBoxMin.z,
				grid_size_w, grid_size_l, grid_size_h);
				*/


				int dx = abs(end.x - start.x);
				int dy = abs(end.y - start.y);
				int dz = abs(end.z - start.z);

				int3 direction;
				direction.x = end.x - start.x;
				direction.y = end.y - start.y;
				direction.z = end.z - start.z;


				float v_len = GetVecLen(v[0], v[1], v[2]);
				float unit_v[3] = { v[0] / v_len, v[1] / v_len, v[2] / v_len };
				float radiance_len = GetVecLen(radiance[0], radiance[1], radiance[2]);


				if (dx >= dy && dx >= dz){
					for (int i = 0; start.x + i<end.x; i++){
						// NOTE: here the meanings of dx,dy,dz change. it is absolute now.
						dx = start.x + i;
						dy = start.y + i*direction.y / direction.x;
						dz = start.z + i*direction.z / direction.x;
						vv = mesh_voxels + dx*N*N + dy*N + dz;
						//if (vv->is_occupied > 0)
						{
							atomicAdd(&(vv->radiance[0]), radiance[0]);
							atomicAdd(&(vv->radiance[1]), radiance[1]);
							atomicAdd(&(vv->radiance[2]), radiance[2]);
							atomicAdd(&(vv->direction[0]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[1]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[2]), radiance_len * unit_v[0]);

							if (DEBUG_PATH_CONDITION)
								printf("store voxel %d radiance %f,%f,%f, occupied: %d\n", (int)(node - mipmap), radiance[0], radiance[1], radiance[2], vv->is_occupied);
						}
						if (!(vv->is_occupied > 0)){
							vv->color[0] = photon_color[0];
							vv->color[1] = photon_color[1];
							vv->color[2] = photon_color[2];
						}
					}
				}
				else if (dy >= dx && dy >= dz){
					for (int i = 0; start.y + i<end.y; i++){
						dx = start.x + i*direction.x / direction.y;
						dy = start.y + i;
						dz = start.z + i*direction.z / direction.y;
						vv = mesh_voxels + dx*N*N + dy*N + dz;
						//if (vv->is_occupied > 0)
						{
							atomicAdd(&(vv->radiance[0]), radiance[0]);
							atomicAdd(&(vv->radiance[1]), radiance[1]);
							atomicAdd(&(vv->radiance[2]), radiance[2]);
							atomicAdd(&(vv->direction[0]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[1]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[2]), radiance_len * unit_v[0]);

							if (DEBUG_PATH_CONDITION)
								printf("store voxel %d radiance %f,%f,%f, occupied: %d\n", (int)(node - mipmap), radiance[0], radiance[1], radiance[2], vv->is_occupied);
						}
						if (!(vv->is_occupied > 0)){
							vv->color[0] = photon_color[0];
							vv->color[1] = photon_color[1];
							vv->color[2] = photon_color[2];
						}
					}
				}
				else {
					for (int i = 0; start.z + i<end.z; i++){
						dx = start.x + i*direction.x / direction.z;
						dy = start.y + i*direction.y / direction.z;
						dz = start.z + i;
						vv = mesh_voxels + dx*N*N + dy*N + dz;
						//if (vv->is_occupied > 0)
						{
							atomicAdd(&(vv->radiance[0]), radiance[0]);
							atomicAdd(&(vv->radiance[1]), radiance[1]);
							atomicAdd(&(vv->radiance[2]), radiance[2]);
							atomicAdd(&(vv->direction[0]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[1]), radiance_len * unit_v[0]);
							atomicAdd(&(vv->direction[2]), radiance_len * unit_v[0]);

							if (DEBUG_PATH_CONDITION)
								printf("store voxel %d radiance %f,%f,%f, occupied: %d\n", (int)(node - mipmap), radiance[0], radiance[1], radiance[2], vv->is_occupied);
						}
						if (!(vv->is_occupied > 0)){
							vv->color[0] = photon_color[0];
							vv->color[1] = photon_color[1];
							vv->color[2] = photon_color[2];
						}
					}
				}
			}
			else {
				// calculate color and radiance. @see Mie Scattering Theory
				vv = mesh_voxels + (int)(node - mipmap);
				// cos^2 = (v0.v1)^2 / (v0^2 * v1^2)
				float dot = vv->direction[0] * direction[0] + vv->direction[1] * direction[1] + vv->direction[2] * direction[2];
				float len1 = vv->direction[0] * vv->direction[0] + vv->direction[1] * vv->direction[1] + vv->direction[2] * vv->direction[2];
				float len2 = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
				float cos2;
				if (len1 < FLT_THRESHOLD || len2 < FLT_THRESHOLD) cos2 = 0.;
				else cos2 = dot*dot / (len1*len2);

				radiance[0] += (vv->color[0] * vv->radiance[0] * vv->scattering_coefficient * (1 + cos2));
				radiance[1] += (vv->color[1] * vv->radiance[1] * vv->scattering_coefficient * (1 + cos2));
				radiance[2] += (vv->color[2] * vv->radiance[2] * vv->scattering_coefficient * (1 + cos2));

				if (DEBUG_PATH_CONDITION)
					printf("added voxel %d radiance %f,%f,%f with color %f,%f,%f in view ray, occupied: %d\n", (int)(node - mipmap), vv->radiance[0],
					vv->radiance[1], vv->radiance[2], vv->color[0], vv->color[1], vv->color[2], vv->is_occupied);
			}

			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("rasterization: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}


			if (DEBUG_PATH_CONDITION)
				printf("[%d]: valid_count: %d, from pos: %f,%f,%f ", id, valid_count, photonpos[0], photonpos[1], photonpos[2]);
			if (DEBUG_PATH_CONDITION)
				printf("to pos: %f,%f,%f \n", photonpos_next[0], photonpos_next[1], photonpos_next[2]);
#ifdef DEBUG_PATH
			if (DEBUG_PATH_CONDITION)
				printf("iteration: %f\n", (photon_list + id)->iteration_count);
#endif


			// update position
			photonpos[0] = photonpos_next[0];
			photonpos[1] = photonpos_next[1];
			photonpos[2] = photonpos_next[2];

			// update direction
			v[0] = v[0] + ds*gradient[0];
			v[1] = v[1] + ds*gradient[1];
			v[2] = v[2] + ds*gradient[2];

			// update radiance
			if (is_light_photon){
				vv = mesh_voxels + (int)(node - mipmap);
				float atten = exp(-(vv->absorption_coefficient + vv->scattering_coefficient)*ds); //extinction coeffient. @see definition of struct Node
				if (DEBUG_PATH_CONDITION)
					printf("[%d] attenuation: %f\n", id, atten);
				radiance[0] = radiance[0] * atten;
				radiance[1] = radiance[1] * atten;
				radiance[2] = radiance[2] * atten;
			}

			// update color
			if (is_light_photon){
				photon_color[0] = MIN(photon_color[0], vv->color[0]);
				photon_color[1] = MIN(photon_color[1], vv->color[1]);
				photon_color[2] = MIN(photon_color[2], vv->color[2]);
			}

			


			if (TIMING_CONDITION){
				stop_time = clock64();
				printf("update: %ld\n", stop_time - start_time);
				start_time = stop_time;
			}


			// out of mesh
			if (photonpos[0] < meshBoundingBoxMin.x || photonpos[0] > meshBoundingBoxMax.x
				|| photonpos[1] < meshBoundingBoxMin.y || photonpos[1] > meshBoundingBoxMax.y
				|| photonpos[2] < meshBoundingBoxMin.z || photonpos[2] > meshBoundingBoxMax.z)
			{
				// get intersection with environment map
				if (!is_last_intersection){
					grid_size[0] = world_size * 2;
					grid_size[1] = world_size * 2;
					grid_size[2] = world_size * 2;
					region_grid_count = 1;
					region_size[0] = world_size * 2;
					region_size[1] = world_size * 2;
					region_size[2] = world_size * 2;
					region_origin[0] = -world_size + world_offset.x;
					region_origin[1] = -world_size + world_offset.y;
					region_origin[2] = -world_size + world_offset.z;

					photonpos[0] = Clamp(photonpos[0], meshBoundingBoxMin.x, meshBoundingBoxMax.x);
					photonpos[1] = Clamp(photonpos[1], meshBoundingBoxMin.y, meshBoundingBoxMax.y);
					photonpos[2] = Clamp(photonpos[2], meshBoundingBoxMin.z, meshBoundingBoxMax.z);

					is_last_intersection = true;
					goto last_intersection_in;
				}
				// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			last_intersection_out:
				// intersection is now in photonpos_next
				if (is_light_photon){
					float* px = environment_map + MapIntersectionToEnvMap(make_float3(photonpos_next[0], photonpos_next[1], photonpos_next[2]), world_size, world_offset);
					// we do not alter the env_map original color, so combine the color and radiance here.
					atomicAdd(px + 0, photon_color[0] * radiance[0]);
					atomicAdd(px + 1, photon_color[1] * radiance[1]);
					atomicAdd(px + 2, photon_color[2] * radiance[2]);
				}
				else {
					float *vm, *vm_color, *em, *em_color;
					vm = view_map + screen_y*W_WIDTH * 3 + screen_x * 3;
					vm_color = view_map_color + screen_y*W_WIDTH * 3 + screen_x * 3;
					int offset = MapIntersectionToEnvMap(make_float3(photonpos_next[0], photonpos_next[1], photonpos_next[2]), world_size, world_offset);
					em = environment_map + offset;
					em_color = environment_map_color + offset;

					// TODO weighted?
					vm[0] = radiance[0];
					vm[1] = radiance[1];
					vm[2] = radiance[2];

					vm_color[0] = em_color[0] + em[0] * 0.1;
					vm_color[1] = em_color[1] + em[0] * 0.1;
					vm_color[2] = em_color[2] + em[0] * 0.1;

					if (DEBUG_PATH_CONDITION)
						printf("[%d] background radiance:%f,%f,%f, accumulated radiance: %f,%f,%f\n", id, em[0], em[1], em[2], vm[0], vm[1], vm[2]);

				}

				atomicAdd(photon_num_trace_dev, -1);

				if (DEBUG_PATH_CONDITION)
					printf("[%d] out of mesh at: %f,%f,%f, global origin:%f,%f,%f, radiance: %f,%f,%f\n", id, photonpos[0], photonpos[1], photonpos[2],
					global_origin[0], global_origin[1], global_origin[2], radiance[0], radiance[1], radiance[2]);


				return;
			}
			// radiance low 
			if (is_light_photon && (GetVecLen(radiance[0], radiance[1], radiance[2]) < 1.0 / 1000.0)){
#ifdef DEBUG_PATH
				//if (DEBUG_PATH_CONDITION)
				printf("[%d] radiance low, stop. inter: %f, radiance: %f,%f,%f\n", id, (photon_list + id)->iteration_count, radiance[0], radiance[1], radiance[2]);
#endif

				atomicAdd(photon_num_trace_dev, -1);
				return;
			}
		}
	}
}


__global__ void SmoothMapLinear(float* map, int width, int height, int step_len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < width*height){
		float* base = map + id * 3;

		float r = base[0];
		float g = base[1];
		float b = base[2];
		if (base[0] > 0 && base[1] > 0 && base[2] > 0){
			float dr = base[step_len * 3 + 0] - r;
			float dg = base[step_len * 3 + 1] - g;
			float db = base[step_len * 3 + 2] - b;

			for (int i = 1; i < step_len; i++){
				base[i * 3 + 0] = r + dr*i / step_len;
				base[i * 3 + 1] = g + dg*i / step_len;
				base[i * 3 + 2] = b + db*i / step_len;
			}
		}
	}
}


__global__ void SmoothMapGaussian(float* map, int width, int height, int step_len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

#define KERNEL_SIZE 3
#define KERNEL_SIZE_HALF (KERNEL_SIZE/2)

	if (id < width*height){
		float gaussian_kernel[] = {
			0.1019, 0.1154, 0.1019,
			0.1154, 0.1308, 0.1154,
			0.1019, 0.1154, 0.1019 };

		/*
		float gaussian_kernel[] = {
		0.0232,    0.0338,    0.0383,    0.0338,    0.0232,
		0.0338,    0.0492,    0.0558,    0.0492,    0.0338,
		0.0383,    0.0558,    0.0632,    0.0558,    0.0383,
		0.0338,    0.0492,    0.0558,    0.0492,    0.0338,
		0.0232,    0.0338,    0.0383,    0.0338,    0.0232 }; */

		float* base = map + id * 3;

		int x = id % width;
		int y = id / width;
		float sum_r = 0;
		float sum_g = 0;
		float sum_b = 0;
		for (int i = -KERNEL_SIZE_HALF; i <= KERNEL_SIZE_HALF; i++){
			for (int j = -2; j <= 2; j++){
				if (x + i < 0 || x + i >= width || y + j < 0 || y + j >= height) continue;

				sum_r += map[(x + i) * 3 + (y + j)*width * 3 + 0] * gaussian_kernel[KERNEL_SIZE_HALF + i + (KERNEL_SIZE_HALF + j) * KERNEL_SIZE];
				sum_g += map[(x + i) * 3 + (y + j)*width * 3 + 1] * gaussian_kernel[KERNEL_SIZE_HALF + i + (KERNEL_SIZE_HALF + j) * KERNEL_SIZE];
				sum_b += map[(x + i) * 3 + (y + j)*width * 3 + 2] * gaussian_kernel[KERNEL_SIZE_HALF + i + (KERNEL_SIZE_HALF + j) * KERNEL_SIZE];
			}
		}

		base[0] = sum_r;
		base[1] = sum_g;
		base[2] = sum_b;

	}
}


__global__ void ResetMeshVoxels(int total, Voxel * mesh_voxels)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < total){
		Voxel * vv = mesh_voxels + id;
		vv->direction[0] = 0;
		vv->direction[1] = 0;
		vv->direction[2] = 0;
		vv->radiance[0] = 0;
		vv->radiance[1] = 0;
		vv->radiance[2] = 0;
	}
}

__host__ int MarchPhotons(bool is_light_photon, int photon_num, int step_len, int minimum_march_grid_count, int maximum_march_grid_count
#ifdef DEBUG_PATH
	, vector<glm::vec3> *paths
#endif
	)
{
	int * photon_num_trace_dev;
	cudaMalloc((void**)&photon_num_trace_dev, sizeof(int));
	cudaMemcpy(photon_num_trace_dev, &photon_num, sizeof(int), cudaMemcpyHostToDevice);
	startTiming();

#ifdef DEBUG_PATH
	float3 *debug_path;
	cudaMalloc((void**)&debug_path, photon_num * 100 * sizeof(float3));
#endif

	// clear the maps
	if (is_light_photon){
		cudaMemset(environment_map, 0, envmap_w * envmap_h * 3 * sizeof(float));
		ResetMeshVoxels << <(N*N*N / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(N*N*N, mesh_voxels);
	}
	else {
		cudaMemset(view_map, 0, W_WIDTH*W_HEIGHT * 3 * sizeof(float));
		cudaMemset(view_map_color, 0, W_WIDTH*W_HEIGHT * 3 * sizeof(float));
	}


	MarchPhoton << <(photon_num / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(
		is_light_photon, mesh_voxels, mipmap, photon_list, minimum_march_grid_count, maximum_march_grid_count,
		photon_num, (int)(photon_num*PHOTON_LOWER_LIMIT), photon_num_trace_dev,
		grid_size[0], grid_size[1], grid_size[2], mbbmin, mbbmax, world_size,
		environment_map, environment_map_color, view_map, view_map_color
#ifdef DEBUG_PATH
		, debug_path
#endif
		);

	printLastError();
	cudaDeviceSynchronize();

	std::cout << "March photons Time: " << stopTiming() << std::endl;

#ifdef DEBUG_PATH
	if (is_light_photon)
	{
		Photon* tphotons;
		tphotons = (struct Photon *)malloc(photon_num*sizeof(struct Photon));
		cudaMemcpy(tphotons, photon_list, photon_num*sizeof(struct Photon), cudaMemcpyDeviceToHost);
		float3* path;
		path = (float3*)malloc(photon_num * 100 * sizeof(float3));
		cudaMemcpy(path, debug_path, photon_num * 100 * sizeof(float3), cudaMemcpyDeviceToHost);
		paths->clear();
		for (int i = 0; i < photon_num; i++)
		if ((tphotons + i)->iteration_count > 30)
		{
			for (int j = 0; j < (tphotons + i)->iteration_count; j++){
				paths->push_back(glm::vec3((path + i * 100 + j)->x, (path + i * 100 + j)->y, (path + i * 100 + j)->z));
			}

			paths->push_back(glm::vec3(-5, -5, -5));

			if (0 && (tphotons + i)->iteration_count < 0)
				cout << "photon " << i << " has an iteration < 0, total: " << photon_num
				<< " pos:" << (tphotons + i)->position[0] << " " << (tphotons + i)->position[1] << " " << (tphotons + i)->position[2]
				<< " " << (tphotons + i)->direction[0] << " " << (tphotons + i)->direction[1] << " " << (tphotons + i)->direction[2] << endl;


			break;
		}
		std::cout << "About " << paths->size() << " points in path" << endl;


		free(tphotons);
		free(path);
	}
		cudaFree(debug_path);

#endif


	// TODO smooth radiance distribution

	// smooth env and view maps
	startTiming();
	SmoothMapLinear << <(W_WIDTH*W_HEIGHT / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(view_map_color, W_WIDTH, W_HEIGHT, step_len);
	std::cout << "SmoothMapLinear view_map_color Time: " << stopTiming() << std::endl;

	cudaDeviceSynchronize();

	startTiming();
	//SmoothMapGaussian << <(W_WIDTH*W_HEIGHT / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(environment_map, envmap_w, envmap_h step_len);
	//SmoothMapGaussian << <(W_WIDTH*W_HEIGHT / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK >> >(view_map_color, W_WIDTH, W_HEIGHT, step_len);

	std::cout << "SmoothMapGaussian environment_map Time: " << stopTiming() << std::endl;


	cudaFree(photon_num_trace_dev);
	//cudaFree(photon_list);

	return 0;
}






/*
float GetPix(int x, int y)
{
return (pixels[(width*y + x) * 3] + pixels[(width*y + x) * 3 + 1] + pixels[(width*y + x) * 3 + 2]) / 3.0;
}
*/




__host__ int GeneratePhotons(bool is_light_photon, GLuint renderedTexture, glm::vec3& lightpos, glm::vec3& light_radiance, int& step_len
#ifdef DEBUG_PATH
	, vector<glm::vec3> *photons_debug
#endif
	)
{
#ifdef DEBUG_PATH
	photons_debug->clear();
#endif

	startTiming();

	int INTER_NUM = 1;
	int STEP_LEN = 1;

	GLfloat* pixels;
	pixels = (GLfloat*)malloc(W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	glBindTexture(GL_TEXTURE_2D, renderedTexture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, pixels);


	int amount = 0;
	int total = W_WIDTH*W_HEIGHT * 3;
	for (int i = 0; i < total; i += 3) {
		if (pixels[i] > 0
			|| pixels[i + 1] > 0
			|| pixels[i + 2] > 0) amount++;

	}


	if (amount > 1 && amount < PHOTON_NUM_LOWER_LIMIT){
		INTER_NUM = (int)(PHOTON_NUM_LOWER_LIMIT / amount);
	}
	if (INTER_NUM > 100) INTER_NUM = 100;

	/*
	if (amount > PHOTON_NUM_UPPER_LIMIT){
	STEP_LEN = (amount) / PHOTON_NUM_UPPER_LIMIT;
	}
	if (STEP_LEN < 1) STEP_LEN = 1;
	*/
	step_len = STEP_LEN;


	Photon *photons;
	photons = (struct Photon*)malloc((INTER_NUM)*amount*sizeof(struct Photon));
	Photon * index = photons;


	amount = 0;
	float x0[3];
	float x1[3];
	int line_bytes = W_WIDTH * 3;
	for (int x = 0; x<W_WIDTH - 1; x += STEP_LEN){
		for (int y = 0; y<W_HEIGHT; y += 1){
			// TODO /255 is not needed for texture
			for (int ii = 0; ii<3; ii++)
				x0[ii] = (float)pixels[line_bytes*y + x * 3 + ii] * (meshBoundingBoxMax - meshBoundingBoxMin)[ii] + meshBoundingBoxMin[ii];


			if (x0[0]>0 || x0[1]>0 || x0[2]>0)
			{
				//cout << "pixel: " << x0[0] << " " << x0[1] << " " << x0[2] << endl;
				for (int ii = 0; ii<3; ii++)
					x1[ii] = (float)pixels[line_bytes*y + x * 3 + STEP_LEN * 3 + ii] * (meshBoundingBoxMax - meshBoundingBoxMin)[ii] + meshBoundingBoxMin[ii];

				// interpolate
				if (x1[0]>0 || x1[1] > 0 || x1[2] > 0){
					//cout << "interpolate: " << x0[0] << " " << x0[1] << " " << x0[2] << " and " << x1[0] << " " << x1[1] << " " << x1[2] << endl;
					for (int k = 0; k < INTER_NUM; k++){
						for (int ii = 0; ii < 3; ii++){
							index->position[ii] = x0[ii] + 1.0*k / INTER_NUM*(x1[ii] - x0[ii]);
							index->direction[ii] = index->position[ii] - lightpos[ii];
							if (is_light_photon)
								index->residual_radiance[ii] = light_radiance[ii];
						}

						if (!is_light_photon){
							// write screen coordinates for view rays
							index->residual_radiance[0] = x;
							index->residual_radiance[1] = y;
						}

						// normalize the direction and initialize it as v0
						// @see <Interactive Relighting of Dynamic Refractive Objects>[ACM Trans. Graph. 27, 3, Article 35, 2008, page 3]
						float n = CalculateRefractive(index->position[0], index->position[1], index->position[2]);
						float direction_len = GetVecLen(index->direction[0], index->direction[1], index->direction[2]);
						for (int ii = 0; ii < 3; ii++){
							index->direction[ii] = n * index->direction[ii] / direction_len;
						}
						//index->is_dead = -1;

#ifdef DEBUG_PATH
						photons_debug->push_back(glm::vec3(index->position[0], index->position[1], index->position[2]));
#endif
						index++;
						amount++;
					}
				}
				else { // last one in the line segment
					for (int ii = 0; ii < 3; ii++){
						index->position[ii] = x0[ii];
						index->direction[ii] = index->position[ii] - lightpos[ii];
						if (is_light_photon)
							index->residual_radiance[ii] = light_radiance[ii];
					}

					if (!is_light_photon){
						// write screen coordinates for view rays
						index->residual_radiance[0] = x;
						index->residual_radiance[1] = y;
					}


					// normalize the direction and initialize it as v0
					// @see <Interactive Relighting of Dynamic Refractive Objects>[ACM Trans. Graph. 27, 3, Article 35, 2008, page 3]
					float n = CalculateRefractive(index->position[0], index->position[1], index->position[2]);
					float direction_len = GetVecLen(index->direction[0], index->direction[1], index->direction[2]);
					for (int ii = 0; ii < 3; ii++){
						index->direction[ii] = n * index->direction[ii] / direction_len;
					}
					//index->is_dead = -1;

#ifdef DEBUG_PATH
					photons_debug->push_back(glm::vec3(index->position[0], index->position[1], index->position[2]));
#endif
					index++;
					amount++;
				}


			}
		}
	}

	// normalise radiance
	/*
	for (int i = 0; i < amount; i++){
	for (int ii = 0; ii < 3; ii++){
	photons[i].residual_radiance[ii] = photons[i].residual_radiance[ii] / amount;
	}
	}
	*/

	//cudaMalloc((void**)&photon_list, amount*sizeof(struct Photon)); // here amount includes the interpolated ones.
	//cudaMemset(photon_list, 0, amount*sizeof(struct Photon));
	cudaMemcpy(photon_list, photons, amount*sizeof(struct Photon), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();


	std::cout << "Generate and copy photon Time: " << stopTiming() << std::endl;
	std::cout << "  - " << amount << " photons with " << INTER_NUM - 1 << " interpolation." << std::endl;


	// TODO maybe need to pull back the photon a bit little.




	free(pixels);
	free(photons);

	return amount;
}


__host__ GLuint GetEnvironmentMapTex(bool is_update)
{
	if (!is_update)
		return envmap_texture_id;

	startTiming();

	cudaMemcpy(envmap_texture, environment_map, envmap_w * envmap_h * 3 * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, envmap_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, envmap_w, envmap_h, 0, GL_RGB, GL_FLOAT, envmap_texture);

	std::cout << "GetEnvironmentMapTex Time: " << stopTiming() << std::endl;

	return envmap_texture_id;
}

__host__ GLuint GetViewMapTex(bool is_update)
{
	if (!is_update)
		return view_map_texture_id;

	startTiming();

	cudaMemcpy(view_map_texture, view_map, W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, view_map_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, W_WIDTH, W_HEIGHT, 0, GL_RGB, GL_FLOAT, view_map_texture);

	std::cout << "GetViewMapTex Time: " << stopTiming() << std::endl;

	return view_map_texture_id;
}

__host__ GLuint GetViewMapColorTex(bool is_update)
{
	if (!is_update)
		return view_map_color_texture_id;

	startTiming();

	cudaMemcpy(view_map_color_texture, view_map_color, W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, view_map_color_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, W_WIDTH, W_HEIGHT, 0, GL_RGB, GL_FLOAT, view_map_color_texture);

	std::cout << "GetViewMapColorTex Time: " << stopTiming() << std::endl;

	return view_map_color_texture_id;
}

__host__ void SetCudaSideParams(glm::vec3 meshBoundingBoxMin_in, glm::vec3 meshBoundingBoxMax_in)
{
	meshBoundingBoxMin = meshBoundingBoxMin_in;
	meshBoundingBoxMax = meshBoundingBoxMax_in;

	mbbmin.x = meshBoundingBoxMin.x;
	mbbmin.y = meshBoundingBoxMin.y;
	mbbmin.z = meshBoundingBoxMin.z;
	mbbmax.x = meshBoundingBoxMax.x;
	mbbmax.y = meshBoundingBoxMax.y;
	mbbmax.z = meshBoundingBoxMax.z;

	grid_size[0] = (meshBoundingBoxMax - meshBoundingBoxMin).x / N;
	grid_size[1] = (meshBoundingBoxMax - meshBoundingBoxMin).y / N;
	grid_size[2] = (meshBoundingBoxMax - meshBoundingBoxMin).z / N;
}



__host__ void static printLastError(){
	cudaDeviceSynchronize();
	cudaError_t cet = cudaGetLastError();
	if (cudaSuccess != cet){
		printf("error: %s\n", cudaGetErrorString(cet));
		fflush(stdout);
		system("pause");
		exit(1);
	}
}


//cudaArray* array;
//cudaGraphicsMapResources(1, &renderedTexture, 0);
//cudaGraphicsSubResourceGetMappedArray(&array, renderedTexture, 0, 0);

