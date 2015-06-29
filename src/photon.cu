#include "photon.h"

#if 0

Node *mipmap;
float grid_size[3];
float cube_w, cube_l, cube_h;




__device__ inline int Sign(float v)
{
	if (v>0)
		return 1;
	else if (v<0)
		return -1;
	else
		return 0;
}

__device__ inline float CalculateRefractive(float x, float y, float z)
{
#ifdef LINEAR
	return fabs(y);
#endif

#define CIRCLE
#ifdef CIRCLE
	return x*x + y*y;
#endif
}

__device__ inline float GetRefractive(int grid_x, int grid_y)
{
	return CalculateRefractive(grid_size[0] * grid_x, grid_size[1] * grid_y, 0);

#ifdef FROM_TEXTURE
	int tileW = width / DIM;
	int tileH = height / DIM;


	float ave = 0;
	int x0 = dimx*tileW;
	int y0 = dimy*tileH;

	for (int x = 0; x<tileW; x++){
		for (int y = 0; y<tileH; y++) {
			ave += GetPix(x0 + x, y0 + y);
		}
	}

	return (ave / (tileW*tileH));
#endif
}



__device__ inline void GetGradient(float gradient[], float x, float y, float z)
{
#ifdef LINEAR
	gradient[0] = 0;
	if (y > 0)
		gradient[1] = 1;
	else if (y<0)
		gradient[1] = -1;
	else
		gradient[1] = 0;
	gradient[2] = 0;
#endif

#define CIRCLE
#ifdef CIRCLE
	gradient[0] = 2 * x;
	gradient[1] = 2 * y;
	gradient[2] = 0;
#endif
}


__device__ inline Node * GetLevelOrigin(int level){
	if (level == 0)
		return mipmap;
	else if (level > log_N + 1)
		return NULL;
	else
		return mipmap + (1 << (log_N + 1)) - (1 << (log_N + 1 - level));
}

__device__ inline int GetLevelOriginOffset(int level){
	if (level == 0)
		return 0;
	else if (level > log_N)
		return -1;
	else
		return ((1 << (3 * log_N + 3)) - (1 << (3 * log_N - 3 * level + 3))) / 7;
}

__device__ inline int GetVoxelOffsetFromPointCoord(float x, float y, float z)
{
	int voxel_x = x / (cube_w / N);
	int voxel_y = y / (cube_l / N);
	int voxel_z = z / (cube_h / N);

	return voxel_x*N*N + voxel_y*N + voxel_z;
}




__host__ int CreateMipmap(float cube_w_param, float cube_l_param, float cube_h_param) 
{
	cube_w = cube_w_param;
	cube_l = cube_l_param;
	cube_h = cube_h_param;
	grid_size[0] = cube_w / N;
	grid_size[1] = cube_l / N;
	grid_size[2] = cube_h / N;


	int totalsize = sizeof(struct Node)*((1 << (3 * log_N + 3)) - 1) / 7;
	//mipmap = (Node*)malloc(totalsize);
	cudaMalloc((void**) &mipmap, totalsize);
	if (!mipmap){
		cout << "Mipmap allocate failed" << endl;
		return -1;
	}


	float n;
	// mipmap level 0
	int threads = (1 << (3 * (log_N - 0)));
	int dim = (1 << (log_N - 0));
	Node * origin = mipmap + GetLevelOriginOffset(0);
	for (int id = 0; id<threads; id++){
		// following code is in parallel
		n = GetRefractive((id / dim / dim) % dim, (id / dim) % dim);
		(origin + id)->min = n;
		(origin + id)->max = (origin + id)->min;
		(origin + id)->level = 0;
	}


	// create mipmap
	for (int level = 1; level <= log_N; level++){
		int threads = (1 << (3 * (log_N - level)));
		int dim = (1 << (log_N - level));
		Node * origin = mipmap + GetLevelOriginOffset(level);
		cout << "origin " << origin - mipmap << endl;
		Node * origin_pre = mipmap + GetLevelOriginOffset(level - 1);
		cout << "origin_pre " << origin_pre - mipmap << endl;

		for (int id = 0; id<threads; id++){
			// following code is in parallel
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

		}
	}


	// label level
	float threshold = 24;
	for (int level = log_N; level>0; level--){
		int threads = (1 << (3 * (log_N - level)));
		int dim = (1 << (log_N - level));
		Node * origin = mipmap + GetLevelOriginOffset(level);
		cout << "origin " << origin - mipmap << endl;
		Node * origin_pre = mipmap + GetLevelOriginOffset(level - 1);
		cout << "origin_pre " << origin_pre - mipmap << endl;

		for (int id = 0; id<threads; id++){
			// following code is in parallel
			if ((origin + id)->level < level) /*actually ==0*/
			{
				if ((origin + id)->max - (origin + id)->min < threshold){
					(origin + id)->level = level;
				}
			} // else (origin + id)->level = 0


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

	for (int id = 0; id<((1 << (3 * log_N + 3)) - 1) / 7; id++){
		cout << (origin + id)->min << " " << (origin + id)->max << " " << (origin + id)->level << endl;
	}







	// TODO read in extinction

	for (int t = 0; t<100; t++) {

		// photon marching
		//float lightpos[] = {3.0, 2.0, 1.5};
		//float photonpos[] = {0.5, 1.0, 0.5};

		float lightpos[] = { (float)((rand() % 100) / 100.0) + cube_w, (float)((rand() % 100) / 100.0) + cube_l, (float)((rand() % 100) / 100.0) + cube_h };
		float photonpos[] = { (float)((rand() % 100) / 100.0)*cube_w, (float)((rand() % 100) / 100.0)*cube_l, (float)((rand() % 100) / 100.0)*cube_h };


		float direction[] = {
			photonpos[0] - lightpos[0],
			photonpos[1] - lightpos[1],
			photonpos[2] - lightpos[2], };

		Node* voxel = mipmap + GetVoxelOffsetFromPointCoord(photonpos[0], photonpos[1], photonpos[2]);

		int region_grid_count = voxel->level + 1;
		float region_size[] = { grid_size[0] * region_grid_count, grid_size[1] * region_grid_count, grid_size[2] * region_grid_count };
		float region_origin[] = {
			((int)(photonpos[0] / region_size[0])) * region_size[0] * 1.0f,
			((int)(photonpos[1] / region_size[1])) * region_size[1] * 1.0f,
			((int)(photonpos[2] / region_size[2])) * region_size[2] * 1.0f };



		float point[3] = { -1, -1, -1 };

		int valid_count = 0;

		for (int known_i = 0; known_i < 3; known_i++) {
			if (Sign(direction[known_i]) == 0) {
				// direction is parallel to this axis, so no intersection with this face
				continue;
				//point[0] = point[1] = point[2] = -1;
			}
			else {
				point[known_i] = region_origin[known_i] + (Sign(direction[known_i]) + 1) * region_grid_count * grid_size[known_i] / 2;
				//int valid_count = 1;
				valid_count = 1;
				float tmp;
				for (int nki = 1; nki < 3; nki++) {
					int not_known_i = (known_i + nki) % 3;
					tmp = (point[known_i] - photonpos[known_i]) / direction[known_i] * direction[not_known_i] + photonpos[not_known_i];
					if (tmp < region_origin[not_known_i]
						|| tmp  > region_grid_count * grid_size[not_known_i] + region_origin[not_known_i]) {
						break;
					}
					else {
						point[not_known_i] = tmp;
						valid_count++;
					}
				}

				// we get 3 valid points coord, stop
				if (valid_count == 3) {
					break;
				}
			}

		}

		assert(point[0] >= region_origin[0]
			&& point[0] <= region_grid_count * grid_size[0] + region_origin[0]);
		assert(point[1] >= region_origin[1]
			&& point[1] <= region_grid_count * grid_size[1] + region_origin[1]);
		assert(point[2] >= region_origin[2]
			&& point[2] <= region_grid_count * grid_size[2] + region_origin[2]);


		cout << "region_origin: " << region_origin[0] << " " << region_origin[1] << " " << region_origin[2] << endl;
		cout << "valid_count: " << valid_count << endl;
		cout << lightpos[0] << " " << lightpos[1] << " " << lightpos[2] << " -> " << photonpos[0] << " " << photonpos[1] << " " << photonpos[2];
		cout << endl << "region_grid_count: " << region_grid_count << " intersection: " << point[0] << " " << point[1] << " " << point[2] << endl;
		assert(valid_count == 3);



		// TODO give new photon position and add a tiny advance here



	}



	return 0;
}








/*
float GetPix(int x, int y)
{
	return (pixels[(width*y + x) * 3] + pixels[(width*y + x) * 3 + 1] + pixels[(width*y + x) * 3 + 2]) / 3.0;
}

			   
*/








__host__ int GeneratePhotons(GLuint renderedTexture, glm::vec3& lightpos, Photon** out,
	glm::vec3 meshBoundingMin, glm::vec3 meshBoundingMax)
{
	startTiming();

	int INTER_NUM = 1;

	GLfloat* pixels;
	pixels = (float*)malloc(W_WIDTH*W_HEIGHT * 3 * sizeof(GLfloat));
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, pixels);


	int amount = 0;
	int total = W_WIDTH*W_HEIGHT * 3;
	for (int i = 0; i < total; i += 3) {
		if (pixels[i] > 0
			|| pixels[i + 1] > 0
			|| pixels[i + 2] > 0) amount++;		   

	}

	if (amount > 1 && amount < 10000){
		INTER_NUM = (int)(10000 / amount);
	}
	if (INTER_NUM > 100) INTER_NUM = 100;

	Photon* photons;
	*(out) = (struct Photon*)malloc((INTER_NUM)*amount*sizeof(struct Photon));
	photons = *out;
	Photon * index = photons;


	amount = 0;
	float x0[3];
	float x1[3];
	int line_bytes = W_WIDTH * 3;
	for (int x = 0; x<W_WIDTH - 1; x++){
		for (int y = 0; y<W_HEIGHT; y++){
			// TODO /255 is not needed for texture
			for (int ii = 0; ii<3; ii++)
				x0[ii] = (float)pixels[line_bytes*y + x * 3 + ii] * (meshBoundingMax - meshBoundingMin)[ii] + meshBoundingMin[ii];


			if (x0[0]>0 || x0[1]>0 || x0[2]>0)
			{
				//cout << "pixel: " << x0[0] << " " << x0[1] << " " << x0[2] << endl;
				for (int ii = 0; ii<3; ii++)
					x1[ii] = (float)pixels[line_bytes*y + x * 3 + 3 + ii] * (meshBoundingMax - meshBoundingMin)[ii] + meshBoundingMin[ii];

				// interpolate
				if (x1[0]>0 || x1[1] > 0 || x1[2] > 0){
					//cout << "interpolate: " << x0[0] << " " << x0[1] << " " << x0[2] << " and " << x1[0] << " " << x1[1] << " " << x1[2] << endl;
					for (int k = 0; k < INTER_NUM; k++){
						for (int ii = 0; ii < 3; ii++){
							index->position[ii] = x0[ii] + 1.0*k / INTER_NUM*(x1[ii] - x0[ii]);
							index->direction[ii] = index->position[ii] - lightpos[ii];
						}
						index++;
						amount++;
					}
				}
				else { // last one in the line segment
					for (int ii = 0; ii < 3; ii++){
						index->position[ii] = x0[ii];
						index->direction[ii] = index->position[ii] - lightpos[ii];
					}

					index++;
					amount++;
				}
			}
		}
	}




	// TODO maybe need to pull back the photon a bit little.








	//cudaArray* array;

	//cudaGraphicsMapResources(1, &renderedTexture, 0);
	//cudaGraphicsSubResourceGetMappedArray(&array, renderedTexture, 0, 0);

	//float* output;
	//cudaMalloc((void**)&output, W_WIDTH*W_HEIGHT*3*sizeof(float));

	//float *p;
	//udaMemcpy(h_data, output, size, cudaMemcpyDeviceToHost);

	/*
	int numVoxels = N*N*N;
	int* d_voxels;
	int* d_values;
	cudaMalloc((void**)&d_voxels, numVoxels*sizeof(int));
	cudaMalloc((void**)&d_values, numVoxels*sizeof(int));

	numVoxels = PhotonGen(m_in, tex, d_voxels, d_values);
	*/

	std::cout  << "Gen photon Time: " << stopTiming() << std::endl;
	std::cout << "  - " << amount << " photons with " << INTER_NUM << " interpolation." << std::endl;

	free(pixels);
	// TODO remember to free photons after copy to gpu 

	return amount;
}


#endif