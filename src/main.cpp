
#include "main.h"
#include "commoninclude.h"
#include "photon.h"

//string path_prefix = "C:\\Users\\ying\\Desktop\\Dynamic-Refractive-Relighting\\windows\\";
string path_prefix = "../../";


int main(int argc, char** argv){

#if defined(__GNUC__)
	path_prefix = "";
#endif

	bool loadedScene = false;

	int choice = 1;
	cout << "Please type which scene to load? '1'(two balls), '2'(bunny)." << endl;
	cout << "Press ENTER after the number input :)\n" << endl;
	//cin >> choice;

	string local_path = path_prefix + "../objs/";
	string data = local_path + "twoballs.obj";
	if (choice == 1)
		data = local_path + "twoballs.obj";
	else if (choice == 2)
		data = local_path + "bunny.obj";

	mesh = new obj();
	objLoader* loader = new objLoader(data, mesh);
	mesh->buildVBOs();
	mesh->setColor(glm::vec3(0.388, 0.239, 0.129));
	meshes.push_back(mesh);
	delete loader;
	loadedScene = true;

	
	SetupWorld();
	SetCudaSideParams(boundingBoxMin, boundingBoxMax);
	OnStarting_Voxel();

	// Pre-added internals. These can be added interactively.
	if (choice == 1) {
		selectedInternal.push_back(glm::vec3(1.4658, 2.3178, 1.5732) + glm::vec3(boundingBoxExpand / 2.0, boundingBoxExpand / 2.0, boundingBoxExpand / 2.0));
		selectedInternal.push_back(glm::vec3(2.4636, 0.405, 2.295) + glm::vec3(boundingBoxExpand / 2.0, boundingBoxExpand / 2.0, boundingBoxExpand / 2.0));
	}
	else if (choice == 2) {
		selectedInternal.push_back(glm::vec3(1.8156, 1.2288, 1.8822) + glm::vec3(boundingBoxExpand / 2.0, boundingBoxExpand / 2.0, boundingBoxExpand / 2.0));
	}

	
	frame = 0;
	seconds = time(NULL);
	fpstracker = 0;

	redo_level = FROM_VOXELIZATION;
	if (init(argc, argv)) {
		// GLFW main loop
		mainLoop();
	}

	system("PAUSE");
	return 0;
}


#ifdef DEBUG_PATH
vector<glm::vec3> photons_debug;
vector<glm::vec3> paths;
#endif


void RenderToTexture(){
	if (VOXELIZE) return;

	if (lightRenderToTexture)
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
	else
		glDrawBuffer(GL_COLOR_ATTACHMENT1);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// set the shader and draw
	glUniform1i(is_depth_location, 1);
	if (lightRenderToTexture)
		glDrawArrays(GL_TRIANGLES, 0, vbosize);
	else
		DrawSkyBox(false);
	glUniform1i(is_depth_location, 0);


	if (lightRenderToTexture) {
		lightRenderToTexture = false;
		int step_len = 0;
		int amount = GeneratePhotons(true, lightRenderedTexture, lightpos, light_radiance, step_len
#ifdef DEBUG_PATH
			, &photons_debug
#endif
			);
		MarchPhotons(true, amount, step_len, minimum_march_grid, maximum_march_grid
#ifdef DEBUG_PATH
			, &paths
#endif
			);

		GetEnvironmentMapTex(true);
	}
	else{
		//cameraRenderToTexture = false;
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		int step_len = 0;
		int amount = GeneratePhotons(false, cameraRenderedTexture, eye, glm::vec3(0.f, 0.f, 0.f), step_len
#ifdef DEBUG_PATH
			, &photons_debug
#endif
			);
		MarchPhotons(false, amount, step_len, 1, 1
#ifdef DEBUG_PATH
			, &paths
#endif
			);

		GetViewMapTex(true);
		GetViewMapColorTex(true);
	}
}

void mainLoop() {
	while (!glfwWindowShouldClose(window)){
		double times, timed = 0.0f;
		times = clock();

		if (redo_level == FROM_VOXELIZATION){
			voxelizeScene();
			CreateMipmap(n_threshold_1);
			redo_level++;
		}

		if (redo_level == FROM_LIGHTING){
			lightRenderToTexture = true;
			runGL();
			RenderToTexture();
			redo_level++;
			lightRenderToTexture = false;
		}

		if (redo_level == FROM_CAMERA){
			cameraRenderToTexture = true;
			runGL();
			RenderToTexture();
			cameraRenderToTexture = false;
		}

		
		if (USE_CUDA_RASTERIZER) {
			runCuda();
		}
		else {
			runGL();
		}


		if (USE_CUDA_RASTERIZER) {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			glBindTexture(GL_TEXTURE_2D, displayImage);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W_WIDTH, W_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		}
		else {
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			if (drawMesh){
				if (redo_level < RAW_VIEW){
					// bind traced maps
					glActiveTexture(GL_TEXTURE1);
					glBindTexture(GL_TEXTURE_2D, GetViewMapTex(false));
					glUniform1i(mesh_radiance_location, 1);
					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, GetViewMapColorTex(false));
					glUniform1i(mesh_color_location, 0);

					glUniform1i(is_transparent_mesh_location, 1);
				}
				else{
					glUniform1i(is_transparent_mesh_location, 0);
				}

				glDrawArrays(GL_TRIANGLES, 0, vbosize);

				glUniform1i(is_transparent_mesh_location, 0);
			}

			DrawRenderedTexture();
			DrawCoordinate();
			DrawSkyBox(true);

#ifdef DEBUG_PATH
			if (drawPhotonGenPath)
				DrawPhotonGenPath();
			if (drawPhotonMarchPath)
				DrawPhotonMarchPath();
#endif
		}


		glfwPollEvents();
		glfwSwapBuffers(window);
		
		{
			timed = clock();
			double diffms = (double)(timed - times) / 1000.0;

			time_t seconds2 = time(NULL);
			if (seconds2 - seconds >= 1){
				fps = fpstracker / (seconds2 - seconds);
				fpstracker = 0;
				seconds = seconds2;
			}
			string title = "Dynamic Refractive | " + utilityCore::convertIntToString((int)fps) + " FPS";
			glfwSetWindowTitle(window, title.c_str());
		}
	}
	glfwDestroyWindow(window);
	glfwTerminate();

	OnExiting_Voxel();
}


void runGL() {

	float newcbo[] = { 
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		1.0, 0.0, 0.0 };


	//Update data
	if (VOXELIZE) {
		vbo = m_vox.vbo;
		vbosize = m_vox.vbosize;
		cbo = m_vox.cbo;
		cbosize = m_vox.cbosize;
		ibo = m_vox.ibo;
		ibosize = m_vox.ibosize;
		nbo = m_vox.nbo;
		nbosize = m_vox.nbosize;
	}
	else {
		vbo = mesh->getVBO();
		vbosize = mesh->getVBOsize();
		cbo = mesh->getCBO();
		cbosize = mesh->getCBOsize();
		ibo = mesh->getIBO();
		ibosize = mesh->getIBOsize();
		nbo = mesh->getNBO();
		nbosize = mesh->getNBOsize();
		tbo = mesh->getTBO();
		tbosize = mesh->getTBOsize();
	}

	if (lightRenderToTexture) {
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		float dis = (boundingBoxMax - boundingBoxMin).length() * 0.87; // 0.87 is just the magic to let the light see all the mesh :-)
		glm::vec3 direction = center - lightpos;
		glm::vec3 cam = center - glm::vec3((direction).x * dis, (direction).y * dis, (direction).z * dis);
		glm::vec3 up = glm::vec3(direction.y, -direction.x, 0);
		if (glm::length(up) < 0.0001) up = glm::vec3(-direction.z, 0, direction.x);
		view = glm::lookAt(cam, center, up);
	}
	else {
		if (cameraRenderToTexture)
			glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
	}

	if (VOXELIZE)
		glUniform1i(is_voxel_location, 1);
	else
		glUniform1i(is_voxel_location, 0);


	modelview = view * glm::mat4();
	glm::mat4 mvp = projection*modelview;

	// Send the MV, MVP, and Normal Matrices
	glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(glm::mat4()));
	glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm::value_ptr(mvp));
	glUniformMatrix4fv(proj_location, 1, GL_FALSE, glm::value_ptr(projection));
	glm::mat3 norm_mat = glm::mat3(glm::transpose(glm::inverse(model)));
	glUniformMatrix3fv(norm_location, 1, GL_FALSE, glm::value_ptr(norm_mat));

	// Send the light position
	glUniform3fv(light_location, 1, glm::value_ptr(lightpos));

	// Send the VBO
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, vbosize*sizeof(float), vbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	// Send the NBO
	glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, nbosize*sizeof(float), nbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

	// Send the CBO
	glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
	glBufferData(GL_ARRAY_BUFFER, cbosize*sizeof(float), cbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(2);

	/*
	// Send the TBO
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, tbosize*sizeof(float), tbo, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(3);
	*/

	// Send the mesh bounding box size
	glUniform3fv(bounding_min_location, 1, glm::value_ptr(boundingBoxMin));
	glUniform3fv(bounding_max_location, 1, glm::value_ptr(boundingBoxMax));

	if (!lightRenderToTexture || !cameraRenderToTexture) {
		frame++;
		fpstracker++;
	}
}


void voxelizeScene() {

	//Construct target mesh
	Mesh m_in;
	m_in.vbo = mesh->getVBO();
	m_in.vbosize = mesh->getVBOsize();
	m_in.nbo = mesh->getNBO();
	m_in.nbosize = mesh->getNBOsize();
	m_in.cbo = mesh->getCBO();
	m_in.cbosize = mesh->getCBOsize();
	m_in.ibo = mesh->getIBO();
	m_in.ibosize = mesh->getIBOsize();
	m_in.tbo = mesh->getTBO();
	m_in.tbosize = mesh->getTBOsize();

	//Load cube
	Mesh m_cube;
	obj* cube = new obj();
	string cubeFile = path_prefix + "../objs/cube.obj";
	objLoader* loader = new objLoader(cubeFile, cube);
	cube->buildVBOs();
	//cube->setColor(glm::vec3(116 / 255.0, 233 / 255.0, 229 / 255.0));
	m_cube.vbo = cube->getVBO();
	m_cube.vbosize = cube->getVBOsize();
	m_cube.nbo = cube->getNBO();
	m_cube.nbosize = cube->getNBOsize();
	m_cube.cbo = cube->getCBO();
	m_cube.cbosize = cube->getCBOsize();
	m_cube.ibo = cube->getIBO();
	m_cube.ibosize = cube->getIBOsize();
	m_cube.tbo = cube->getTBO();
	m_cube.tbosize = cube->getTBOsize();
	delete cube;

	//Voxelize
	if (OCTREE){
		//voxelizeSVOCubes(m_in, &tex, m_cube, m_vox);
	}
	else {
		if (m_vox.vbo) free(m_vox.vbo);
		if (m_vox.ibo) free(m_vox.ibo);
		if (m_vox.nbo) free(m_vox.nbo);
		if (m_vox.cbo) free(m_vox.cbo);
		m_vox.vbosize = 0;
		m_vox.ibosize = 0;
		m_vox.nbosize = 0;
		m_vox.cbosize = 0;

		voxelizeToCubes(m_in, &tex, m_cube, m_vox, selectedInternal);
	}
}
																    

void SetupWorld()
{
	glm::vec3 span = mesh->getBoundingMax() - mesh->getBoundingMin();
	float size = glm::max(glm::max(span.x, span.y), span.z);
	size += boundingBoxExpand;
	boundingBoxMin.x = (mesh->getBoundingMax() + mesh->getBoundingMin()).x / 2.0 - size / 2.0;
	boundingBoxMin.y = (mesh->getBoundingMax() + mesh->getBoundingMin()).y / 2.0 - size / 2.0;
	boundingBoxMin.z = (mesh->getBoundingMax() + mesh->getBoundingMin()).z / 2.0 - size / 2.0;
	boundingBoxMax.x = (mesh->getBoundingMax() + mesh->getBoundingMin()).x / 2.0 + size / 2.0;
	boundingBoxMax.y = (mesh->getBoundingMax() + mesh->getBoundingMin()).y / 2.0 + size / 2.0;
	boundingBoxMax.z = (mesh->getBoundingMax() + mesh->getBoundingMin()).z / 2.0 + size / 2.0;

	world_size = size/2.0;
}


bool init(int argc, char* argv[]) {
	glfwSetErrorCallback(errorCallback);
	if (!glfwInit()) {
		return false;
	}

	//readBMP((path_prefix + string("../texture/bottom.bmp")).c_str(), tex);
	window = glfwCreateWindow(W_WIDTH, W_HEIGHT, "Dynamic Refractive", NULL, NULL);
	if (!window){
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, MouseClickCallback);
	glfwSetCursorEnterCallback(window, CursorEnterCallback);
	glfwSetCursorPosCallback(window, CursorCallback);
	glfwSetScrollCallback(window, ScrollCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK){
		return false;
	}

	// Initialize other stuff
	if (USE_CUDA_RASTERIZER) {
		initCudaTextures();
		initCudaVAO();
		initCuda();
		initCudaPBO();
		initPassthroughShaders();
		glActiveTexture(GL_TEXTURE0);
	}
	else {
		initGL();
		initDefaultShaders();
		initRenderFrameBuffer();
		initTexture();
	}

	return true;
}

void initGL() {
	glGenBuffers(4, buffers);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLineWidth(2.0);
}

GLuint initDefaultShaders() {
	const char *attribLocations[] = { "v_position", "v_normal", "v_color", "v_texcoord" };

	string vs, fs;
	/*
	if (VOXELIZE) {
	vs = path_prefix + "../shaders/voxels.vert";
	fs = path_prefix + "../shaders/voxels.frag";
	} else */
	{
		vs = path_prefix + "../src/shaders/default.vert";
		fs = path_prefix + "../src/shaders/default.frag";
	}
	const char *vertShader = vs.c_str();
	const char *fragShader = fs.c_str();

	default_program = glslUtility::createProgram(attribLocations, 4, vertShader, fragShader);

	glUseProgram(default_program);
	model_location = glGetUniformLocation(default_program, "u_modelMatrix");
	mvp_location = glGetUniformLocation(default_program, "u_mvpMatrix");
	proj_location = glGetUniformLocation(default_program, "u_projMatrix");
	norm_location = glGetUniformLocation(default_program, "u_normMatrix");
	light_location = glGetUniformLocation(default_program, "u_light");

	is_swatch_location = glGetUniformLocation(default_program, "is_swatch_vs");
	is_voxel_location = glGetUniformLocation(default_program, "is_voxel_fs");
	is_depth_location = glGetUniformLocation(default_program, "is_depth_fs");
	is_sky_box_location = glGetUniformLocation(default_program, "is_sky_box_fs");
	is_transparent_mesh_location = glGetUniformLocation(default_program, "is_transparent_mesh_fs");
	bounding_min_location = glGetUniformLocation(default_program, "bounding_min");
	bounding_max_location = glGetUniformLocation(default_program, "bounding_max");
	sky_box_color_tex_location = glGetUniformLocation(default_program, "skyBoxColorTex");
	sky_box_radiance_tex_location = glGetUniformLocation(default_program, "skyBoxRadianceTex");
	mesh_color_location = glGetUniformLocation(default_program, "meshColorTex");
	mesh_radiance_location = glGetUniformLocation(default_program, "meshRadianceTex");

	glUniform1i(is_swatch_location, 0);
	glUniform1i(is_voxel_location, 0);
	glUniform1i(is_depth_location, 0);
	glUniform1i(is_sky_box_location, 0);

	return default_program;
}




void initRenderFrameBuffer()
{
	// The texture we're going to render to
	glGenTextures(1, &lightRenderedTexture);
	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, lightRenderedTexture);
	// Give an empty image to OpenGL ( the last "0" means "empty" )
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W_WIDTH, W_HEIGHT, 0, GL_RGB, GL_FLOAT, 0);
	// Poor filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


	// Do the same to the rendered texture for camera
	glGenTextures(1, &cameraRenderedTexture);
	glBindTexture(GL_TEXTURE_2D, cameraRenderedTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W_WIDTH, W_HEIGHT, 0, GL_RGB, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


	// Gen framebuffer
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	// Set the list of draw buffers.
	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, DrawBuffers); // "2" is the size of DrawBuffers
	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Set colour attachements
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightRenderedTexture, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, cameraRenderedTexture, 0);
	// The depth buffer
	GLuint depthrenderbuffer;
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, W_WIDTH, W_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);


	// Always check that our framebuffer is ok
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
		cout << "Frame buffer status check failed" << endl;
		return;
	}

	// Share this texture with CUDA
	//cudaGraphicsGLRegisterImage(&cudarRenderedTexture, lightRenderedTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);

	// reset to default
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void initTexture()
{
	//bottom_tex = load_texture((path_prefix + string("../texture/bottom.bmp")).c_str());
	//back_tex = load_texture((path_prefix + string("../texture/back.bmp")).c_str());
	//left_tex = load_texture((path_prefix + string("../texture/left.bmp")).c_str());
	//right_tex = load_texture((path_prefix + string("../texture/right.bmp")).c_str());

	background_tex = load_texture((path_prefix + string("../texture/envmap.bmp")).c_str());
	background_tex_blur = load_texture((path_prefix + string("../texture/envmap_blur.bmp")).c_str());

	initCudaSideTexture(background_tex);
}


/*
 *
 * Implementations of draw and glew related functions.
 * Use header file to both share the global variables
 * and in the meanwhile not make this file too long.
 *
 */
#include "main_inline.h"
