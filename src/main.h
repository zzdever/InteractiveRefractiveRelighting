
#ifndef MAIN_H
#define MAIN_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glslUtil/glslUtility.hpp>
#include <objUtil/objloader.h>


#include "timingUtils.h"
#include "commoninclude.h"
//#include "rasterizeKernels.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "voxelization.h"
//#include "svo.h"
#include "photon.h"


using namespace std;

#define NOT_OCCUPIED -1
#define LEFT_INDEX 1
#define BACK_INDEX  2
#define RIGHT_INDEX  3
#define FRONT_INDEX  4
#define TOP_INDEX 	5
#define BOTTOM_INDEX  6

obj* mesh;
vector <obj*> meshes;
//Voxelized mesh
Mesh m_vox;
Mesh m_light;
bmp_texture tex;
// GLuint bottom_tex, back_tex, left_tex, right_tex;
GLuint background_tex;
GLuint background_tex_blur;
GLuint environment_tex;
GLuint environment_color_tex;
GLuint FramebufferName = 0;
GLuint lightRenderedTexture;
GLuint cameraRenderedTexture; 
GLuint test_tex;
//cudaGraphicsResource_t cudaRenderedTexture;


// Params
// TODO adjust this by user
vector<glm::vec3> selectedInternal;
glm::vec3 internalSelector(0.,0.,0.);
int minimum_march_grid = 1;
int maximum_march_grid = 2;
float n_threshold_1 = 0.01;
float boundingBoxExpand = 1.0;


// Render control
float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo;
int nbosize;
float* tbo;
int tbosize;
vector<glm::vec4>* texcoord;

bool VOXELIZE = false;
bool drawMesh = true;
bool lightRenderToTexture = false;
bool cameraRenderToTexture = false;
bool drawPhotonGenPath = false;
bool drawPhotonMarchPath = false;
bool isPathOriginLight = true;
bool isMoveLight = false;
bool isDrawOriginal = false;
bool isMoveSelectedInternal = false;

// Redo control
int redo_level = 0;
#define FROM_VOXELIZATION 1
#define FROM_LIGHTING 2
#define FROM_CAMERA 3
#define RAW_VIEW 4

// Shader programs
GLuint default_program;

// Uniform locations for the GL shaders
GLuint model_location;
GLuint mvp_location;
GLuint proj_location;
GLuint norm_location;
GLuint light_location;

GLuint is_swatch_location;
GLuint is_voxel_location;
GLuint is_depth_location;
GLuint is_sky_box_location;
GLuint is_transparent_mesh_location;
GLuint bounding_min_location;
GLuint bounding_max_location;
GLuint sky_box_color_tex_location;
GLuint sky_box_radiance_tex_location;
GLuint mesh_color_location;
GLuint mesh_radiance_location;

// VAO's for the GL pipeline
GLuint buffers[3];


// View control
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;
int mode = 0;
bool barycenter = false;
//bool operation_block = false;

glm::vec3 eye(0.0f, 7.0f, 0.0f);
glm::vec3 center(0.0f, 0.0f, 0.0f);
float zNear = 0.001;
float zFar = 10000.0;
glm::mat4 projection = glm::perspective(60.0f, (float)(W_WIDTH) / (float)(W_HEIGHT), zNear, zFar);
glm::mat4 model = glm::mat4();
glm::mat4 view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
glm::mat4 modelview = view * glm::mat4();
glm::vec3 lightpos = glm::vec3(-10.0f, 7.0f, 0.0f);
glm::vec3 light_radiance = glm::vec3(1.0f, 1.0f, 1.0f);

//Make the camera move on the surface of sphere
float vPhi = 0.0f;
float vTheta = 3.14105926f / 2.0f;
float R = glm::length(eye);

// world params
glm::vec3 boundingBoxMin, boundingBoxMax;



// Main
int main(int argc, char** argv);
void loadMultipleObj(int choice, int type);
void mainLoop();
bool init(int argc, char* argv[]);
void SetupWorld();
void voxelizeScene();
void runGL();

// GL Rasterizer 
void initGL();
GLuint initDefaultShaders();
void initRenderFrameBuffer();
GLuint buildupEnvColorMap();
void initTexture();
void DrawRenderedTexture();
void DrawCoordinate();
void DrawSkyBox(bool is_normal_draw);
#ifdef DEBUG_PATH
void DrawPhotonGenPath();
void DrawPhotonMarchPath();
#endif


// CUDA Rasterizer
void initCudaPBO();
void initCuda();
void initCudaTextures();
void initCudaVAO();
GLuint initPassthroughShaders();
void runCuda();

// Cleanup
void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);


// GUI control
double MouseX = 0.0;
double MouseY = 0.0;
bool LB = false;
bool RB = false;
bool MB = false;
bool inwindow = false;
GLFWwindow *window;

void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void MouseClickCallback(GLFWwindow *window, int button, int action, int mods);
void CursorCallback(GLFWwindow *window, double x, double y);
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void CursorEnterCallback(GLFWwindow *window, int entered);

#ifdef __APPLE__
void display();
#else
void display();
void keyboard(unsigned char key, int x, int y);
#endif

// Tools
GLuint load_texture(const char* file_name);
void readBMP(const char* filename, bmp_texture &tex);

#endif
