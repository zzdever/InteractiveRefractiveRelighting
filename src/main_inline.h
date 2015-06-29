/*
 *
 * This file is included at the end of "main.cpp" as an implementation file.
 * DO NOT include this file again.
 *
 */




void DrawRenderedTexture()
{
	//glUniform1i(is_swatch_location, 1);
	glUseProgram(0);

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, -1.0, 10);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	//glUniform1i(is_depth_location, 0);
	{
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBindTexture(GL_TEXTURE_2D, lightRenderedTexture);
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
		glTexCoord2f(1.0, 0.0); glVertex2f(-0.6, -1.0);
		glTexCoord2f(1.0, 1.0); glVertex2f(-0.6, -0.6);
		glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -0.6);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, cameraRenderedTexture);
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0, 0.0); glVertex2f(-0.6, -1.0);
		glTexCoord2f(1.0, 0.0); glVertex2f(-0.2, -1.0);
		glTexCoord2f(1.0, 1.0); glVertex2f(-0.2, -0.6);
		glTexCoord2f(0.0, 1.0); glVertex2f(-0.6, -0.6);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, GetViewMapTex(false));
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0, 0.0); glVertex2f(-0.2, -1.0);
		glTexCoord2f(1.0, 0.0); glVertex2f(0.2, -1.0);
		glTexCoord2f(1.0, 1.0); glVertex2f(0.2, -0.6);
		glTexCoord2f(0.0, 1.0); glVertex2f(-0.2, -0.6);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, GetViewMapColorTex(false));
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0, 0.0); glVertex2f(0.2, -1.0);
		glTexCoord2f(1.0, 0.0); glVertex2f(0.6, -1.0);
		glTexCoord2f(1.0, 1.0); glVertex2f(0.6, -0.6);
		glTexCoord2f(0.0, 1.0); glVertex2f(0.2, -0.6);
		glEnd();

		

		glBindTexture(GL_TEXTURE_2D, GetEnvironmentMapTex(false));
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0, 0.0); glVertex2f(0.6, -1.0);
		glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0); //glVertex2f(0.2 + 0.4 / 3 * 4, -1.0);
		glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -0.6); //glVertex2f(0.2 + 0.4 / 3 * 4, -0.6);
		glTexCoord2f(0.0, 1.0); glVertex2f(0.6, -0.6);
		glEnd();
		glLineWidth(2.0);
		glBegin(GL_LINES);
		glVertex3f(0.6, -0.6 - 0.4 / 3 * 1, 1.0); glVertex3f(1.0, -0.6 - 0.4 / 3 * 1, 1.0);
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(0.6, -0.6 - 0.4 / 3 * 2, 1.0); glVertex3f(1.0, -0.6 - 0.4 / 3 * 2, 1.0);
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(0.6 + 0.4 / 4 * 1, -1.0, 1.0); glVertex3f(0.6 + 0.4 / 4 * 1, -0.6, 1.0);
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(0.6 + 0.4 / 4 * 2, -1.0, 1.0); glVertex3f(0.6 + 0.4 / 4 * 2, -0.6, 1.0);
		glEnd();
		glBegin(GL_LINES);
		glVertex3f(0.6 + 0.4 / 4 * 3, -1.0, 1.0); glVertex3f(0.6 + 0.4 / 4 * 3, -0.6, 1.0);
		glEnd();
	}


	// Making sure we can render 3d again
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();


	glUseProgram(default_program);
	//glUniform1i(is_swatch_location, 0);
}

void DrawCoordinate()
{
	// coord
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(3.0, 0, 0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 4.0, 0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 5.0);
	glEnd();


	glBegin(GL_LINES);
	glVertex3f(internalSelector.x + boundingBoxMin.x, internalSelector.y + boundingBoxMin.y, internalSelector.z + boundingBoxMin.z);
	glVertex3f(internalSelector.x + 3 + boundingBoxMin.x, internalSelector.y + boundingBoxMin.y, internalSelector.z + boundingBoxMin.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(internalSelector.x + boundingBoxMin.x, internalSelector.y + boundingBoxMin.y, internalSelector.z + boundingBoxMin.z);
	glVertex3f(internalSelector.x + boundingBoxMin.x, internalSelector.y + 3 + boundingBoxMin.y, internalSelector.z + boundingBoxMin.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(internalSelector.x + boundingBoxMin.x, internalSelector.y + boundingBoxMin.y, internalSelector.z + boundingBoxMin.z);
	glVertex3f(internalSelector.x + boundingBoxMin.x, internalSelector.y + boundingBoxMin.y, internalSelector.z + 3 + boundingBoxMin.z);
	glEnd();


	glBegin(GL_LINES);
	glVertex3f(lightpos.x, lightpos.y, lightpos.z);
	glVertex3f(lightpos.x + 0.2, lightpos.y, lightpos.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(lightpos.x, lightpos.y, lightpos.z);
	glVertex3f(lightpos.x, lightpos.y + 0.2, lightpos.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(lightpos.x, lightpos.y, lightpos.z);
	glVertex3f(lightpos.x, lightpos.y, lightpos.z + 0.2);
	glEnd();
}

void DrawSkyBox(bool is_normal_draw)
{
	// deprecated. do not add offset, this is the world.
	float offsetx = 0;
	float offsety = world_size + boundingBoxMin.y;
	float offsetz = 0;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (!is_normal_draw) {
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_TEXTURE_2D);
	}
		

	if (is_normal_draw){
		glUniform1i(is_sky_box_location, 1);

		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, background_tex_blur);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// TODO change to lighter

		glActiveTexture(GL_TEXTURE1);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, GetEnvironmentMapTex(false));

		glUniform1i(sky_box_color_tex_location, 0);
		glUniform1i(sky_box_radiance_tex_location, 1);


		glActiveTexture(GL_TEXTURE2);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, GetViewMapTex(false));
		glUniform1i(mesh_radiance_location, 2);
	}


	// bottom
	GLfloat bottomCoordData[] = {
		1.0f / 4 * 1, 1.0f / 3 * 0,
		1.0f / 4 * 2, 1.0f / 3 * 0,
		1.0f / 4 * 2, 1.0f / 3 * 1,
		1.0f / 4 * 1, 1.0f / 3 * 1
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), bottomCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat bottomData[] = {
		//  X     Y     Z 
		-world_size + offsetx, -world_size + offsety, world_size + offsetz,
		world_size + offsetx, -world_size + offsety, world_size + offsetz,
		world_size + offsetx, -world_size + offsety, -world_size + offsetz,
		-world_size + offsetx, -world_size + offsety, -world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), bottomData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);


	// right
	GLfloat rightCoordData[] = {
		1.0f / 4 * 2, 1.0f / 3 * 1,
		1.0f / 4 * 3, 1.0f / 3 * 1,
		1.0f / 4 * 3, 1.0f / 3 * 2,
		1.0f / 4 * 2, 1.0f / 3 * 2
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), rightCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat rightData[] = {
		//  X     Y     Z 
		world_size + offsetx, -world_size + offsety, -world_size + offsetz,
		world_size + offsetx, -world_size + offsety, world_size + offsetz,
		world_size + offsetx, world_size + offsety, world_size + offsetz,
		world_size + offsetx, world_size + offsety, -world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), rightData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);


	// left
	GLfloat leftCoordData[] = {
		1.0f / 4 * 0, 1.0f / 3 * 1,
		1.0f / 4 * 1, 1.0f / 3 * 1,
		1.0f / 4 * 1, 1.0f / 3 * 2,
		1.0f / 4 * 0, 1.0f / 3 * 2
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), leftCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat leftData[] = {
		//  X     Y     Z 
		-world_size + offsetx, -world_size + offsety, world_size + offsetz,
		-world_size + offsetx, -world_size + offsety, -world_size + offsetz,
		-world_size + offsetx, world_size + offsety, -world_size + offsetz,
		-world_size + offsetx, world_size + offsety, world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), leftData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);


	// back
	GLfloat backCoordData[] = {
		1.0f / 4 * 1, 1.0f / 3 * 1,
		1.0f / 4 * 2, 1.0f / 3 * 1,
		1.0f / 4 * 2, 1.0f / 3 * 2,
		1.0f / 4 * 1, 1.0f / 3 * 2
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), backCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat backData[] = {
		//  X     Y     Z  
		-world_size + offsetx, -world_size + offsety, -world_size + offsetz,
		world_size + offsetx, -world_size + offsety, -world_size + offsetz,
		world_size + offsetx, world_size + offsety, -world_size + offsetz,
		-world_size + offsetx, world_size + offsety, -world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), backData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);


	// front
	GLfloat frontCoordData[] = {
		1.0f / 4 * 3, 1.0f / 3 * 1,
		1.0f / 4 * 4, 1.0f / 3 * 1,
		1.0f / 4 * 4, 1.0f / 3 * 2,
		1.0f / 4 * 3, 1.0f / 3 * 2
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), frontCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat frontData[] = {
		//  X     Y     Z  
		world_size + offsetx, -world_size + offsety, world_size + offsetz,
		-world_size + offsetx, -world_size + offsety, world_size + offsetz,
		-world_size + offsetx, world_size + offsety, world_size + offsetz,
		world_size + offsetx, world_size + offsety, world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), frontData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);



	// top
	GLfloat topCoordData[] = {
		1.0f / 4 * 1, 1.0f / 3 * 2,
		1.0f / 4 * 2, 1.0f / 3 * 2,
		1.0f / 4 * 2, 1.0f / 3 * 3,
		1.0f / 4 * 1, 1.0f / 3 * 3
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), topCoordData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_TRUE, 0, NULL);
	glEnableVertexAttribArray(3);

	GLfloat topData[] = {
		//  X     Y     Z  
		-world_size + offsetx, world_size + offsety, -world_size + offsetz,
		world_size + offsetx, world_size + offsety, -world_size + offsetz,
		world_size + offsetx, world_size + offsety, world_size + offsetz,
		-world_size + offsetx, world_size + offsety, world_size + offsetz,
	};
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), topData, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_QUADS, 0, 4);



	if (is_normal_draw)
		glUniform1i(is_sky_box_location, 0);

	glPopAttrib();
}



#ifdef DEBUG_PATH
void DrawPhotonGenPath()
{
	for (int i = 0; i < photons_debug.size(); i++){
		glBegin(GL_LINES);
		if (isPathOriginLight)
			glVertex3f(lightpos.x, lightpos.y, lightpos.z);
		else
			glVertex3f(eye.x, eye.y, eye.z);
		glm::vec3 p = photons_debug.at(i);
		glVertex3f(p.x, p.y, p.z);
		glEnd();
	}
}				   
#endif

#ifdef DEBUG_PATH
void DrawPhotonMarchPath()
{
	if (paths.size() < 4) return;
	int i = 0;
	//cout << paths.size() << " vertices to draw" << endl;
	glm::vec3 tmp, tmp2;
	while (i < paths.size()-1){
		tmp = paths.at(i++);
		tmp2 = paths.at(i);
		if (tmp2.x < -4 || tmp2.y < -4 || tmp2.z < -4){
			i++;
			continue;
		}
		glBegin(GL_LINES);
		glVertex3f(tmp.x, tmp.y, tmp.z);
		glVertex3f(tmp2.x, tmp2.y, tmp2.z);
		glEnd();
	}
}
#endif




/*
 *
 * CLEANUP
 *
 */

void cleanupCuda(){
	if (pbo) deletePBO(&pbo);
	if (displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex){
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void shut_down(int return_code){
	kernelCleanup();
	cudaDeviceReset();
#ifdef __APPLE__
	glfwTerminate();
#endif
	exit(return_code);
}



/*
*
* GLFW CALLBACKS
*
*/

void errorCallback(int error, const char* description){
	fputs(description, stderr);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_W && action == GLFW_PRESS){
		eye.y += 0.1f;
	}
	if (key == GLFW_KEY_S && action == GLFW_PRESS){
		eye.y -= 0.1f;
	}
	if (key == GLFW_KEY_A && action == GLFW_PRESS){
		eye.x += 0.1f;
	}
	if (key == GLFW_KEY_D && action == GLFW_PRESS){
		eye.x -= 0.1f;
	}
	if (key == GLFW_KEY_Q && action == GLFW_PRESS){
		eye.z += 0.1f;
	}
	if (key == GLFW_KEY_E && action == GLFW_PRESS){
		eye.z -= 0.1f;
	}
	if (key == GLFW_KEY_N && action == GLFW_PRESS){
		mode++;
		if (mode > 2)
			mode = 0;
	}
	if (key == GLFW_KEY_M && action == GLFW_PRESS){
		if (barycenter)
			barycenter = false;
		else barycenter = true;
	}
	if (key == GLFW_KEY_L && action == GLFW_PRESS){
		lightRenderToTexture = true;
		drawPhotonGenPath = true;
		isPathOriginLight = true;
	}
	if (key == GLFW_KEY_V && action == GLFW_PRESS){
		VOXELIZE = !VOXELIZE;
		if (VOXELIZE)
			redo_level = RAW_VIEW;
		else
			redo_level = FROM_CAMERA;
	}
	if (key == GLFW_KEY_X && action == GLFW_PRESS){
		drawPhotonGenPath = !drawPhotonGenPath;
	}
	if (key == GLFW_KEY_M && action == GLFW_PRESS){
		drawMesh = !drawMesh;
	}
	if (key == GLFW_KEY_C && action == GLFW_PRESS){
		cameraRenderToTexture = true;
		drawPhotonGenPath = true;
		isPathOriginLight = false;
	}
	if (key == GLFW_KEY_P && action == GLFW_PRESS){
		drawPhotonMarchPath = !drawPhotonMarchPath;
	}
	if (key == GLFW_KEY_I && action == GLFW_PRESS){
		isMoveLight = !isMoveLight;
	}
	if (key == GLFW_KEY_O && action == GLFW_PRESS){
		isDrawOriginal = !isDrawOriginal;
		if (isDrawOriginal)
			redo_level = RAW_VIEW;
		else
			redo_level = FROM_CAMERA;
	}
	if (key == GLFW_KEY_U && action == GLFW_PRESS){
		isMoveSelectedInternal = !isMoveSelectedInternal;
	}
	if (key == GLFW_KEY_Z && action == GLFW_PRESS){
		selectedInternal.push_back(internalSelector - boundingBoxMin + glm::vec3(boundingBoxExpand / 2.0, boundingBoxExpand / 2.0, boundingBoxExpand / 2.0));
		redo_level = FROM_VOXELIZATION;
		cout << "selectedInternal add, global: " << internalSelector.x << " " << internalSelector.y << " " << internalSelector.z<<endl;
		cout << "selectedInternal add, local: " << selectedInternal.back().x << " " << selectedInternal.back().y << " " << selectedInternal.back().z << endl;
	}
}

void CursorEnterCallback(GLFWwindow *window, int entered){
	if (entered == GL_TRUE)
		inwindow = true;
	else
		inwindow = false;
}

void MouseClickCallback(GLFWwindow *window, int button, int action, int mods){
	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT){
		glfwGetCursorPos(window, &MouseX, &MouseY);
		LB = true;
	}

	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT){
		glfwGetCursorPos(window, &MouseX, &MouseY);
		RB = true;
	}

	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_MIDDLE){
		glfwGetCursorPos(window, &MouseX, &MouseY);
		MB = true;
	}

	if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT)
		LB = false;

	if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT)
		RB = false;

	if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_MIDDLE)
		MB = false;

}

void CursorCallback(GLFWwindow *window, double x, double y){
	x = glm::max(0.0, x);
	x = glm::min(x, (double)W_WIDTH);
	y = glm::max(0.0, y);
	y = glm::min(y, (double)W_HEIGHT);

	int changeX = x - MouseX;
	int changeY = y - MouseY;

	//if ((abs(changeX) > 1 || abs(changeY) > 1) && isMoveLight)
		//redo_level = FROM_LIGHTING;

	if (LB&&inwindow){ //camera rotate
		if (isMoveLight){
			lightpos += glm::vec3(0.00005 * changeX * MIDDLE_SPEED, 0, 0);
			redo_level = FROM_LIGHTING;
		}
		else if (isMoveSelectedInternal){
			internalSelector += glm::vec3(0.00005 * changeX * MIDDLE_SPEED, 0, 0);
		}
		else{
			vPhi -= changeX * MOUSE_SPEED;
			vTheta -= changeY * MOUSE_SPEED;
			vTheta = glm::clamp(vTheta, float(1e-6), float(PI - (1e-6)));
		}
	}

	if (RB&&inwindow){ //zoom in and out
		if (isMoveLight){
			lightpos += glm::vec3(0, 0.00005 * changeX * MIDDLE_SPEED, 0);
			redo_level = FROM_LIGHTING;
		}
		else if (isMoveSelectedInternal){
			internalSelector += glm::vec3(0, 0.00005 * changeX * MIDDLE_SPEED, 0);
		}
		else {
			float scale = -changeX / MouseX + changeY / MouseY;
			R = (1.0f + 0.003f * scale * ZOOM_SPEED) * R;
			R = glm::clamp(R, zNear, zFar);
		}
	}

	if (MB&&inwindow)
	{
		if (isMoveLight){
			lightpos += glm::vec3(0, 0, 0.00005 * changeX * MIDDLE_SPEED);
			redo_level = FROM_LIGHTING;
		}
		else if (isMoveSelectedInternal){
			internalSelector += glm::vec3(0, 0, 0.00005 * changeX * MIDDLE_SPEED);
		}
		else{
			eye -= glm::vec3(0.00001 * MIDDLE_SPEED, 0, 0) * (float)changeX;
			eye += glm::vec3(0, 0.00001 * MIDDLE_SPEED, 0) * (float)changeY;
			center -= glm::vec3(0.00001 * MIDDLE_SPEED, 0, 0) * (float)changeX;
			center += glm::vec3(0, 0.00001 * MIDDLE_SPEED, 0) * (float)changeY;
			view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
		}
	}


	if (!isMoveLight){
		eye = glm::vec3(R*sin(vTheta)*sin(vPhi), R*cos(vTheta) + center.y, R*sin(vTheta)*cos(vPhi));
		view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
	}

	if (isMoveSelectedInternal && inwindow && (LB||MB||RB)){
		glm::vec3 ins = (internalSelector - boundingBoxMin + glm::vec3(boundingBoxExpand / 2.0, boundingBoxExpand / 2.0, boundingBoxExpand / 2.0));
		cout << "internalSelector move to, global: " << internalSelector.x << " " << internalSelector.y << " " << internalSelector.z << endl;
		cout << "internalSelector move to, local: " << ins.x << " " << ins.y << " " << ins.z << endl;
	}

}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	R = (1.0f - 0.006f * yoffset * ZOOM_SPEED) * R;
	R = glm::clamp(R, zNear, zFar);
	eye = glm::vec3(R*sin(vTheta)*sin(vPhi), R*cos(vTheta) + center.y, R*sin(vTheta)*cos(vPhi));
	view = glm::lookAt(eye, center, glm::vec3(0, 1, 0));
}



/*
*
* Cuda rasterizer initilization.
*
*/


void initCudaPBO(){
	// set up vertex data parameter
	int num_texels = W_WIDTH*W_HEIGHT;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte)* num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
	// Use device with highest Gflops/s
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initCudaTextures(){
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W_WIDTH, W_HEIGHT, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

void initCudaVAO(void){
	GLfloat vertices[] =
	{
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, 1.0f,
	};

	GLfloat texcoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initPassthroughShaders() {
	const char *attribLocations[] = { "Position", "Tex" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}


void runCuda() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr = NULL;

	glm::mat4 rotationM = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 20.0f - 0.5f*frame, glm::vec3(0.0f, 1.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));

	float newcbo[] = { 0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		1.0, 0.0, 0.0 };

	//Update data
	if (VOXELIZE) {
		vbo = m_vox.vbo;
		vbosize = m_vox.vbosize;
		cbo = newcbo;
		cbosize = 9;
		ibo = m_vox.ibo;
		ibosize = m_vox.ibosize;
		nbo = m_vox.nbo;
		nbosize = m_vox.nbosize;
	}
	else {
		vbo = mesh->getVBO();
		vbosize = mesh->getVBOsize();
		cbo = newcbo;
		cbosize = 9;
		ibo = mesh->getIBO();
		ibosize = mesh->getIBOsize();
		nbo = mesh->getNBO();
		nbosize = mesh->getNBOsize();
	}

	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(W_WIDTH, W_HEIGHT), rotationM, frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, &tex, texcoord, eye, center, view, lightpos, mode, barycenter);
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;

	frame++;
	fpstracker++;

}



/*
 *
 * Tools.
 *
 */


#define BMP_Header_Length 54

// judge whether n is a number of power of 2
int power_of_two(int n) {
	if (n <= 0)
		return 0;

	return (n & (n - 1)) == 0;
}

// open file_name, read in image data
// return a new texture ID
GLuint load_texture(const char* file_name) {
	GLint width, height, total_bytes;
	GLubyte* pixels = 0;
	GLint last_texture_ID;
	GLuint texture_ID = 0;

	// open BMP image file
	FILE* pFile = fopen(file_name, "rb");
	if (pFile == 0){
		cout << "BMP file open failed" << endl;
		return 0;
	}

	// read BMP file width and height
	fseek(pFile, 0x0012, SEEK_SET);
	fread(&width, 4, 1, pFile);
	fread(&height, 4, 1, pFile);
	fseek(pFile, BMP_Header_Length, SEEK_SET);

	// calculate the total bytes of data
	{
		GLint line_bytes = width * 3;
		while (line_bytes % 4 != 0)
			++line_bytes;
		total_bytes = line_bytes * height;
	}
	// allocate memory
	pixels = (GLubyte*)malloc(total_bytes);
	if (pixels == 0)
	{
		fclose(pFile);
		return 0;
	}
	// read image data
	if (fread(pixels, total_bytes, 1, pFile) <= 0) {
		free(pixels);
		fclose(pFile);
		return 0;
	}

	// scale the image if it's not aligned, or too large
	if (0)
	{
		GLint max;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max);
		if (!power_of_two(width) || !power_of_two(height) || width > max || height > max)
		{
			// dimension after scaling
			const GLint new_width = 1024;
			const GLint new_height = 1024;
			GLint new_line_bytes, new_total_bytes;
			GLubyte* new_pixels = 0;

			// calculate the total bytes of one line
			new_line_bytes = new_width * 3;
			while (new_line_bytes % 4 != 0)
				++new_line_bytes;
			new_total_bytes = new_line_bytes * new_height;
			// allocate memory
			new_pixels = (GLubyte*)malloc(new_total_bytes);
			if (new_pixels == 0)
			{
				free(pixels);
				fclose(pFile);
				return 0;
			}

			// scale
			gluScaleImage(GL_RGB, width, height, GL_UNSIGNED_BYTE, pixels,
				new_width, new_height, GL_UNSIGNED_BYTE, new_pixels);

			free(pixels);
			pixels = new_pixels;
			width = new_width;
			height = new_height;
		}
	}

	// allocate a new texture id
	glGenTextures(1, &texture_ID);
	if (texture_ID == 0) {
		free(pixels);
		fclose(pFile);
		cout << "Failed to generate texture" << endl;
		return 0;
	}

	// set the attributes of the texture
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);
	glBindTexture(GL_TEXTURE_2D, texture_ID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);
	glBindTexture(GL_TEXTURE_2D, last_texture_ID);


	free(pixels);

	cout << "Loaded BMP texture of id " << texture_ID << ": " << file_name << endl;
	return texture_ID;
}


void readBMP(const char* filename, bmp_texture &tex)
{
	int i;
	FILE* f = fopen(filename, "rb");
	if (!f){
		cout << "Fail to read BMP texture" << endl;
		return;
	}
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
	glm::vec3 *color_data = new glm::vec3[size / 3];
	for (i = 0; i < size; i += 3){
		color_data[i / 3].r = (int)data[i + 2] / 255.0f;
		color_data[i / 3].g = (int)data[i + 1] / 255.0f;
		color_data[i / 3].b = (int)data[i] / 255.0f;
	}
	delete[]data;
	tex.data = color_data;
	tex.height = height;
	tex.width = width;
}