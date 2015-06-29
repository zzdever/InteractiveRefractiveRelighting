#version 330                                                                        
                                                                                    
layout (location = 0) in vec3 v_position;                                             
layout (location = 1) in vec3 v_normal;   
layout (location = 2) in vec3 v_color;
layout (location = 3) in vec2 v_texcoord;                                             


uniform mat4 u_mvpMatrix;
uniform mat4 u_modelMatrix;
uniform mat4 u_projMatrix;
uniform mat3 u_normMatrix;
uniform bool is_swatch_vs;
uniform vec3 bounding_min;
uniform vec3 bounding_max;



//attribute vec3 v_position;
//attribute vec3 v_normal;
//attribute vec3 v_color;

out vec3 fs_position;
out vec3 fs_normal;
out vec3 fs_color;
out vec3 fs_meshpos;
out vec2 fs_texcoord;
out vec3 fs_bounding_min;
out vec3 fs_bounding_max;


void main (void){
	

	if(is_swatch_vs){
		fs_position = v_position;
		fs_normal = v_normal;
		gl_Position = vec4(v_position, 1.0);
	}
	else {
		fs_position = vec3(u_mvpMatrix * vec4(v_position, 1.0));
		fs_normal = u_normMatrix * v_normal;
		fs_color = v_color;
		gl_Position = u_mvpMatrix * vec4(v_position, 1.0);
		fs_meshpos = (v_position - bounding_min) / (bounding_max - bounding_min);
	}

	fs_texcoord = v_texcoord;
	fs_bounding_min = bounding_min;
	fs_bounding_max = bounding_max;


	//vec2 res = textureQueryLod(texUnit, VertexIn.texCoord.xy);
 
		
}