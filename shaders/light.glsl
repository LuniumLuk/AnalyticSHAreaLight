@beg: vert
	#version 460 core

@extern: macros

	layout (location = 0) in vec3 a_position;

	layout (location = 0) uniform mat4 u_model;
	layout (location = 1) uniform mat4 u_view;
	layout (location = 2) uniform mat4 u_proj;
	layout (location = 3) uniform vec3 u_color;
	layout (location = 4) uniform float u_brightness;

	layout (location = 0) out vec3 v_color;

	void main() {
		v_color = u_color * u_brightness;
		gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
	}
@end: vert
@beg: frag
	#version 460 core

@extern: macros

	layout (location = 0) in vec3 v_color;

	layout (location = 0) out vec4 fragColor;

	// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
	vec3 aces(vec3 x) {
		const float a = 2.51;
		const float b = 0.03;
		const float c = 2.43;
		const float d = 0.59;
		const float e = 0.14;
		return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
	}

	void main() {
		vec3 color = aces(v_color);
		fragColor = vec4(color, 1.0);
	}
@end: frag