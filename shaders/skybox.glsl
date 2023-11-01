@beg: vert
	#version 460 core

@extern: macros

	layout (location = 0) in vec3 a_position;

	layout (location = 0) uniform mat4 u_model;
	layout (location = 1) uniform mat4 u_view;
	layout (location = 2) uniform mat4 u_proj;

	layout (location = 0) out vec3 v_pos;

	void main() {
		v_pos = a_position;
		gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
	}
@end: vert
@beg: frag
	#version 460 core

@extern: macros

	layout (location = 0) in vec3 v_pos;

	layout (location = 0) out vec4 fragColor;

	layout (location = 3, binding = 0) uniform sampler2D u_envmap;
	layout (location = 4) uniform float u_brightness;
	layout (location = 5) uniform float u_envmap_intensity;

	#define PI 3.14159265358979

	// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
	vec3 aces(vec3 x) {
		const float a = 2.51;
		const float b = 0.03;
		const float c = 2.43;
		const float d = 0.59;
		const float e = 0.14;
		return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
	}

	vec2 car2sph(vec3 dir) {
		float r = length(dir);
		float phi = atan(dir.y, dir.x);
		float theta = acos(dir.z / r);
		phi += (1.0 - sign(phi)) * PI;
		return vec2(phi, theta);
	}

	vec2 sph2uv(vec2 sph) {
		return vec2(1.0 - sph.x / (2 * PI), sph.y / PI);
	}

	void main() {
		vec3 dir = normalize(v_pos);
		vec2 sph = car2sph(dir);
		vec2 uv = sph2uv(sph);
		uv.y = 1.0 - uv.y;
		vec3 color = texture(u_envmap, uv).rgb * u_brightness * u_envmap_intensity;
		color = aces(color);
		fragColor = vec4(color, 1.0);
		// fragColor = vec4(uv.x, 0.0, 0.0, 1.0);
	}
@end: frag