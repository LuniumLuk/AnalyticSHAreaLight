@beg: vert
	#version 460 core

@extern: macros

	// these macros will be added in python
	// #define USE_PRT
	// #define N_AREA_LIGHT
	// #define N_AREA_LIGHT_VERTEX
	// #define N_ZH_LOBE
	// #define N_ALPHA
	// #define MAX_L

	layout (location = 0) in vec3 a_position;
	layout (location = 1) in vec3 a_normal;
	// layout (location = 2) in vec2 a_uv;
	// for support up to order 8 SH, we need 81 SH coefficients
	// therefore, we have to pass these coefficient by 9 mat3

	// support up to order 4 SH for precomputed radiance transfer per-vertex
	#if USE_PRT
	layout (location = 2) in mat3 a_prt[3];
	#endif

	layout (location = 0) uniform mat4 u_model;
	layout (location = 1) uniform mat4 u_view;
	layout (location = 2) uniform mat4 u_proj;
	layout (location = 3) uniform float u_brightness;
	layout (location = 4) uniform float u_envmap_intensity;
	layout (location = 5) uniform vec3 diffuse_color;

	// support up to order 4 SH for environment
	layout (location = 6 + 0) uniform mat3 u_env_sh_r[3];
	layout (location = 6 + 3) uniform mat3 u_env_sh_g[3];
	layout (location = 6 + 6) uniform mat3 u_env_sh_b[3];

	#define W N_ZH_LOBE
	#define E N_AREA_LIGHT_VERTEX
	#define PI 3.141592653589793

	layout (std140, binding = 0) uniform LightIndexBlock {
		ivec4 u_light_index[N_AREA_LIGHT_VERTEX];
	};

	layout (std140, binding = 1) uniform LightPositionBlock {
		vec4 u_light_position[N_AREA_LIGHT_VERTEX];
	};

	layout (std140, binding = 2) uniform OmegaBlock {
		vec4 u_omega[N_ZH_LOBE];
	};

	layout (std140, binding = 3) uniform AlphaBlock {
		vec4 u_alpha[N_ALPHA / 4];
	};

	layout (std140, binding = 4) uniform LightAttributeBlock {
		vec4 u_light_attribute[N_AREA_LIGHT];
	};

	layout (std140, binding = 5) uniform LightMatrixBlock {
		mat4 u_light_matrix[N_AREA_LIGHT];
	};

	layout (location = 0) out vec3 v_color;

	// calculate solid angle and scaled by the color of each light
	vec3 solid_angle(vec3 omega[E]) {
		float sum[N_AREA_LIGHT];
		for (int i = 0; i < N_AREA_LIGHT; i++) {
			sum[i] = 0.0;
		}

		for (int e = 0; e < E; e++) {
			int e_p = u_light_index[e][1];
			int e_n = u_light_index[e][0];
			vec3 a = cross(omega[e_p], omega[e]);
			vec3 b = cross(omega[e], omega[e_n]);
			sum[u_light_index[e][2]] += acos(-dot(a, b) / (length(a) * length(b)));
		}

		vec3 color = vec3(0.0);
		for (int i = 0; i < N_AREA_LIGHT; i++) {
			sum[i] = max(sum[i] - (int(u_light_attribute[i].w) - 2) * PI, 0.0);
			color += sum[i] * u_light_attribute[i].rgb;
		}

		return color;
	}

	#define FACTORIAL_CACHE_SIZE 12

	float factorial(int x) {
		const float factorial_cache[FACTORIAL_CACHE_SIZE] = {
			1, 1, 2, 6, 24, 120, 720, 5040,
			40320, 362880, 3628800, 39916800
		};

		if (x < FACTORIAL_CACHE_SIZE) {
			return factorial_cache[x];
		}
		else {
			float s = factorial_cache[FACTORIAL_CACHE_SIZE - 1];
			for (int n = FACTORIAL_CACHE_SIZE; n <= x; n++) {
				s *= n;
			}
			return s;
		}
	}

	float double_factorial(int x) {
		const float dbl_factorial_cache[FACTORIAL_CACHE_SIZE] = {
			1, 1, 2, 3, 8, 15, 48, 105,
			384, 945, 3840, 10395
		};

		if (x < FACTORIAL_CACHE_SIZE) {
			return dbl_factorial_cache[x];
		}
		else {
			float s = dbl_factorial_cache[FACTORIAL_CACHE_SIZE - (2 - (x % 2))];
			float n = x;
			while (n >= FACTORIAL_CACHE_SIZE) {
				s *= n;
				n -= 2.0;
			}
			return s;
		}
	}

	// account for the cosine term in integrating the SH for lambertian surface
	float A(int l) {
		if (l == 0) {
			return 3.141592653589793; // pi
		}
		else if (l == 1) {
			return 2.0943951023931953; // pi * 2 / 3
		}
		else if (l == 2) {
			return 0.7853981633974483;
		}
		else if (l == 4) {
			return -0.1308996938995747;
		}
		else if (l == 6) {
			return 0.04908738521234052;
		}
		else if (l % 2 == 0) {
			return 2 * PI * (pow(-1, l / 2 - 1) / ((l + 2) * (l - 1)))
				* (factorial(l) / (pow(2, l) * pow(factorial(l / 2), 2)));
		}
		return 0;
	}

	// evaluate zonal harmonics
	float P(int l, float z) {
		if (l == 0) {
			return 0.282095;
		}
		else if (l == 1) {
			return 0.488603 * z;
		}
		else if (l == 2) {
			return 0.315392 * (3 * z * z - 1);
		}
		else if (l == 3) {
			return 0.373176 * z * (5 * z * z - 1);
		}
		else if (l == 4) {
			float z2 = z * z;
			return 0.105786 * (35 * z2 * z2 - 30 * z2 + 3);
		}
		return 0;
	}

	// reference: https://github.com/google/spherical-harmonics
	float legendre_polynomial(int l, int m, float x) {
		float pmm = 1;
		if (m > 0) {
			pmm = sign(0.5 - m % 2) * double_factorial(2 * m - 1) * pow(1 - x * x, m / 2.0);
		}

		if (l == m) {
			return pmm;
		}

		float pmm1 = x * (2 * m + 1) * pmm;
		if (l == m + 1) {
			return pmm1;
		}

		for (int n = m + 2; n <= l; n++) {
			float pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m);
			pmm = pmm1;
			pmm1 = pmn;
		}
		return pmm1;
	}

	// evaluate spherical harmonics
	float sh(int l, int m, vec3 v) {
		float x = v.x;
		float y = v.y;
		float z = v.z;

		float phi = atan(y, x);

		if (l == 0) {
			return 0.282095;
		}
		else if(l == 1) {
			if (m == -1) {
				return -0.488603 * y;
			}
			else if (m == 0) {
				return 0.488603 * z;
			}
			else if (m == 1) {
				return -0.488603 * x;
			}
		}
		else if (l == 2) {
			if (m == -2) {
				return 1.092548 * x * y;
			}
			else if (m == -1) {
				return -1.092548 * y * z;
			}
			else if (m == 0) {
				// in the original paper, this is 3z^2 - 1, which assumes that
				// the input cartesian coordinates are normalized
				return 0.315392 * (-x * x - y * y + 2 * z * z);
			}
			else if (m == 1) {
				return -1.092548 * x * z;
			}
			else if (m == 2) {
				return 0.546274 * (x * x - y * y);
			}
		}

		float kml = sqrt((2 * l + 1) * factorial(l - abs(m)) / (4 * PI * factorial(l + abs(m))));
		const float sqrt2 = 1.4142135623730951;

		if (m > 0) {
			return sqrt2 * kml * cos(m * phi) * legendre_polynomial(l, m, z);
		}
		else if (m < 0) {
			return sqrt2 * kml * sin(-m * phi) * legendre_polynomial(l, -m, z);
		}
		else {
			return kml * legendre_polynomial(l, 0, z);
		}
	}

	int get_index(int l, int m) {
		return l * (l + 1) + m;
	}

	#if USE_PRT
	vec3 calc_sh_light(mat3 sh_r, mat3 sh_g, mat3 sh_b) {
		float r = dot(sh_r[0], a_prt[0][0]) 
				+ dot(sh_r[1], a_prt[0][1])
				+ dot(sh_r[2], a_prt[0][2]);
		float g = dot(sh_g[0], a_prt[0][0])
				+ dot(sh_g[1], a_prt[0][1])
				+ dot(sh_g[2], a_prt[0][2]);
		float b = dot(sh_b[0], a_prt[0][0])
				+ dot(sh_b[1], a_prt[0][1])
				+ dot(sh_b[2], a_prt[0][2]);

		return vec3(r, g, b);
	}
	#endif

	void main() {
		vec3 x = (u_model * vec4(a_position, 1.0)).xyz;
		// per-vertex attribute for polygons
		vec3 omega[E];
		vec3 lambda[E];
		vec3 mu[E];
		float gamma[E];

		for (int e = 0; e < E; e++) {
			// transpose light matrix here because this matrix is passed in
			// as a row-major matrix with numpy
			omega[e] = normalize((transpose(u_light_matrix[u_light_index[e][2]]) * u_light_position[e]).xyz - x);
		}

		for (int e = 0; e < E; e++) {
			int e_n = u_light_index[e][1];
			lambda[e] = cross(normalize(cross(omega[e], omega[e_n])), omega[e]);
			mu[e] = cross(omega[e], lambda[e]);
			gamma[e] = acos(dot(omega[e], omega[e_n]));
		}

		vec3 S[MAX_L + 1], B[MAX_L + 1], L[MAX_L + 1][W];

		S[0] = solid_angle(omega);

		for (int w = 0; w < W; w++) {
			S[1] = vec3(0.0);
			B[0] = vec3(0.0);
			B[1] = vec3(0.0);
			float a[E], b[E], c[E];
			float _B[MAX_L + 1][E];
			float _C[MAX_L + 1][E];
			float _D[MAX_L + 1][E];
			for (int e = 0; e < E; e++) {
				vec3 lobe = normalize(u_omega[w].xyz);
				/*
				* adjust coordinate system of the SH to match opengl
				*/
				lobe = vec3(-lobe.x, lobe.z, lobe.y);
				a[e] = dot(lobe, omega[e]);
				b[e] = dot(lobe, lambda[e]);
				c[e] = dot(lobe, mu[e]);

				S[1] += (0.5 * c[e] * gamma[e]) * u_light_attribute[u_light_index[e][2]].rgb;

				_B[0][e] = gamma[e];

				_B[1][e] = a[e] * sin(gamma[e]) - b[e] * cos(gamma[e]) + b[e];

				_D[0][e] = 0;
				_D[1][e] = gamma[e];
				_D[2][e] = 3 * _B[1][e];

				B[0] += c[e] * _B[0][e] * u_light_attribute[u_light_index[e][2]].rgb;
				B[1] += c[e] * _B[1][e] * u_light_attribute[u_light_index[e][2]].rgb;
			}

			L[0][w] = sqrt(float(2 * 0 + 1) / (4 * PI)) * S[0];
			L[1][w] = sqrt(float(2 * 1 + 1) / (4 * PI)) * S[1];

			for (int l = 2; l <= MAX_L; l++) {
				B[l] = vec3(0.0);
				for (int e = 0; e < E; e++) {
					_C[l - 1][e] = 1.0 / l * (
						(a[e] * sin(gamma[e]) - b[e] * cos(gamma[e])) * P(l - 1, a[e] * cos(gamma[e]) + b[e] * sin(gamma[e]))
						+ b[e] * P(l - 1, a[e])
						+ (a[e] * a[e] + b[e] * b[e] - 1) * _D[l - 1][e]
						+ (l - 1) * _B[l - 2][e]);

					_B[l][e] = float(2 * l - 1) / l * _C[l - 1][e] - (l - 1) * _B[l - 2][e];

					B[l] += c[e] * _B[l][e] * u_light_attribute[u_light_index[e][2]].rgb;

					_D[l][e] = (2 * l - 1) * _B[l - 1][e] + _D[l - 2][e];
				}
				S[l] = float(2 * l - 1) / (l * (l + 1)) * B[l - 1]
					+ float((l - 2) * (l - 1)) / (l * (l + 1)) * S[l - 2];

				L[l][w] = sqrt(float(2 * 2 + 1) / (4 * PI)) * S[l];
			}
		}

		vec3 L_light = vec3(0.0);
		vec3 L_envmap = vec3(0.0);

	#if USE_PRT
		mat3 L_lm_r[3] = { mat3(0.0), mat3(0.0), mat3(0.0) };
		mat3 L_lm_g[3] = { mat3(0.0), mat3(0.0), mat3(0.0) };
		mat3 L_lm_b[3] = { mat3(0.0), mat3(0.0), mat3(0.0) };
		for (int l = 0; l <= 0; l++) {
			for (int m = -l; m <= l; m++) {
				const int idx = get_index(l, m);
				const int n = l * 2 + 1;
				const int a_offset = l * (4 * l * l - 1) / 3 + (m + l) * n;

				for (int i = 0; i < n; i++) {
					L_lm_r[idx / 9][(idx % 9) / 3][(idx % 9) % 3] += u_alpha[(a_offset + i) / 4][(a_offset + i) % 4] * L[l][i].r;
					L_lm_g[idx / 9][(idx % 9) / 3][(idx % 9) % 3] += u_alpha[(a_offset + i) / 4][(a_offset + i) % 4] * L[l][i].g;
					L_lm_b[idx / 9][(idx % 9) / 3][(idx % 9) % 3] += u_alpha[(a_offset + i) / 4][(a_offset + i) % 4] * L[l][i].b;
				}
			}
		}

		for (int i = 0; i < 3; i++) {
			L_light += calc_sh_light(L_lm_r[0], L_lm_g[0], L_lm_b[0]);
			L_envmap += calc_sh_light(u_env_sh_r[0], u_env_sh_g[0], u_env_sh_b[0]) * u_envmap_intensity;
		}
	#else
		vec3 normal = normalize(transpose(inverse(mat3(u_model))) * a_normal);

		const int N_COEFF = (MAX_L + 1) * (MAX_L + 1);
		vec3 L_lm[N_COEFF];

		for (int i = 0; i < N_COEFF; i++) {
			L_lm[i] = vec3(0.0);
		}

		for (int l = 0; l <= MAX_L; l++) {
			for (int m = -l; m <= l; m++) {
				const int idx = get_index(l, m);
				const int n = l * 2 + 1;
				const int a_offset = l * (4 * l * l - 1) / 3 + (m + l) * n;

				for (int i = 0; i < n; i++) {
					L_lm[idx] += u_alpha[a_offset + i][0] * L[l][i];
				}
			}
		}

		for (int l = 0; l <= MAX_L; l++) {
			for (int m = -l; m <= l; m++) {
				const int idx = get_index(l, m);
				L_light += A(l) * L_lm[idx] * sh(l, m, normal);
			}
		}
	#endif

		L_light = max(L_light, vec3(0.0));
		L_envmap = max(L_envmap, vec3(0.0));
		
		v_color = diffuse_color * (L_light + L_envmap) * u_brightness;

		gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
	}
@end: vert
@beg: frag
	#version 460 core

@extern: macros

	layout (location = 0) in vec3 v_color;

	layout (location = 0) out vec4 fragColor;

	// Markowicz 2015, "ACES Filmic Tone Mapping Curve"
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