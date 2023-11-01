import glfw
import OpenGL.GL as gl
import glm
import numpy as np
import argparse
import ctypes
import utils.spherical_harmonics as sh
import utils.opengl_helper as opengl_helper
import utils.zonal_harmonics as zhf
import imgui
from imgui.integrations.glfw import GlfwRenderer
import random

# !! counter clockwise as front face !!
light_vertices = [
	# ############ pantagon ############
	[
		 0, -20,  6,
		 2, -20,  2,
		-2, -20, -1,
		-6, -20,  2,
		-4, -20,  6,
	],
	# ############ pantagon ############
	# ############ cross ############
	[
		 0.2, -16,  16,
		 0.2, -16, -16,
		-0.2, -16, -16,
		-0.2, -16,  16,
	],
	[
		 10, -16,  0.2,
		 10, -16, -0.2,
		-10, -16, -0.2,
		-10, -16,  0.2,
	],
	# ############ cross ############
	# ############ triangle ############
	[
		-10, -14, -10,
		  0, -10, -12,
		 10, -14, -10,
	],
	# ############ triangle ############
]

# light_vertices += [
# 	[
# 		random.randint(-8, 8), -16 - i, random.randint(-8, 8),
# 		random.randint(-8, 8), -16 - i, random.randint(-8, 8),
# 		random.randint(-8, 8), -16 - i, random.randint(-8, 8),
# 	] for i in range(8)
# ]

def load_polygon(v):
	vao = gl.glGenVertexArrays(1)
	gl.glBindVertexArray(vao)

	vbo = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
	gl.glBufferData(gl.GL_ARRAY_BUFFER, len(v) * gl.sizeof(gl.GLfloat), v, gl.GL_STATIC_DRAW)

	gl.glEnableVertexAttribArray(0)
	gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * 3, ctypes.c_void_p(0))

	gl.glBindVertexArray(0)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
	gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

	def render():
		gl.glBindVertexArray(vao)
		gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, len(v) // 3)
		gl.glBindVertexArray(0)

	return render

def vec3_to_vec4(v, w):
	v = v.flatten()
	v = v.reshape((-1, 3))
	v = np.concatenate((v, np.ones((len(v), 1), dtype=np.float32) * w), axis=-1)
	v = v.flatten()
	return v

def float_to_vec4(v, w):
	v = v.flatten()
	v = v.reshape((-1, 1))
	v = np.concatenate((v, np.ones((len(v), 3), dtype=np.float32) * w), axis=-1)
	v = v.flatten()
	return v

def get_light_attribute(light_color, light_edge_num, light_intensity):
	return np.concatenate((np.array(light_color) * np.array(light_intensity)[:, None], np.array(light_edge_num)[:, None]), axis=-1, dtype=np.float32).flatten()

def get_light_matrix(model, light_rotate):
	return np.concatenate([
		np.array(glm.rotate(model, glm.radians(r), glm.vec3(0.0, 0.0, 1.0)), dtype=np.float32).flatten()
		for r in light_rotate])

def main(args):
	global light_vertices

	DIM = args.dim
	if not glfw.init():
		return
	window = glfw.create_window(DIM, DIM, 'PRT Viewer', None, None)
	if not window:
		glfw.terminate()
		return

	glfw.make_context_current(window)
	glfw.swap_interval(0)

	imgui.create_context()
	imgui_impl = GlfwRenderer(window)

	hovering_camera = opengl_helper.HoveringCamera(args)

	glfw.set_mouse_button_callback(window, hovering_camera.get_mouse_button_callback())
	glfw.set_scroll_callback(window, hovering_camera.get_scroll_callback())

	# zonal harmonics factorization
	zhf_compress = np.load(f'output/zhf_compress_{args.max_l}.npz')
	zhf_phi = zhf_compress['phi']
	zhf_theta = zhf_compress['theta']
	zhf_alpha = zhf_compress['alpha']
	zhf_omega = zhf.make_array_np(zhf.spherical_to_cartesian_np(zhf_phi, zhf_theta))

	# set up light buffers
	num_polygon = len(light_vertices)
	num_omega = len(zhf_omega)
	num_alpha = len(zhf_alpha)
	polygon_info = []
	num_polygon_vertex = 0
	for p in range(len(light_vertices)):
		N = len(light_vertices[p]) // 3
		edge = [[
			(i-1) % N + num_polygon_vertex,
			(i+1) % N + num_polygon_vertex,
			p, N] for i in range(N)]
		polygon_info += edge
		num_polygon_vertex += N
	polygon_info = np.array(polygon_info, np.int32).flatten()

	light_rotate = [args.init_rotate if args.sync_rotate else i * args.init_rotate for i in range(num_polygon)]
	light_rotate_speed = [args.init_speed if args.sync_rotate else i * args.init_speed + args.init_speed for i in range(num_polygon)]
	light_color = [glm.vec3(np.random.rand(3)) for _ in range(num_polygon)]
	light_intensity = [5.0] * num_polygon
	light_edge_num = [len(p) // 3 for p in light_vertices]

	u_light_vertices = vec3_to_vec4(np.concatenate([np.array(p, np.float32) for p in light_vertices], axis=0), 1.0)
	u_zhf_omega = vec3_to_vec4(zhf_omega, 0.0)
	# u_zhf_alpha = float_to_vec4(zhf_alpha, 0.0)
	u_zhf_alpha = zhf_alpha

	light_attribute = get_light_attribute(light_color, light_edge_num, light_intensity)
	light_matrix = get_light_matrix(glm.mat4(1.0), light_rotate)

	uniforms = [
		opengl_helper.load_uniform_buffer(0, polygon_info),
		opengl_helper.load_uniform_buffer(1, u_light_vertices),
		opengl_helper.load_uniform_buffer(2, u_zhf_omega),
		opengl_helper.load_uniform_buffer(3, u_zhf_alpha),
		opengl_helper.load_uniform_buffer(4, light_attribute),
		opengl_helper.load_uniform_buffer(5, light_matrix),
	]

	macros = f'''
	#define USE_PRT ({1 if args.prt else 0})
	#define N_AREA_LIGHT {num_polygon}
	#define N_AREA_LIGHT_VERTEX {num_polygon_vertex}
	#define N_ZH_LOBE {num_omega}
	#define N_ALPHA {num_alpha}
	#define MAX_L {args.max_l}
	'''

	print(macros)

	draw_model = opengl_helper.load_model(args.model, args.prt, args.prt_ir, args.max_l)
	draw_skybox = opengl_helper.load_skybox()
	lights = [load_polygon(np.array(p, np.float32)) for p in light_vertices]
	prt_shader = opengl_helper.load_shader('shaders/ash.glsl', macros)
	light_shader = opengl_helper.load_shader('shaders/light.glsl', macros)
	skybox_shader = opengl_helper.load_shader('shaders/skybox.glsl', '')
	env_sh = np.load(f'output/{args.envmap}_{min(args.max_l, 4)}.npy')
	envmap_texture = opengl_helper.load_texture_float(f'data/{args.envmap}.exr')

	gl.glViewport(0, 0, DIM, DIM)
	gl.glEnable(gl.GL_DEPTH_TEST)

	last_time = glfw.get_time()
	imgui_io = imgui.get_io()
	fps_frame_count = 0
	fps_update_time = 0.0
	fps = 0.0

	diffuse_color = glm.vec3(0.2)

	while not glfw.window_should_close(window):
		current_time = glfw.get_time()
		delta_time = current_time - last_time
		last_time = current_time

		fps_frame_count += 1
		fps_update_time += delta_time
		if fps_update_time > 1.0:
			fps = fps_frame_count
			fps_frame_count = 0
			fps_update_time = 0.0

		for i in range(num_polygon):
			light_rotate[i] += delta_time * light_rotate_speed[i]

		gl.glClearColor(0.0, 0.0, 0.0, 1.0)
		gl.glClearDepth(1.0)
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

		if not imgui_io.want_capture_mouse:
			hovering_camera.tick(window)
		model = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))

		light_attribute = get_light_attribute(light_color, light_edge_num, light_intensity)
		light_matrix = get_light_matrix(model, light_rotate)

		light_attribute_ubo = uniforms[4]
		light_matrix_ubo = uniforms[5]

		opengl_helper.update_uniform_buffer(light_attribute_ubo, light_attribute)
		opengl_helper.update_uniform_buffer(light_matrix_ubo, light_matrix)

		view = hovering_camera.view
		proj = hovering_camera.proj
		world_rotation = hovering_camera.world_rotation
		env_rotation = glm.angleAxis(world_rotation, glm.vec3(0.0, 0.0, 1.0))
		env_sh_rotated = sh.rotate(env_sh, env_rotation)

		env_sh_r = env_sh_rotated[..., 2].flatten()
		env_sh_g = env_sh_rotated[..., 1].flatten()
		env_sh_b = env_sh_rotated[..., 0].flatten()

		u_env_sh_r = np.concatenate((env_sh_r, np.zeros(27 - len(env_sh_r), dtype=np.float32)))
		u_env_sh_g = np.concatenate((env_sh_g, np.zeros(27 - len(env_sh_r), dtype=np.float32)))
		u_env_sh_b = np.concatenate((env_sh_b, np.zeros(27 - len(env_sh_r), dtype=np.float32)))

		skybox_view = glm.mat4(glm.mat3(view))
		skybox_model = model * glm.inverse(glm.mat4(env_rotation))
		gl.glDepthMask(gl.GL_FALSE)
		gl.glUseProgram(skybox_shader)
		gl.glUniformMatrix4fv(0, 1, gl.GL_FALSE, glm.value_ptr(skybox_model))
		gl.glUniformMatrix4fv(1, 1, gl.GL_FALSE, glm.value_ptr(skybox_view))
		gl.glUniformMatrix4fv(2, 1, gl.GL_FALSE, glm.value_ptr(proj))
		gl.glActiveTexture(gl.GL_TEXTURE0)
		gl.glBindTexture(gl.GL_TEXTURE_2D, envmap_texture)
		gl.glUniform1f(4, args.brightness)
		gl.glUniform1f(5, args.envmap_intensity)

		draw_skybox()
		gl.glDepthMask(gl.GL_TRUE)

		gl.glUseProgram(prt_shader)
		gl.glUniformMatrix4fv(0, 1, gl.GL_FALSE, glm.value_ptr(model))
		gl.glUniformMatrix4fv(1, 1, gl.GL_FALSE, glm.value_ptr(view))
		gl.glUniformMatrix4fv(2, 1, gl.GL_FALSE, glm.value_ptr(proj))
		gl.glUniform1f(3, args.brightness)
		gl.glUniform1f(4, args.envmap_intensity)
		gl.glUniform3fv(5, 1, glm.value_ptr(diffuse_color))
		gl.glUniformMatrix3fv(6 + 0, 3, gl.GL_FALSE, u_env_sh_r)
		gl.glUniformMatrix3fv(6 + 3, 3, gl.GL_FALSE, u_env_sh_g)
		gl.glUniformMatrix3fv(6 + 6, 3, gl.GL_FALSE, u_env_sh_b)

		gl.glEnable(gl.GL_CULL_FACE)
		draw_model()

		gl.glUseProgram(light_shader)
		gl.glUniformMatrix4fv(1, 1, gl.GL_FALSE, glm.value_ptr(view))
		gl.glUniformMatrix4fv(2, 1, gl.GL_FALSE, glm.value_ptr(proj))
		gl.glUniform1f(4, args.brightness)

		gl.glDisable(gl.GL_CULL_FACE)
		for i in range(num_polygon):
			light_model = glm.rotate(model, glm.radians(light_rotate[i]), glm.vec3(0.0, 0.0, 1.0))
			gl.glUniformMatrix4fv(0, 1, gl.GL_FALSE, glm.value_ptr(light_model))
			u_color = light_color[i] * light_intensity[i]
			gl.glUniform3fv(3, 1, glm.value_ptr(u_color))
			lights[i]()

		imgui.new_frame()

		imgui.begin('PRT Viewer')
		imgui.text(f'FPS: {fps:.2f}')
		_, args.brightness = imgui.input_float('brightness', args.brightness)
		_, args.envmap_intensity = imgui.input_float('envmap intensity', args.envmap_intensity)
		_, diffuse_color = imgui.color_edit3('diffuse color', *diffuse_color)
		diffuse_color = glm.vec3(diffuse_color)
		for i in range(num_polygon):
			imgui.text(f'light {i}')
			_, light_rotate_speed[i] = imgui.input_float(f'speed {i}', light_rotate_speed[i])
			_, light_color[i] = imgui.color_edit3(f'color {i}', *light_color[i])
			_, light_intensity[i] = imgui.input_float(f'intensity {i}', light_intensity[i])
			light_color[i] = glm.vec3(light_color[i])
		imgui.end()

		imgui.render()
		imgui.end_frame()

		imgui_impl.render(imgui.get_draw_data())
		glfw.swap_buffers(window)

		imgui_impl.process_inputs()
		glfw.poll_events()

	glfw.terminate()

if __name__ == '__main__':
	args = argparse.ArgumentParser()

	args.add_argument('--dim', type=int, default=920)

	args.add_argument('--camera_radius', type=float, default=120.0)
	args.add_argument('--camera_fov', type=float, default=30.0)

	args.add_argument('--envmap', type=str, default='studio')
	args.add_argument('--model', type=str, default='mesh')
	args.add_argument('--prt', action='store_true')
	args.add_argument('--prt_ir', action='store_true')
	args.add_argument('--brightness', type=float, default=1.0)
	args.add_argument('--envmap_intensity', type=float, default=0.1)
	args.add_argument('--max_l', type=int, default=2)

	args.add_argument('--init_rotate', type=int, default=30)
	args.add_argument('--init_speed', type=int, default=10)
	args.add_argument('--sync_rotate', action='store_true')

	args.add_argument('--output', type=str, default='output/')

	args = args.parse_args()

	main(args)
