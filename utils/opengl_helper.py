import glfw
import OpenGL.GL as gl
import glm
import open3d as o3d
import numpy as np
import ctypes
import utils.spherical_harmonics as sh
import math
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import trange

def load_skybox():
	vertices = np.array([
		-1, -1, -1,
		 1, -1, -1,
		 1,  1, -1,
		 1,  1, -1,
		-1,  1, -1,
		-1, -1, -1,

		-1, -1,  1,
		 1, -1,  1,
		 1,  1,  1,
		 1,  1,  1,
		-1,  1,  1,
		-1, -1,  1,

		-1,  1,  1,
		-1,  1, -1,
		-1, -1, -1,
		-1, -1, -1,
		-1, -1,  1,
		-1,  1,  1,

		 1,  1,  1,
		 1,  1, -1,
		 1, -1, -1,
		 1, -1, -1,
		 1, -1,  1,
		 1,  1,  1,

		-1, -1, -1,
		 1, -1, -1,
		 1, -1,  1,
		 1, -1,  1,
		-1, -1,  1,
		-1, -1, -1,

		-1,  1, -1,
		 1,  1, -1,
		 1,  1,  1,
		 1,  1,  1,
		-1,  1,  1,
		-1,  1, -1,
	], dtype=np.float32)

	vao = gl.glGenVertexArrays(1)
	gl.glBindVertexArray(vao)

	vbo = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
	gl.glBufferData(gl.GL_ARRAY_BUFFER, len(vertices) * gl.sizeof(gl.GLfloat), vertices, gl.GL_STATIC_DRAW)

	gl.glEnableVertexAttribArray(0)
	gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * 3, ctypes.c_void_p(0))

	gl.glBindVertexArray(0)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
	gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

	def render():
		gl.glBindVertexArray(vao)
		gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(vertices) // 3)
		gl.glBindVertexArray(0)

	return render

class HoveringCamera:
	def __init__(self, args):
		self.left_mouse_button_pressed = False
		self.middle_mouse_button_pressed = False
		self.right_mouse_button_pressed = False
		self.mouse_delta_position = (0, 0)
		self.mouse_current_position = (0, 0)
		self.camera_radius = args.camera_radius
		self.camera_fov = args.camera_fov
		self.rotate_angle_x = 0.0
		self.rotate_angle_y = 0.0
		self.world_rotation = 0.0
		self.center = glm.vec3(0.0, 0.0, 0.0)

	def get_mouse_button_callback(self):
		def mouse_button_callback(window, button, action, mods):
			if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
				self.left_mouse_button_pressed = True
			if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
				self.left_mouse_button_pressed = False
			if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
				self.middle_mouse_button_pressed = True
			if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
				self.middle_mouse_button_pressed = False
			if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
				self.right_mouse_button_pressed = True
			if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
				self.right_mouse_button_pressed = False

		return mouse_button_callback
	
	def get_scroll_callback(self):
		def scroll_callback(window, xoffset, yoffset):
			self.camera_radius *= math.exp(-yoffset * 0.02)

		return scroll_callback
	
	def tick(self, window):
		xpos, ypos = glfw.get_cursor_pos(window)
		self.mouse_delta_position = (xpos - self.mouse_current_position[0], ypos - self.mouse_current_position[1])
		self.mouse_current_position = (xpos, ypos)

		eye = glm.vec3(
				np.sin(self.rotate_angle_x) * np.cos(self.rotate_angle_y) * self.camera_radius,
				np.sin(self.rotate_angle_y) * self.camera_radius,
				np.cos(self.rotate_angle_x) * np.cos(self.rotate_angle_y) * self.camera_radius)
		at = glm.vec3(0.0, 0.0, 0.0)
		up = glm.vec3(0.0, 1.0, 0.0)

		if self.left_mouse_button_pressed:
			self.rotate_angle_x -= self.mouse_delta_position[0] * 0.005
			self.rotate_angle_y += self.mouse_delta_position[1] * 0.005
		elif self.middle_mouse_button_pressed:
			forward = glm.normalize(at - eye)
			right = glm.normalize(glm.cross(forward, up))
			up = glm.normalize(glm.cross(right, forward))
			self.center += (-right * self.mouse_delta_position[0] + up * self.mouse_delta_position[1]) * self.camera_radius * 0.00025
		elif self.right_mouse_button_pressed:
			self.world_rotation -= self.mouse_delta_position[0] * 0.01

		eye += self.center
		at += self.center

		self.eye = eye
		self.at = at
		self.up = up

		self.view = glm.lookAt(eye, at, up)
		self.proj = glm.perspective(glm.radians(self.camera_fov), 1.0, 0.01, 1000.0)

def compile_shader(shader_source, shader_type):
	shader = gl.glCreateShader(shader_type)
	gl.glShaderSource(shader, shader_source)
	gl.glCompileShader(shader)
	compile_status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
	if compile_status != gl.GL_TRUE:
		error_log = gl.glGetShaderInfoLog(shader).decode()
		print(f'Shader compilation failed:\n{error_log}')
		gl.glDeleteShader(shader)
		return None
	return shader

def load_model(model, use_prt=False, prt_ir=False, prt_max_l=2):
	mesh = o3d.io.read_triangle_mesh(f'data/{model}.obj')
	mesh.compute_triangle_normals()
	mesh.compute_vertex_normals()

	f = np.asarray(mesh.triangles, dtype=np.uint32).flatten()

	v = np.asarray(mesh.vertices, dtype=np.float32)
	vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
	# vt = np.asarray(mesh.triangle_uvs, dtype=np.float32)

	v = v[f]
	vn = vn[f]
	v_attribs = [v, vn]
	
	if use_prt:
		assert prt_max_l <= 4
		if prt_ir:
			prt = np.load(f'output/{model}_prt_coeff_{prt_max_l}_ir.npy')
		else:
			prt = np.load(f'output/{model}_prt_coeff_{prt_max_l}.npy')
		prt = prt.astype(np.float32)
		prt = prt[f]
		prt = np.concatenate((prt, np.zeros((prt.shape[0], 27 - prt.shape[1]), dtype=np.float32)), axis=-1)
		v_attribs.append(prt)

	v_attribs = np.concatenate(v_attribs, axis=1).flatten()

	vao = gl.glGenVertexArrays(1)
	gl.glBindVertexArray(vao)

	vbo = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
	gl.glBufferData(gl.GL_ARRAY_BUFFER, len(v_attribs) * gl.sizeof(gl.GLfloat), v_attribs, gl.GL_STATIC_DRAW)

	# max_attribute = gl.glGetInteger(gl.GL_MAX_VERTEX_ATTRIBS)
	# print('max_attribute', max_attribute)

	stride = (6 + 27) if use_prt else 6
	gl.glEnableVertexAttribArray(0)
	gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * stride, ctypes.c_void_p(0))
	gl.glEnableVertexAttribArray(1)
	gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * stride, ctypes.c_void_p(12))
	# gl.glEnableVertexAttribArray(2)
	# gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * stride, ctypes.c_void_p(24))
	if use_prt:
		for i in range(3 * 3):
			gl.glEnableVertexAttribArray(2 + i)
			gl.glVertexAttribPointer(2 + i, 3, gl.GL_FLOAT, gl.GL_FALSE, gl.sizeof(gl.GLfloat) * stride, ctypes.c_void_p(24 + i * 12))

	gl.glBindVertexArray(0)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
	gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

	def render():
		gl.glBindVertexArray(vao)
		gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(f))
		gl.glBindVertexArray(0)

	return render

def load_shader(path, macros):
	with open(path, 'r') as file:
		source = file.read()

	types = [
		('vert', gl.GL_VERTEX_SHADER),
		('frag', gl.GL_FRAGMENT_SHADER),
		('geom', gl.GL_GEOMETRY_SHADER),
		('comp', gl.GL_COMPUTE_SHADER),
		('tess', gl.GL_TESS_CONTROL_SHADER),
		('tese', gl.GL_TESS_EVALUATION_SHADER)
	]
	shaders = []
	for (type, glenum) in types:
		beg = source.find(f'@beg: {type}')
		end = source.find(f'@end: {type}')
		if beg != -1 and end != -1:
			code = source.split(f'@beg: {type}')[1].split(f'@end: {type}')[0]
			code = code.replace('@extern: macros', macros)
			shaders.append(compile_shader(code, glenum))
	
	shader_program = gl.glCreateProgram()
	for shader in shaders:
		gl.glAttachShader(shader_program, shader)
	gl.glLinkProgram(shader_program)

	link_status = gl.glGetProgramiv(shader_program, gl.GL_LINK_STATUS)

	if link_status != gl.GL_TRUE:
		error_log = gl.glGetProgramInfoLog(shader_program).decode()
		print(f'Program linking failed:\n{error_log}')
		gl.glDeleteProgram(shader_program)
		return
	
	return shader_program

def create_framebuffer(args):
	DIM = args.dim
	framebuffer = gl.glGenFramebuffers(1)

	gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)

	depth_renderbuffer = gl.glGenRenderbuffers(1)
	gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_renderbuffer)
	gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, DIM, DIM)
	gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_renderbuffer)

	color_texture = gl.glGenTextures(1)
	gl.glBindTexture(gl.GL_TEXTURE_2D, color_texture)
	gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, DIM, DIM, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
	gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_texture, 0)

	gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

	status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
	if status != gl.GL_FRAMEBUFFER_COMPLETE:
		print('Framebuffer is not complete!')

	gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

	return framebuffer

def load_texture(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
	image = image[::-1,...]
	image_data = image.flatten()

	texture = gl.glGenTextures(1)
	gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

	gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.shape[1], image.shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image_data)
	gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

	return texture

def load_texture_float(path):
	image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
	image = np.asarray(image, dtype=np.float32)
	image = image[::-1,...]
	image_data = image.flatten()

	texture = gl.glGenTextures(1)
	gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
	gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

	gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.shape[1], image.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, image_data)
	gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

	return texture

def load_uniform_buffer(binding, data):
	def get_size(data):
		if data.dtype == np.float32:
			return len(data) * gl.sizeof(gl.GLfloat)
		elif data.dtype == np.int32:
			return len(data) * gl.sizeof(gl.GLint)

	ubo = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, ubo)
	gl.glBufferData(gl.GL_UNIFORM_BUFFER, get_size(data), data, gl.GL_STATIC_DRAW)
	gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, binding, ubo)
	gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)

	return ubo

def update_uniform_buffer(ubo, data):
	if data.dtype == np.float32:
		ptr_type = ctypes.c_float
	elif data.dtype == np.int32:
		ptr_type = ctypes.c_int32

	gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, ubo)
	data_addr = gl.glMapBuffer(gl.GL_UNIFORM_BUFFER, gl.GL_WRITE_ONLY)
	data_ptr = ctypes.cast(data_addr, ctypes.POINTER(ptr_type))
	data_np = np.ctypeslib.as_array(data_ptr, shape=data.shape)
	data_np[:] = data
	gl.glUnmapBuffer(gl.GL_UNIFORM_BUFFER)
	gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)

def main(args):
	if not glfw.init():
		return

	window = glfw.create_window(256, 256, 'OpenGL', None, None)
	if not window:
		glfw.terminate()
		return

	glfw.make_context_current(window)

	DIM = args.dim

	draw_model = load_model(args.model)
	shader = load_shader(args.shader, args.shader_macros)
	framebuffer = create_framebuffer(args)
	texture = load_texture(args.albedo)

	model_loc = 0
	view_loc = 1
	projection_loc = 2
	env_shading_sh_r_loc = 3
	env_shading_sh_g_loc = 4
	env_shading_sh_b_loc = 5
	env_sh_r_loc = 6
	env_sh_g_loc = 7
	env_sh_b_loc = 8
	tex_loc = 9
	eye_loc = 10

	envmap_name = args.envmap
	env_sh = np.load(f'output/cache/envmap_sh/{envmap_name}.npy')

	gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
	gl.glViewport(0, 0, DIM, DIM)
	gl.glEnable(gl.GL_DEPTH_TEST)

	for angle_x in trange(180, 360, 180):
		for angle_z in trange(0, 360, 30):
			rotate_angle_x = math.radians(angle_x)
			rotate_envmap_z = math.radians(angle_z)

			eye = glm.vec3(
					np.sin(rotate_angle_x) * args.camera_radius,
					np.cos(rotate_angle_x) * args.camera_radius,
					0)
			at = glm.vec3(0.0, 0.0, 0.0)
			up = glm.vec3(0.0, 0.0, 1.0)

			gl.glClearColor(0.0, 0.0, 0.0, 0.0)
			gl.glClearDepth(1.0)
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

			gl.glUseProgram(shader)

			model = glm.mat4(1.0)
			view = glm.lookAt(eye, at, up)
			projection = glm.perspective(glm.radians(args.camera_fov), DIM / DIM, 0.1, 100.0)

			env_rotation = glm.angleAxis(rotate_envmap_z, glm.vec3(0.0, 0.0, 1.0))
			env_sh_rotated_r = sh.rotate_single_channel(env_sh[:, 2], env_rotation)
			env_sh_rotated_g = sh.rotate_single_channel(env_sh[:, 1], env_rotation)
			env_sh_rotated_b = sh.rotate_single_channel(env_sh[:, 0], env_rotation)

			env_render_mat_r = sh.get_sh_render_matrix(env_sh_rotated_r)
			env_render_mat_g = sh.get_sh_render_matrix(env_sh_rotated_g)
			env_render_mat_b = sh.get_sh_render_matrix(env_sh_rotated_b)

			env_sh_r = env_sh_rotated_r.reshape((3, 3))
			env_sh_g = env_sh_rotated_g.reshape((3, 3))
			env_sh_b = env_sh_rotated_b.reshape((3, 3))

			gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model))
			gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
			gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
			gl.glUniformMatrix3fv(env_sh_r_loc, 1, gl.GL_FALSE, env_sh_r)
			gl.glUniformMatrix3fv(env_sh_g_loc, 1, gl.GL_FALSE, env_sh_g)
			gl.glUniformMatrix3fv(env_sh_b_loc, 1, gl.GL_FALSE, env_sh_b)
			gl.glUniformMatrix4fv(env_shading_sh_r_loc, 1, gl.GL_FALSE, env_render_mat_r)
			gl.glUniformMatrix4fv(env_shading_sh_g_loc, 1, gl.GL_FALSE, env_render_mat_g)
			gl.glUniformMatrix4fv(env_shading_sh_b_loc, 1, gl.GL_FALSE, env_render_mat_b)
			gl.glUniform1i(tex_loc, 0)
			gl.glUniform3fv(eye_loc, 1, glm.value_ptr(eye))

			gl.glActiveTexture(gl.GL_TEXTURE0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

			draw_model()

			buffer = np.empty((DIM, DIM, 4), dtype=np.float32)
			gl.glReadPixels(0, 0, DIM, DIM, gl.GL_RGBA, gl.GL_FLOAT, buffer)
			buffer = np.flipud(buffer)

			buffer[...,0:3] = buffer[...,2::-1]

			OUT_DIM = args.output_dim
			buffer = cv2.resize(buffer, (OUT_DIM, OUT_DIM), interpolation=cv2.INTER_AREA)

			output_folder = args.output
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			cv2.imwrite(f'{output_folder}/output_{angle_x:03}_{angle_z:03}.exr', buffer)

	glfw.terminate()

if __name__ == '__main__':
	args = argparse.ArgumentParser()

	args.add_argument('--dim', type=int, default=2048)
	args.add_argument('--output_dim', type=int, default=512)

	args.add_argument('--camera_radius', type=float, default=80.0)
	args.add_argument('--camera_fov', type=float, default=20.0)

	args.add_argument('--envmap', type=str, default='sunrise')
	args.add_argument('--model', type=str, default='data/')
	args.add_argument('--albedo', type=str, default='data/albedo.png')
	# args.add_argument('--prt', type=str, default='output/test/prt_coeffs.npy')
	args.add_argument('--prt', type=str, default='')
	
	args.add_argument('--shader', type=str, default='shaders/main.glsl')
	args.add_argument('--shader_macros', type=str, default='#define SHADER_DIFFUSE_COLOR')

	args.add_argument('--output', type=str, default='output/_test_opengl')

	args = args.parse_args()

	main(args)
