import os
import mitsuba as mi
import drjit as dr
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from tqdm import tqdm
import math
import argparse

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

import utils.spherical_harmonics as spherical_harmonics

mi.set_variant('cuda_ad_rgb')
EPSILON = 1e-4
FLOAT = np.float32

from mitsuba import ScalarTransform4f as T

def spherical_to_cartesian(phi, theta):
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

def sh_project_funcion(func, order=2, dim=64):
    x, y = np.meshgrid(
        np.linspace(0, dim-1, dim, dtype=FLOAT),
        np.linspace(0, dim-1, dim, dtype=FLOAT),
    )
    x = (x + 0.5) / dim
    y = (y + 0.5) / dim

    phi, theta = spherical_harmonics.equirectangular_project(x, y)
    phi = phi.flatten()
    theta = theta.flatten()

    weight = spherical_harmonics.calc_solid_angle(theta, dim, dim)

    val = func(phi, theta, dim**2)

    bn = val.shape[0]
    coeffs = np.zeros((bn, (order + 1) ** 2), dtype=FLOAT)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            i = spherical_harmonics.get_index(l, m)
            sh = spherical_harmonics.eval_sh(l, m, phi, theta)
            coeffs[:,i] += np.sum(sh[None,:] * weight[None,:] * val, axis=-1)
    
    return coeffs

def run(mesh_path, bn=1024, order=2, inter_reflection=False):
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path'},
        'teapot': {
            'type': 'obj',
            'filename': mesh_path,
            'to_world': T.translate([0, 0, 0]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [1, 1, 1]},
            },
        },
    })

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    m_v = np.asarray(mesh.vertices, dtype=FLOAT)
    m_vn = np.asarray(mesh.vertex_normals, dtype=FLOAT)
    m_f = np.asarray(mesh.triangles, dtype=np.int32)
    nv = m_v.shape[0]

    def sh_func(idx, bn):
        def func(phi, theta, sample_count):
            d_x, d_y, d_z = spherical_to_cartesian(phi, theta)

            ray_o = m_v[idx:idx+bn] + m_vn[idx:idx+bn] * EPSILON
            ray_o = np.repeat(ray_o[:,None,:], sample_count, axis=1)
            ray_d = np.concatenate((d_x[:,None], d_y[:,None], d_z[:,None]), axis=1)
            ray_d = np.repeat(ray_d[None,:], bn, axis=0)

            ray_o = np.reshape(ray_o, (-1, 3))
            ray_d = np.reshape(ray_d, (-1, 3))

            mi_ray_o = mi.Point3f(ray_o)
            mi_ray_d = dr.normalize(mi.Vector3f(ray_d))

            ray = mi.Ray3f(o=mi_ray_o, d=mi_ray_d)
            hit = scene.ray_test(ray)

            visibility = 1.0 - np.asarray(hit, dtype=FLOAT)
            n = np.repeat(m_vn[idx:idx+bn][:,None, :], sample_count, axis=1)
            n = np.reshape(n, (-1, 3))
            # prt for diffuse
            h = np.clip(np.sum(ray_d * n, axis=-1), 0.0, 1.0)
            # prt for scatter
            h = np.abs(np.sum(ray_d * n, axis=-1))

            ret = np.reshape(visibility * h, (bn, sample_count))

            return ret

        return func

    def sh_inter_reflection_func(idx, bn, coeffs):
        def func(phi, theta, sample_count):
            order = int(math.sqrt(coeffs.shape[1])) - 1

            d_x, d_y, d_z = spherical_to_cartesian(phi, theta)

            ray_o = m_v[idx:idx+bn] + m_vn[idx:idx+bn] * EPSILON
            ray_o = np.repeat(ray_o[:,None,:], sample_count, axis=1)
            ray_d = np.concatenate((d_x[:,None], d_y[:,None], d_z[:,None]), axis=1)
            ray_d = np.repeat(ray_d[None,:], bn, axis=0)

            ray_o = np.reshape(ray_o, (-1, 3))
            ray_d = np.reshape(ray_d, (-1, 3))

            mi_ray_o = mi.Point3f(ray_o)
            mi_ray_d = dr.normalize(mi.Vector3f(ray_d))

            ray = mi.Ray3f(o=mi_ray_o, d=mi_ray_d)
            si = scene.ray_intersect(ray)

            p = np.asarray(si.p, dtype=FLOAT)

            face = m_f[si.prim_index]
            v0 = m_v[face[:, 0]]
            v1 = m_v[face[:, 1]]
            v2 = m_v[face[:, 2]]

            v01 = v1 - v0
            v02 = v2 - v0
            v0p = p - v0

            dot11 = np.sum(v01 * v01, axis=-1)
            dot12 = np.sum(v01 * v02, axis=-1)
            dot1p = np.sum(v01 * v0p, axis=-1)
            dot22 = np.sum(v02 * v02, axis=-1)
            dot2p = np.sum(v02 * v0p, axis=-1)

            denom = dot11 * dot22 - dot12 * dot12
            denom = np.clip(denom, 1e-6, None)
            u = (dot22 * dot1p - dot12 * dot2p) / denom
            v = (dot11 * dot2p - dot12 * dot1p) / denom

            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            w = 1.0 - u - v

            sh0 = coeffs[face[:, 0]]
            sh1 = coeffs[face[:, 1]]
            sh2 = coeffs[face[:, 2]]

            sh = w[:, None] * sh0 + u[:, None] * sh1 + v[:, None] * sh2
            sh = np.reshape(sh, (bn, sample_count, -1))

            value = np.zeros((bn, sample_count), dtype=FLOAT)
            for l in range(order + 1):
                for m in range(-l, l + 1):
                    value += sh[..., spherical_harmonics.get_index(l, m)] * spherical_harmonics.eval_sh(l, m, 2 * np.pi - phi, theta)[None, ...]

            n = np.repeat(m_vn[idx:idx+bn][:,None, :], sample_count, axis=1)
            n = np.reshape(n, (-1, 3))
            h = np.clip(np.sum(ray_d * n, axis=-1), 0.0, 1.0)
            h = np.reshape(h, (bn, sample_count))

            ret = np.reshape(value * h, (bn, sample_count))

            return ret

        return func

    v_coeffs = []
    for i in tqdm(range(0, nv, bn)):
        coeffs = sh_project_funcion(sh_func(i, bn if i + bn < nv else nv - i), order)
        v_coeffs.append(coeffs)

    v_coeffs = np.concatenate(v_coeffs, axis=0)

    if inter_reflection:
        for i in tqdm(range(0, nv, bn)):
            coeffs = sh_project_funcion(sh_inter_reflection_func(i, bn if i + bn < nv else nv - i, v_coeffs), order)
            v_coeffs[i:i+bn] += coeffs

    return v_coeffs

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--order', type=int, default=2)
    args.add_argument('--mesh', type=str, default='data/mesh.obj')
    args.add_argument('--ir', action='store_true')                  # inter-reflection
    args.add_argument('--bn', type=int, default=1024)               # batch-num
    args.add_argument('--out_dir', type=str, default='output/')
    args = args.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    coeffs = run(args.mesh, bn=args.bn, order=args.order, inter_reflection=args.ir)

    name = os.path.splitext(os.path.basename(args.mesh))[0]

    if args.ir:
        out_file = f'{name}_prt_coeff_{args.order}_ir.npy'
    else:
        out_file = f'{name}_prt_coeff_{args.order}.npy'

    np.save(os.path.join(args.out_dir, out_file), coeffs)