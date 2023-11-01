import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import math
from scipy.special import factorial, lpmv
import glm
import sys
import argparse
import cv2

'''
Spherical harmonics
Reference:
[1] "An efficient representation for irradiance environment maps" by Ravi Ramamoorthi, Pat Hanrahan, SIGGRAPH 2001
[2] "Sparse Zonal Harmonic Factorization for Efficient SH Rotation" by Derek Nowrouzezahrai et al., SIGGRAPH 2012
[3] "spherical-harmonics" https://github.com/google/spherical-harmonics
[4] "SphericalHarmonics" https://github.com/chalmersgit/SphericalHarmonics
we replaced the hand written Legendre polynomial evaluation with scipy.special.lpmv
and added Al(l) evaluation for rendering
'''

# reference: [1]
def get_sh_render_matrix(L):
    assert len(L) == 9

    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708

    return np.array([
        [c1 * L[8],  c1 * L[4], c1 * L[7], c2 * L[3]],
        [c1 * L[4], -c1 * L[8], c1 * L[5], c2 * L[1]],
        [c1 * L[7],  c1 * L[5], c3 * L[6], c2 * L[2]],
        [c2 * L[3],  c2 * L[1], c2 * L[2], c4 * L[0] - c5 * L[6]],
    ]) / np.pi

def eval_Al(l : int):
    if l == 0:
        return 3.141592653589793 # pi
    elif l == 1:
        return 2.0943951023931953 # pi * 2 / 3
    elif l == 2:
        return 0.7853981633974483
    elif l == 4:
        return -0.1308996938995747
    elif l == 6:
        return 0.04908738521234052
    elif l % 2 == 0:
        return 2 * math.pi * (math.pow(-1, l // 2 - 1) / ((l + 2) * (l - 1))) * \
               (factorial(l) / (math.pow(2, l) * math.pow(factorial(l // 2), 2)))
    return 0

# reference:
# https://patapom.com/blog/SHPortal/

# Evaluate an Associated Legendre Polynomial P(l,m,x) at x
# For more, see “Numerical Methods in C: The Art of Scientific Computing”, Cambridge University Press, 1992, pp 252-254
# when m < 0, the evaluated value is different from scipy.special.lpmv, but the difference is negligible since we only use m >= 0
def ref_P(l : int, m : int, x : float):
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll

def car2sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return phi, theta

def sph2cart(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def equirectangular_project(x, y):
    return (1.0 - x) * (2.0 * math.pi), y * math.pi

def calc_solid_angle(theta, w, h):
    pixel_size_x = 2.0 * math.pi / w
    pixel_size_y = math.pi / h

    # sin(theta) * d(theta) * d(phi) -> -d(cos(theta)) * d(phi)
    return pixel_size_x * abs(np.cos(theta - (pixel_size_y / 2.0)) - np.cos(theta + (pixel_size_y / 2.0)))

def get_index(l, m):
    return l * (l + 1) + m

def eval_sh(l, m, phi, theta):
    assert l >= 0
    assert -l <= m and m <= l

    x, y, z = sph2cart(phi, theta)
    if l == 0:
        return 0.282095 * np.ones_like(x)
    elif l == 1:
        if m == -1:
            return -0.488603 * y
        elif m == 0:
            return 0.488603 * z
        elif m == 1:
            return -0.488603 * x
    elif l == 2:
        if m == -2:
            return 1.092548 * x * y
        elif m == -1:
            return -1.092548 * y * z
        elif m == 0:
            # in the original paper, this is 3z^2 - 1, which assumes that
            # the input cartesian coordinates are normalized
            return 0.315392 * (-x * x - y * y + 2 * z * z)
        elif m == 1:
            return -1.092548 * x * z
        elif m == 2:
            return 0.546274 * (x * x - y * y)

    kml = math.sqrt(
        (2.0 * l + 1) * factorial(l - abs(m)) /
        (4.0 * math.pi * factorial(l + abs(m)))
    )

    if m > 0:
        return math.sqrt(2.0) * kml * np.cos(m * phi) * lpmv(m, l, np.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kml * np.sin(-m * phi) * lpmv(-m, l, np.cos(theta))
    else:
        return kml * lpmv(0, l, np.cos(theta))

def project_envmap(envmap, order=2, double_precision=True):
    assert order >= 0

    FLOAT = np.float64 if double_precision else np.float32

    h, w, c = envmap.shape
    envmap = envmap.astype(FLOAT)

    x, y = np.meshgrid(
        np.linspace(0, w-1, w, dtype=FLOAT),
        np.linspace(0, h-1, h, dtype=FLOAT),
    )
    x = (x + 0.5) / w
    y = (y + 0.5) / h

    phi, theta = equirectangular_project(x, y)

    weight = calc_solid_angle(theta, w, h)

    buffer = np.zeros((h, w, c, (order + 1) ** 2), dtype=FLOAT)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            i = get_index(l, m)
            sh = eval_sh(l, m, phi, theta)
            buffer[..., i] += sh[..., None] * weight[..., None] * envmap

    coeffs = np.sum(buffer, axis=(0, 1))
    return coeffs.T

# reference: [3]
def near_by_margin(actual : float, expected : float):
    diff = abs(actual - expected)
    # 5 bits of error in mantissa (source of '32 *')
    return diff < 32 * sys.float_info.epsilon

def kronecker_delta(i : int, j : int):
    return 1.0 if i == j else 0.0

def get_centered_element(r : np.ndarray, i : int, j : int):
    offset = int((r.shape[0] - 1) / 2)
    return r[i + offset, j + offset]

def P(i : int, a : int, b : int, l : int, r : list[np.ndarray]):
    if b == l:
        return get_centered_element(r[1], i, 1) *          \
               get_centered_element(r[l - 1], a, l - 1) -  \
               get_centered_element(r[1], i, -1) *         \
               get_centered_element(r[l - 1], a, -l + 1)
    elif b == -l:
        return get_centered_element(r[1], i, 1) *          \
               get_centered_element(r[l - 1], a, -l + 1) + \
               get_centered_element(r[1], i, -1) *         \
               get_centered_element(r[l - 1], a, l - 1)
    else:
        return get_centered_element(r[1], i, 0) * get_centered_element(r[l - 1], a, b)

def U(m : int, n : int, l : int, r : list[np.ndarray]):
    return P(0, m, n, l, r)

def V(m : int, n : int, l : int, r : list[np.ndarray]):
    if m == 0:
        return P(1, 1, n, l, r) + P(-1, -1, n, l, r)
    elif m > 0:
        return P(1, m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, 1)) - \
            P(-1, -m + 1, n, l, r) * (1 - kronecker_delta(m, 1))
    else:
        return P(1, m + 1, n, l, r) * (1 - kronecker_delta(m, -1)) + \
            P(-1, -m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, -1))

def W(m : int, n : int, l : int, r : list[np.ndarray]):
    if (m == 0):
        return 0.0
    elif m > 0:
        return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r)
    else:
        return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r)

def cmpute_uvw_coeff(m : int, n : int, l : int):
    d = 1.0 if m == 0 else 0.0
    denom = 2.0 * l * (2.0 * l - 1) if abs(n) == l else (l + n) * (l - n)

    u = math.sqrt((l + m) * (l - m) / denom)
    v = 0.5 * math.sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom) * (1 - 2 * d)
    w = -0.5 * math.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d)

    return u, v, w

def calculate_band_rotation(l : int, band_rotation : list[np.ndarray]):
    assert len(band_rotation) == l

    r = np.identity(2 * l + 1)

    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            u, v, w = cmpute_uvw_coeff(m, n, l)

            if not near_by_margin(u, 0.0):
                u *= U(m, n, l, band_rotation)
            if not near_by_margin(v, 0.0):
                v *= V(m, n, l, band_rotation)
            if not near_by_margin(w, 0.0):
                w *= W(m, n, l, band_rotation)

            r[m + l, n + l] = u + v + w

    return r

def rotate_single_channel(coeffs : np.ndarray, rotation : glm.quat):
    assert coeffs.ndim == 1

    order = int(math.sqrt(coeffs.shape[0])) - 1

    band_rotations = []

    # order 0 (first band) is simply the 1x1 identity matrix
    r = np.identity(1)
    band_rotations.append(r)

    mat = glm.mat3_cast(rotation)
    r = np.identity(3)
    r[0, 0] =  mat[1, 1]
    r[0, 1] = -mat[1, 2]
    r[0, 2] =  mat[1, 0]
    r[1, 0] = -mat[2, 1]
    r[1, 1] =  mat[2, 2]
    r[1, 2] = -mat[2, 0]
    r[2, 0] =  mat[0, 1]
    r[2, 1] = -mat[0, 2]
    r[2, 2] =  mat[0, 0]
    band_rotations.append(r)

    for l in range(2, order + 1):
        r = calculate_band_rotation(l, band_rotations)
        band_rotations.append(r)

    # apply rotations
    rotated_coeffs = np.zeros_like(coeffs)
    for l in range(order + 1):
        band_coeffs = np.zeros((2 * l + 1))

        for m in range(-l, l + 1):
            band_coeffs[m + l] = coeffs[get_index(l, m)]

        band_coeffs = np.matmul(band_rotations[l], band_coeffs)

        for m in range(-l, l + 1):
            rotated_coeffs[get_index(l, m)] = band_coeffs[m + l]
    
    return rotated_coeffs

def rotate(coeffs : np.ndarray, rotation : glm.quat):
    assert coeffs.ndim == 2

    num_channels = coeffs.shape[1]

    rotated_coeffs = np.zeros_like(coeffs)
    for c in range(num_channels):
        rotated_coeffs[:, c] = rotate_single_channel(coeffs[:, c], rotation)

    return rotated_coeffs

if __name__ == '__main__':

    # about the coordinates in SH
    # 1. spherical coordinates to cartesion follows the convention of below:
    #      x = sin(theta) * cos(phi)
    #      y = sin(theta) * sin(phi)
    #      z = cos(theta)
    #    that is:
    #      +x -> theta = pi / 2, phi = 0
    #      +y -> theta = pi / 2, phi = pi / 2
    #      +z -> theta = 0, phi = (0, 2 * pi)

    # 2. the SH coordinate system follows the above convention, and we can
    #    check the coordinate by evaluate the three SH basis of order 1:
    # ## find +x
    # print(sph2cart(0.0, math.pi / 2))
    # print(eval_sh(1, -1, 0.0, math.pi / 2))
    # print(eval_sh(1,  0, 0.0, math.pi / 2))
    # print(eval_sh(1,  1, 0.0, math.pi / 2)) # = -0.488603

    # ## find +y
    # print(sph2cart(math.pi / 2, math.pi / 2))
    # print(eval_sh(1, -1, math.pi / 2, math.pi / 2)) # = -0.488603
    # print(eval_sh(1,  0, math.pi / 2, math.pi / 2))
    # print(eval_sh(1,  1, math.pi / 2, math.pi / 2))

    # ## find +z
    # print(sph2cart(0.0, 0.0))
    # print(eval_sh(1, -1, 0.0, 0.0))
    # print(eval_sh(1,  0, 0.0, 0.0)) # = 0.488603
    # print(eval_sh(1,  1, 0.0, 0.0))
    #    and we can find the relation below:
    #        +x -> -Y(1,  1)
    #        +y -> -Y(1, -1)
    #        +z -> +Y(1,  0)

    # 3. by experiment in OpenGL, we find the following relation:
    #    +x -> +Y(1,  1)
    #    +y -> -Y(1, -1)
    #    +z -> +Y(1,  0)

    # 4. conclusion, the mapping relation between SH and OpenGL is:
    #    SH -> OpenGL
    #    +x -> -x
    #    +y -> +y
    #    +z -> +z

    args = argparse.ArgumentParser()
    args.add_argument('--order', type=int, default=2)
    args.add_argument('--envmap', type=str, default='data/studio.exr')
    args.add_argument('--out_dir', type=str, default='output/')
    args = args.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    envmap = cv2.imread(args.envmap, cv2.IMREAD_UNCHANGED)
    coeffs = project_envmap(envmap, args.order)
    print('coeffs:')
    print(coeffs)

    # the first row of coeffs is equal to
    #   the solid angle weighted average of envmap scaled by
    #       the first order of SH basis function
    #           which is 0.282095

    name = os.path.splitext(os.path.basename(args.envmap))[0]
    print('saving coeffs to:', os.path.join(args.out_dir, f'{name}_{args.order}.npy'))
    np.save(os.path.join(args.out_dir, f'{name}_{args.order}.npy'), coeffs)

    # SH rotation
    # quat = glm.angleAxis(glm.radians(30), glm.vec3(0, 1, 0))
    # coeffs = np.load('forest_sh.npy')
    # rotated_r = rotate(coeffs[:, 0], quat)
