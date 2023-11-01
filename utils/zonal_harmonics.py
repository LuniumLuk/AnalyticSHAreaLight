import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import factorial, lpmv
from tqdm import tqdm, trange
import argparse

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

import utils.spherical_harmonics as sh

'''
Zonal Harmonics Factorization for SH Rotation
Reference:
[1] 'Sparse Zonal Harmonic Factorization for Efficient SH Rotation' by Derek Nowrouzezahrai et al., SIGGRAPH 2012
'''

def spherical_to_cartesian_np(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def spherical_to_cartesian_torch(phi, theta):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return x, y, z

def P_torch(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(1, m + 1):
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

def eval_sh_torch(l, m, phi, theta):
    assert l >= 0
    assert -l <= m and m <= l

    x, y, z = spherical_to_cartesian_torch(phi, theta)
    if l == 0:
        return 0.282095 * torch.ones_like(x)
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
        return math.sqrt(2.0) * kml * torch.cos(m * phi) * P_torch(l, m, torch.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kml * torch.sin(-m * phi) * P_torch(l, -m, torch.cos(theta))
    else:
        return kml * P_torch(l, 0, torch.cos(theta))

def eval_zh_np(l, z):
    assert l >= 0

    if l == 0:
        return 0.282095 * np.ones_like(z)
    elif l == 1:
        return 0.488603 * z
    elif l == 2:
        return 0.315392 * (3 * z * z - 1)
    
    kml = math.sqrt((2 * l + 1) / (4 * math.pi))
    return kml * lpmv(0, l, z)


def minimize(func, variables, iters=4000, lr=0.1, converge_iter=200, on_step=None, decay_func=None, on_reset=None):
    min_loss = float('inf')
    best_fit = None
    optimizer = torch.optim.Adam(variables, lr=lr)
    converge_count = 0
    for i in trange(iters):
        loss = func(variables)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if on_step:
            on_step()

        converge_count += 1

        # save best result:
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_fit = [var.detach().numpy() for var in variables]
            tqdm.write(f'Iter: {i} Loss: {loss.item()} [Best Fit]')
            converge_count = 0
        
        if decay_func:
            new_lrate = lr * decay_func(i / iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

        if i % 100 == 0:
            tqdm.write(f'Iter: {i} Loss: {loss.item()}')

        if converge_count > converge_iter:
            print('Converged')
            break

        if (converge_count > converge_iter // 2) and (min_loss > 1):
            if on_reset:
                on_reset()

    return best_fit

def optimize_zhf(l, visualize=False):
    m = 2 * l + 1
    phi = torch.tensor(np.random.rand(m), dtype=torch.float32, requires_grad=True)
    theta = torch.tensor(np.random.rand(m), dtype=torch.float32, requires_grad=True)
    alpha = torch.tensor(np.random.rand(m, m), dtype=torch.float32, requires_grad=True)

    def func(variables):
        phi, theta, alpha = variables

        A = alpha
        D = torch.eye(m, dtype=torch.float32) * math.sqrt(4 * math.pi / (2 * l + 1))
        Y = torch.concatenate([
            eval_sh_torch(l, m, phi, theta)[..., None] for m in range(-l, l + 1)
        ], dim=-1)

        return torch.sum((A @ D @ Y - torch.eye(m, dtype=torch.float32)) ** 2)

    fit = minimize(func, [phi, theta, alpha])

    phi, theta, alpha = fit

    if visualize:
        x, y, z = spherical_to_cartesian_np(phi, theta)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, marker='o', s=20)
        ax.axis('equal')
        plt.show()

    np.savez(f'output/zhf/zhf_{l}.npz', phi=phi, theta=theta, alpha=alpha)

def make_array_np(components):
    array = [c[..., None] for c in components]
    return np.concatenate(array, axis=-1)

def optimize_zhf_share(l_max, random_first_order=False):
    phis = []
    thetas = []
    alphas = []
    for l in range(l_max + 1):
        if l == 0:
            if random_first_order:
                phi_shared = torch.tensor([], dtype=torch.float32)
                theta_shared = torch.tensor([], dtype=torch.float32)
            else:
                phis.append(np.array([0], dtype=np.float32))
                thetas.append(np.array([0], dtype=np.float32))
                alphas.append(np.array([[1]], dtype=np.float32))
        else:
            phi_shared = torch.tensor(phis[-1], dtype=torch.float32)
            theta_shared = torch.tensor(thetas[-1], dtype=torch.float32)

        m = 2 * l + 1
        phi = torch.tensor(np.random.rand(m - len(phi_shared)), dtype=torch.float32, requires_grad=True)
        theta = torch.tensor(np.random.rand(m - len(theta_shared)), dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor(np.random.rand(m, m), dtype=torch.float32, requires_grad=True)

        def func(variables):
            phi, theta, alpha = variables

            A = alpha
            D = torch.eye(m, dtype=torch.float32) * math.sqrt(4 * math.pi / (2 * l + 1))
            Y = torch.concatenate([
                eval_sh_torch(l, m,
                    torch.concatenate((phi_shared, phi)),
                    torch.concatenate((theta_shared, theta)))[..., None] for m in range(-l, l + 1)
            ], dim=-1)

            return torch.sum((A @ D @ Y - torch.eye(m, dtype=torch.float32)) ** 2)

        fit = minimize(func, [phi, theta, alpha])

        phi, theta, alpha = fit

        phis.append(np.concatenate((phi_shared.detach().numpy(), phi)))
        thetas.append(np.concatenate((theta_shared.detach().numpy(), theta)))
        alphas.append(alpha)
    
    for l in range(l_max + 1):
        np.savez(f'output/zhf/zhf_{l}.npz', phi=phis[l], theta=thetas[l], alpha=alphas[l])

def optimize_zhf_share_compress(args):
    # lobes from
    # Sparse Zonal Harmonic Factorization for Efficient SH Rotation: Supplemental Material and Implementation Sketch
    LOBES = np.array([[1.5708, 1.5708],[0.9553, -2.3562],[3.1416, 2.3562],[0.9553, 0.7854],[2.1863, 2.3562]], dtype=np.float32)
    n = args.l_max * 2 + 1


    # [-0.27508008 -0.36825427  0.8833151  -0.15443112  1.8604635 ]
    # [-1.39523     1.2319051   1.3549458  -0.0086099   0.36543268]
    # [ 1.         -0.03794615  0.6704387  -1.0096903   1.8842524   1.8981442
    #   0.18611863  0.6972663  -0.16064952 -0.3192759   0.522913   -0.46431652
    #   1.1585373   1.253893   -0.83105075 -0.01267773 -0.37397915 -0.37628445
    #   1.1215832  -1.7496003  -0.01425202  0.01954121  0.00495442  1.0004194
    #   0.00233062  0.960373   -1.3859018  -0.39669693  0.14914693 -0.42939883
    #   1.3227667   0.24047206  0.8300638   1.4710302  -0.53402096]

    phis = LOBES.T[:n, 0]
    thetas = LOBES.T[:n, 1]
    # phis = np.array([0, 0, np.pi / 2], dtype=np.float32)
    # thetas = np.array([0, np.pi / 2, np.pi / 2], dtype=np.float32)
    phis = np.array([], dtype=np.float32)
    thetas = np.array([], dtype=np.float32)
    alphas = np.array([1], dtype=np.float32)
    for l in range(1, args.l_max + 1):
        m = 2 * l + 1

        phi_shared = torch.tensor(phis, dtype=torch.float32)
        theta_shared = torch.tensor(thetas, dtype=torch.float32)

        phi = torch.tensor(np.random.rand(m - len(phi_shared)), dtype=torch.float32, requires_grad=True)
        theta = torch.tensor(np.random.rand(m - len(theta_shared)), dtype=torch.float32, requires_grad=True)
        alpha = torch.tensor(np.random.rand(m, m), dtype=torch.float32, requires_grad=True)

        def func(variables):
            phi, theta, alpha = variables

            A = alpha
            D = torch.eye(m, dtype=torch.float32) * math.sqrt(4 * math.pi / (2 * l + 1))
            Y = torch.concatenate([
                eval_sh_torch(l, m,
                    torch.concatenate((phi_shared, phi)),
                    torch.concatenate((theta_shared, theta)))[..., None] for m in range(-l, l + 1)
            ], dim=-1)

            return torch.sum((A @ D @ Y - torch.eye(m, dtype=torch.float32)) ** 2)

        def on_step():
            phi.data.nan_to_num_(nan=0.0)
            phi.data.clamp_(0.0, 2 * torch.pi)
            theta.data.nan_to_num_(nan=0.0)
            theta.data.clamp_(0.0, torch.pi)
        
        def decay_func(x):
            return 0.1 ** x
        
        def on_reset():
            tqdm.write('Reset')
            phi.data.random_(0.0, 2 * torch.pi)
            theta.data.random_(0.0, torch.pi)

        fit = minimize(func, [phi, theta, alpha], 2000 * l, 0.1 * 0.9 ** l, 1000, on_step=on_step, decay_func=decay_func, on_reset=on_reset)

        phi, theta, alpha = fit

        phis = np.concatenate((phis, phi))
        thetas = np.concatenate((thetas, theta))
        alphas = np.concatenate((alphas, alpha.flatten()))


    return {
        'phi': phis,
        'theta': thetas,
        'alpha': alphas,
    }

def unzip_alpha(alphas, l):
    def an(x):
        return int(x * (4 * x ** 2 - 1) / 3)
    return alphas[an(l):an(l+1)].reshape((m, m))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--l_max', type=int, default=2)
    args.add_argument('--out_dir', type=str, default='output/')
    args = args.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    zhf_compress = optimize_zhf_share_compress(args)
    np.savez(os.path.join(args.out_dir, f'zhf_compress_{args.l_max}.npz'),
             phi=zhf_compress['phi'], theta=zhf_compress['theta'], alpha=zhf_compress['alpha'])

    phis = zhf_compress['phi']
    thetas = zhf_compress['theta']
    alphas = zhf_compress['alpha']

    print('phis', phis.shape)
    print(phis)
    print('thetas', thetas.shape)
    print(thetas)
    print('alphas', alphas.shape)
    print(alphas)

    # validate result
    for l in range(args.l_max + 1):
        m = 2 * l + 1
        phi = phis[:m]
        theta = thetas[:m]
        alpha = unzip_alpha(alphas, l)

        sample_N = 4
        for m in range(-l, l + 1):
            sample_phi = np.random.rand(sample_N)
            sample_theta = np.random.rand(sample_N)

            sample = make_array_np(spherical_to_cartesian_np(sample_phi, sample_theta))
            omega = make_array_np(spherical_to_cartesian_np(phi, theta))

            z = sample @ omega.T
            zh = eval_zh_np(l, z)
            a = alpha[m + l]

            print(alpha[m + l])
            estimated = zh @ a

            print(f'validate m={m}')
            print(sh.eval_sh(l, m, sample_phi, sample_theta))
            print(estimated)

    # for l in range(3):
    #     optimize_zhf(l)

    #     # validate
    #     zhf = np.load(f'output/zhf/zhf_{l}.npz')
    #     phi = zhf['phi']
    #     theta = zhf['theta']
    #     alpha = zhf['alpha']

    #     print('phi', phi * 180 / np.pi)
    #     print('theta', theta * 180 / np.pi)

    #     sample_N = 4
    #     for m in range(-l, l + 1):
    #         sample_phi = np.random.rand(sample_N)
    #         sample_theta = np.random.rand(sample_N)

    #         sample = make_array_np(spherical_to_cartesian_np(sample_phi, sample_theta))
    #         omega = make_array_np(spherical_to_cartesian_np(phi, theta))

    #         z = sample @ omega.T
    #         zh = eval_zh_np(l, z)
    #         a = alpha[m + l]
    #         estimated = zh @ a

    #         print(f'validate m={m}')
    #         print(sh.eval_sh(l, m, sample_phi, sample_theta))
    #         print(estimated)
