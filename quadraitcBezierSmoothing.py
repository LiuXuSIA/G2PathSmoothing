# -*- coding: utf-8 -*-

'''
path_smootging.py
author: 'liuxu'
email: liuxu172@mails.ucas.ac.cn
reference: https://github.com/zhipeiyan/kappa-Curves
'''

'''
uasge:
1) run the code
2) click "left mouse button" to determine the control points
   Ensure there are at least four control points
3) press the "space key" to generate the curve
4) click the "right mouse button" to tweak the control points
5) press the "escape key" to clear the canvas
6) 2)->5)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, special

def bernstein_poly(n, i, t):
    return special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier_point(t, control_points):
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def bezier_curve(control_points, n_points = 100):
    inters = []
    for control_point in control_points:
        for t in np.linspace(0, 1, n_points):
            inters.append(bezier_point(t, control_point))
    return np.array(inters)

def bezier_derivatives_control_points(control_points, n_order):
    # A derivative of a n-order bezier curve is a (n-1)-order bezier curve.
    w = {0: control_points}
    for i in range(n_order):
        for k in range(len(w[i])):
            n = len(w[i][k])
            if k == 0:
                w[i + 1] = [[(n - 1) * (w[i][k][j + 1] - w[i][k][j]) for j in range(n - 1)]]
            else:
                w[i + 1].append([(n - 1) * (w[i][k][j + 1] - w[i][k][j]) for j in range(n - 1)])
    return w

def bezier_curvature(dx, dy, ddx, ddy):
    #Compute curvature at one point given first and second derivatives.
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

'''
Solve the cubic equation shown in equation (17) in the form of
a*t^3 + b*t^2 + c*t + d = 0, 
and the solution will be in [0, 1]
'''
def max_t(c0, cp, c2):
    # the coefficients
    # Notice: there is an typo of coefficient a in the original manuscript.
    # The correct one should be:
    a = np.dot(c2 - c0, c2 - c0)
    b = 3 * np.dot(c2 - c0, c0 - cp)
    c = np.dot(3 * c0 - 2 * cp - c2, c0 - cp)
    d = - np.dot(c0 - cp, c0 - cp)

    # to transform the equation to be x^3 + p*x + q = 0, 
    # where x = t + b / 3a
    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    
    # discriminant of whether there are multiple roots 
    delta = -(4 * p ** 3 + 27 * q ** 2)
    if (delta <= 0):
        # single real root
	    return np.cbrt(-q / 2 + np.sqrt(q ** 2 / 4 + p ** 3 / 27)) + np.cbrt(-q / 2 - np.sqrt(q * q / 4 + p ** 3 / 27)) - b / 3 / a
    else:
        # three real roots
        for k in range(3):
            t = 2 * np.sqrt(-p / 3) * np.cos(1 / 3 * np.arccos(3 * q / 2 / p * np.sqrt(-3 / p)) - 2 * np.pi * k / 3) - b / 3 / a
            if 0 <= t and t <= 1:
                return t
    # error
    return -1

'''
Solve the lambda using equation (14)
'''
def lamb(c0, c1, c3, c4):
    def temp(A, B, C):
        mat = np.c_[B-A, C-B]
        return np.abs(np.linalg.det(mat))
    return np.sqrt(temp(c0, c1, c3)) / (np.sqrt(temp(c0, c1, c3)+1e-15) + np.sqrt(temp(c1, c3, c4)+1e-15))

# paras_iter_play -> if display the iteration of the curve
def ctrls_optimization(pts, max_iter = 50, paras_iter_play = False):
    print("smoothing start!")
    pts = np.array(pts)
    n = len(pts) - 2
    # return bezier control points, a vector of triple of 2d points, {{q0,q1,q2}, {q0,q1,q2}, ...}
    ctrls = np.ones([n,3]).tolist()
    for i in range(1, n + 1):
        ctrls[i - 1][0] = (pts[i] + pts[i - 1]) / 2
        ctrls[i - 1][1] = pts[i]
        ctrls[i - 1][2] = (pts[i] + pts[i + 1]) / 2

    ctrls[0][0] = pts[0]
    ctrls[n - 1][2] = pts[n + 1]

    t, ld = [0 for i in range(n)], [0 for i in range(n - 1)]
    matA = np.mat(np.zeros([n, n]))
    matB = pts[1:-1]*1.0

    for iter_i in range(max_iter):
        for i in range(n):
            t[i] = max_t(ctrls[i][0], pts[i + 1], ctrls[i][2])
        for i in range(n - 1):
            ld[i] = lamb(ctrls[i][0], ctrls[i][1], ctrls[i + 1][1], ctrls[i + 1][2])
    
        matA[0, 0] = 2 * (1 - t[0]) * t[0] + (1 - ld[0]) * t[0] * t[0]
        matA[0, 1] = ld[0] * t[0] * t[0]
        for i in range(1, n - 1):
            matA[i, i - 1] = (1 - ld[i - 1]) * (1 - t[i]) * (1 - t[i])
            matA[i, i] = ld[i - 1] * (1 - t[i]) * (1 - t[i]) + 2 * (1 - t[i]) * t[i] + (1 - ld[i]) * t[i] * t[i]
            matA[i, i + 1] = ld[i] * t[i] * t[i]
        matA[n - 1, n - 2] = (1 - ld[n - 2]) * (1 - t[n - 1]) * (1 - t[n - 1])
        matA[n - 1, n - 1] = ld[n - 2] * (1 - t[n - 1]) * (1 - t[n - 1]) + 2 * (1 - t[n - 1]) * t[n - 1]

        matB[0] = pts[1] - (1 - t[0]) * (1 - t[0]) * pts[0]
        matB[n - 1] = pts[n] - t[n - 1] * t[n - 1] * pts[n + 1]
        # linear solver of Ax=B
        corners = linalg.solve(matA, matB)
        for i in range(n): 
            ctrls[i][1] = corners[i]  
        for i in range(n - 1):
            ctrls[i + 1][0] = ctrls[i][2] = (1 - ld[i]) * ctrls[i][1] + ld[i] * ctrls[i + 1][1]

        if paras_iter_play: plot_curve_change(ctrls, iter_i)

    return ctrls

def loadData(filePath):
    Data = []
    fr = open(filePath)
    initialData = fr.readlines()
    fr.close()
    for element in initialData:
        lineArr = element.strip().split(' ')
        Data.append([float(x) for x in lineArr])
    return np.array(Data)

if __name__ == "__main__":

    point_hill = 'pathPoints\\point_hill.txt'

    road_points = loadData(point_hill)

    control_points = ctrls_optimization(road_points, max_iter = 50, paras_iter_play=False)
    curve_inters = bezier_curve(control_points, n_points = 500)

    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.title('our method', fontsize = 15)
    plt.plot(curve_inters[:,0], curve_inters[:,1],linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(road_points[:,0], road_points[:,1], c='g',s=40, label='path points')
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()