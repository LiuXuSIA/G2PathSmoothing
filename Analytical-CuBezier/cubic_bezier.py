import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def calc_bezier_path(control_points, ds=0.01):
        traj = []
        for t in np.arange(0, 1, ds):
            traj.append(bezier(t, control_points))
        return np.array(traj)

def bernstein_poly(n, i, t):
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier(t, control_points):
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def pathsmoothbezier(wayPointsX, wayPointsY, ds, *varargin):
    num = len(wayPointsX)
    if len(wayPointsX) != len(wayPointsY) or num < 3:
        raise ValueError('wayPointsX and wayPointsY must have at least three points')
    
    maximumCurvature = 1
    wayPointsZ = np.zeros(num)
    if len(varargin) > 0 and len(varargin) < 3:
        if len(varargin) == 1:
            maximumCurvature = varargin[0]
        else:
            arg = varargin[0]
            if len(arg) != num:
                raise ValueError('wayPointsZ is not a vector of same size of wayPointsX')
            wayPointsZ = arg
            maximumCurvature = varargin[1]
    elif len(varargin) > 2:
        raise ValueError('Number of arguments are more that allowed by this function')
        
    c1 = 7.2364
    c2 = (2*(np.sqrt(6)-1))/5
    c3 = (c2+4)/(c1+6)

    X = [wayPointsX[0]]
    Y = [wayPointsY[0]]
    Z = [wayPointsZ[0]]

    for i in range(num-2):
        w1 = np.array([wayPointsX[i], wayPointsY[i], wayPointsZ[i]]).reshape(-1,1)
        w2 = np.array([wayPointsX[i+1], wayPointsY[i+1], wayPointsZ[i+1]]).reshape(-1,1)
        w3 = np.array([wayPointsX[i+2], wayPointsY[i+2], wayPointsZ[i+2]]).reshape(-1,1)

        bias = np.array([1e-2, 1e-4, 0]).reshape(-1,1)

        ut = (w2-w1)/(np.linalg.norm(w2-w1))
        up = (w2-w3+bias)/(np.linalg.norm(w2-w3+bias))

        alpha = np.arccos(np.sum(ut * up))

        # while np.abs(alpha - np.pi) < 1e-2:
        #     print(np.abs(alpha - np.pi))
        #     ut = (w2-w1)/(np.linalg.norm(w2-w1))+bias
        #     up = (w2-w3)/(np.linalg.norm(w2-w3))+bias
        #     alpha = np.arccos(np.sum(ut * up))

        ub = np.cross(up[:,0],ut[:,0]).reshape(-1,1)
        # np.seterr(divide='ignore',invalid='ignore')
        # print(alpha, ub)
        ub = ub/(np.linalg.norm(ub))
        un = np.cross(ub[:,0],ut[:,0]).reshape(-1,1)

        rotationMat     = np.c_[ut, un, ub]
        positionMat     = w1
        transformMat    = np.r_[np.c_[rotationMat, np.zeros([3,1])], np.r_[positionMat, np.ones([1, 1])].T]
        # print(transformMat)
        transformMatInv = np.linalg.inv(transformMat)

        m1 = np.dot(transformMatInv, np.append(w1, 1).reshape(-1,1))[:-1]
        m2 = np.dot(transformMatInv, np.append(w2, 1).reshape(-1,1))[:-1]
        m3 = np.dot(transformMatInv, np.append(w3, 1).reshape(-1,1))[:-1]

        m1m2 = m2-m1
        m2m3 = m3-m2

        m1m2 = m1m2/np.linalg.norm(m1m2)
        m2m3 = m2m3/np.linalg.norm(m2m3)
        u1 = -m1m2
        u2 = m2m3

        gamma = np.arccos(np.sum(m1m2 * m2m3))
        beta = gamma/2
        if beta == 0:
            beta = 0.00001
        d = ((c2+4)**2/(54*c3))*(np.sin(beta)/(maximumCurvature*(np.cos(beta))**2))

        hb = c3*d
        he = hb
        gb = c2*c3*d
        ge = gb
        kb = ((6*c3*np.cos(beta))/(c2+4))*d
        ke = kb

        B0 = m2 + d*u1
        B1 = B0 - gb*u1
        B2 = B1 - hb*u1

        E0 = m2 + d*u2
        E1 = E0 - ge*u2
        E2 = E1 - he*u2

        B2E2 = E2-B2
        ud = B2E2/np.linalg.norm(B2E2)

        B3 = B2 + kb*ud
        E3 = E2 - ke*ud

        Bmatrix = np.dot(transformMat, np.r_[np.c_[B0, B1, B2, B3], np.ones([1,4])])
        Ematrix = np.dot(transformMat, np.r_[np.c_[E0, E1, E2, E3], np.ones([1,4])])

        Bmatrix = Bmatrix[:-1,:].T
        Ematrix = Ematrix[:-1,:].T

        # print(Bmatrix)

        # print(Bmatrix)
        bezierB = calc_bezier_path(Bmatrix, ds)
        bezierE = calc_bezier_path(Ematrix, ds)[::-1]

        X.extend(bezierB[:,0].tolist())
        Y.extend(bezierB[:,1].tolist())
        Z.extend(bezierB[:,2].tolist())

        X.extend(bezierE[:,0].tolist())
        Y.extend(bezierE[:,1].tolist())
        Z.extend(bezierE[:,2].tolist())

    X.append(wayPointsX[-1])
    Y.append(wayPointsY[-1])
    Z.append(wayPointsZ[-1])

    return np.c_[X, Y, Z]

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

    fx = road_points[:, 0]
    fy = road_points[:, 1]
    # print(fx)
    # print(fy)

    # fx = [0,10,10,20,20,30,30,40,40,-10,-10,  0]
    # fy = [0, 0,10, 0,10,10, 0, 0,20, 20,-20,-20]

    path = pathsmoothbezier(fx,fy, 0.01, 0.45)
    plt.figure()
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2.2)
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_linewidth(2.2)
    plt.title('analytical cubic Bezier', fontsize = 15)
    plt.plot(path[:,0], path[:,1],linewidth=4, c='darkorange',label='trajectory')
    plt.scatter(fx, fy,c='g',s=40, label='path points')
    # plt.plot(fx, fy,c='gray',label='path',linewidth=3)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()