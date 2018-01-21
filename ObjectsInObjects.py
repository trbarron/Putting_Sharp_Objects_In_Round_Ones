import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg

# --------------- #
# -- GEO TESTS -- #
# --------------- #

def same_side(p1,p2,A,B):
    cp1 = np.cross(B-A,p1-A)
    cp2 = np.cross(B-A,p2-A)
    if np.dot(cp1,cp2) >= 0:
       return True
    else:
        return False

def same_dir(p1,A,C):
    cp1 = (A-C)
    cp2 = (A-p1)
    if np.dot(cp1,cp2) >=0:
        return True
    else:
        return False

def enclosed(alpha,beta,gamma,delta,center):
    array_0 = [[alpha[0],alpha[1],alpha[2],1],
               [beta[0],beta[1],beta[2],1],
               [gamma[0],gamma[1],gamma[2],1],
               [delta[0],delta[1],delta[2],1]]

    array_1 = [[center[0],center[1],center[2],1],
               [beta[0],beta[1],beta[2],1],
               [gamma[0],gamma[1],gamma[2],1],
               [delta[0],delta[1],delta[2],1]]

    array_2 = [[alpha[0],alpha[1],alpha[2],1],
               [center[0],center[1],center[2],1],
               [gamma[0],gamma[1],gamma[2],1],
               [delta[0],delta[1],delta[2],1]]

    array_3 = [[alpha[0],alpha[1],alpha[2],1],
               [beta[0],beta[1],beta[2],1],
               [center[0],center[1],center[2],1],
               [delta[0],delta[1],delta[2],1]]

    array_4 = [[alpha[0],alpha[1],alpha[2],1],
               [beta[0],beta[1],beta[2],1],
               [gamma[0],gamma[1],gamma[2],1],
               [center[0],center[1],center[2],1]]

    if (np.linalg.det(array_0)>=0) == (np.linalg.det(array_1)>=0) == (np.linalg.det(array_2)>=0) == (np.linalg.det(array_3)>=0) == (np.linalg.det(array_4)>=0):
        return True
    else:
        return False


def check_point_in_triangle(alpha,beta,gamma,center):   
   return (same_side(center,alpha,beta,gamma) and same_side(center,beta,gamma,alpha) and same_side(center,gamma,alpha,beta))
 


# --------------- #
# ----- 2D ------ #
# --------------- #

def rand_point_on_2d_circle():
    angle = random.random()*math.pi*2;
    return np.array([math.cos(angle),math.sin(angle)])

def draw_triangles(coords,iteration,fail):
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

    verts = [
        (coords[0][0], coords[0][1]), # left, bottom
        (coords[2][0], coords[2][1]), # right, top
        (coords[1][0], coords[1][1]), # right, bottom
        (0., 0.), # ignored
        ]

    path = Path(verts, codes)

    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    ax.add_artist(circle1)
    
    if fail:
        patch = patches.PathPatch(path, ec='r', facecolor='none', lw=1)
    else:
        patch = patches.PathPatch(path, ec='g', facecolor='none', lw=1)
    ax.add_patch(patch)

    xs, ys = zip(*verts[0:3])
    ax.plot(xs, ys, 'x', lw=2, color='black', ms=10)
    ax.plot(verts[3][0], verts[3][1], 'o', lw=2, color='black', ms=3)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    plt.title("Triangle in a circle \nIteration "+str(i))

    plt.savefig(str(iteration)+'.png')
    #plt.show()
    plt.clf()
    plt.close()
    return

def run_2d_iterations(iterations):
    wins = 0
    for i in range(0,iterations):    
        alpha = rand_point_on_2d_circle()
        beta = rand_point_on_2d_circle()
        gamma = rand_point_on_2d_circle()
        center = np.array([0,0])
        #draw_triangles([alpha,beta,gamma],i,not(same_side(center,alpha,beta,gamma) and same_side(center,beta,gamma, alpha) and same_side(center,gamma,alpha,beta)))
        if check_point_in_triangle(alpha,beta,gamma,center):
            wins += 1
    print(str(wins/iterations))
    return
	

	
# --------------- #
# ----- 3D ------ #
# --------------- #

def rand_point_on_3d_sphere():
    x1 = 2
    x2 = 2
    while (x1**2+x2**2 >= 1):
        x1 = random.random()*2-1
        x2 = random.random()*2-1
    return np.array([2*x1*(1-x1**2-x2**2)**0.5,2*x2*(1-x1**2-x2**2)**0.5,1-2*(x1**2+x2**2)])

def draw_triangles_3d(coords,iteration,won):
    verts = [
        (coords[0][0], coords[0][1], coords[0][2]), # left, bottom
        (coords[2][0], coords[2][1], coords[2][2]), # right, top
        (coords[3][0], coords[3][1], coords[3][2]), # right, top
        (coords[1][0], coords[1][1], coords[1][2]), # right, bottom
        ]
    x = np.array([coords[0][0],coords[1][0],coords[2][0],coords[3][0]])
    y = np.array([coords[0][1],coords[1][1],coords[2][1],coords[3][1]])
    z = np.array([coords[0][2],coords[1][2],coords[2][2],coords[3][2]])

    data = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis=1)
    fig = plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')

    ax = fig.add_subplot(111, projection = '3d')

    center = data.mean(axis=0)
    distances = np.empty((0))
    for row in verts:
        distances = np.append(distances, linalg.norm(row - center))
        #print(row)
    #print("\n")

    vertices = distances.argsort()[-4:]
    Vertices_reorder = [vertices[0], vertices[2], vertices[1], vertices[3], vertices[0], vertices[1], vertices[3], vertices[2]]


    # draw sphere
    u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:50j]
    circ_x = np.cos(u)*np.sin(v)
    circ_y = np.sin(u)*np.sin(v)
    circ_z = np.cos(v)
    ax.plot_wireframe(circ_x, circ_y, circ_z, lw=0.1, color="black", linestyle="--")

    xs, ys, zs = zip(*verts[0:4])
    ax.plot(xs, ys, zs, 'x', lw=2, color='black', ms=5)
    ax.scatter(0,0,0, color='black', marker="o", lw=0.1)
    
    if won:
        ax.plot(x[Vertices_reorder], y[Vertices_reorder], z[Vertices_reorder],color = "green")
    else:
        ax.plot(x[Vertices_reorder], y[Vertices_reorder], z[Vertices_reorder],color = "red")


    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    plt.title("Tetrahedron in a sphere\nIteration: "+str(iteration) + "\n Enclosed:" + str(won))
    plt.savefig(str(iteration)+'.png')
    
    #plt.show()
    plt.clf()
    plt.close()
    return

def run_3d_iterations(iterations):
    wins = 0
    for i in range(0,iterations):    
        alpha = rand_point_on_3d_sphere()
        beta = rand_point_on_3d_sphere()
        gamma = rand_point_on_3d_sphere()
        delta = rand_point_on_3d_sphere()
        center = np.array([0,0,0])
        won = False
        if run_3d_test_take_2(alpha,beta,gamma,delta,center):
            wins += 1
            won = True
        draw_triangles_3d([alpha,beta,gamma,delta],i,won)

        if (i%1000 == 0) and (i > 0):
            print("% Wins in " + str(i) + " is: " + str(wins/i))
    return
    
def run_3d_test(alpha,beta,gamma,delta,center):
    points_in_tri = check_point_in_triangle(alpha,beta,gamma,center) + check_point_in_triangle(alpha,beta,delta,center) + check_point_in_triangle(alpha,beta,delta,center) + check_point_in_triangle(alpha,beta,delta,center)
    if (points_in_tri >= 4):
        return True
    else:
        return

def run_3d_test_take_2(alpha,beta,gamma,delta,center):
    points_in_tri = check_point_in_triangle(alpha,beta,gamma,center) + check_point_in_triangle(alpha,beta,delta,center) + check_point_in_triangle(alpha,beta,delta,center) + check_point_in_triangle(alpha,beta,delta,center)
    if enclosed(alpha,beta,gamma,delta,center):
        return True
    else:
        return 
		
		
run_3d_iterations(1000000)
