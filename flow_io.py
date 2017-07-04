import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from itertools import *
import math
# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def read_flow(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (w[0], h[0], 2))
            return data2D



# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def write_flow(flow,path):
    with open(path, 'wb') as f:
        magic=np.array([202021.25], np.float32)
        magic.tofile(f)
        w=np.array([flow.shape[0]], np.int32)
        h=np.array([flow.shape[1]], np.int32)
        w.tofile(f)
        h.tofile(f)
        data = np.resize(flow.astype(np.float32), (2*w[0]*h[0]))
        data.tofile(f)


def visualize_flow(flow,spacing):
    plt.quiver(flow[::spacing,::spacing,1],flow[::spacing,::spacing,0])

# write_flow(flow,path)
# flow2=read_flow(path)
# plt.subplot(1,2,1)
# visualize_flow(flow)
# plt.subplot(1,2,2)
# visualize_flow(flow2)
# plt.show()

def gen_flow_cos_sin(height, width, amplitude=10):
    X, Y = np.meshgrid(np.linspace(0, np.pi, width), np.linspace(0, np.pi, height))
    V = np.cos(X)*amplitude
    U = np.sin(Y)*amplitude
    flow=np.stack([U,V],-1)
    return flow

def gen_flow_zoom(height, width, amplitude=10):
    X=np.linspace(-amplitude,amplitude,width)
    Y=np.linspace(-amplitude,amplitude,height)
    V,U=np.meshgrid(X,Y)
    flow=np.stack([U,V],-1)
    return flow

def gen_flow_fisheye(height, width, amplitude=10):
    flow=np.zeros((height,width,2))
    for i,j in product(range(flow.shape[0]),range(flow.shape[1])):
        i_new=i-height/2
        j_new=j-width/2
        l=math.sqrt(i_new*i_new+j_new*j_new)/(2*math.sqrt(height*height+width*width))
        flow[i,j]=[l*i_new,l*j_new]
    return flow

def gen_flow_rotation(height, width, amplitude=10):
    X=np.linspace(-amplitude/2,amplitude/2,width)
    Y=np.linspace(-amplitude/2,amplitude/2,height)
    V,U=np.meshgrid(X,Y)
    flow=np.stack([V-U,-U-V],-1)
    return flow

def gen_flow_spiral(height, width, amplitude=10):
    X=np.linspace(-amplitude/2,amplitude/2,width)
    Y=np.linspace(-amplitude/2,amplitude/2,height)
    V,U=np.meshgrid(X,Y)
    flow=np.stack([V-U,-U-V],-1)
    length=np.linalg.norm(flow,axis=2)
    flow_unit=flow/length.reshape([flow.shape[0],flow.shape[1],1]) #normalize
    dist_center=np.sqrt(np.power(U,2)+np.power(V,2))+2.0/amplitude
    flow_def=flow_unit/dist_center.reshape([flow.shape[0],flow.shape[1],1])+flow_unit*amplitude/2    #normalize
    return flow_def

def gen_flow_cylinder(height, width, amplitude=10):
    X=np.linspace(0,np.pi,width)
    X=np.sin(X)*amplitude
    Y=np.zeros(height)
    V,U=np.meshgrid(X,Y)
    flow=np.stack([U,V],-1)
    return flow

def inverse_flow(flow):
    invflow_cum=np.zeros(flow.shape)
    invflow_sum=np.zeros(flow.shape[:2])
    invflow=np.zeros(flow.shape)
    for i,j in product(range(flow.shape[0]),range(flow.shape[1])):
    #for i,j in product(range(2,6),range(2,3)):
        #print ("========== "+str(i)+" "+str(j))
        #print flow[i,j]
        pos_f=[i,j]+flow[i,j]
        pos=np.floor(pos_f).astype(int)
        #print pos_f
        #print pos
        for di,dj in product(range(2),range(2)):
            x=pos[0]+di
            y=pos[1]+dj
            #print ("==== "+str(x)+" "+str(y))
            weight=(1-abs(pos_f[0]-x))*(1-abs(pos_f[1]-y))
            #print weight
            if x<flow.shape[0] and y<flow.shape[1] and x>0 and y>0:
                invflow_cum[x,y] += -flow[i,j]*weight
                invflow_sum[x,y] += weight
    for i,j in product(range(flow.shape[0]),range(flow.shape[1])):
        if invflow_sum[i,j] > 0:
            invflow[i,j]=invflow_cum[i,j]/invflow_sum[i,j]
        else:
            invflow[i,j]=[1e30,1e30]
    return invflow


def gen_flow_constant(height, width, value):
    flow=np.zeros((height,width,2))
    for i,j in product(range(flow.shape[0]),range(flow.shape[1])):
        flow[i,j] = value
    return flow


# def apply_flow(img,flow):
#     output=np.zeros(img.shape)
#     for i,j in product(range(output.shape[0]),range(output.shape[1])):
#         pos=np.round([i,j]+flow[i,j]).astype(int)
#         if pos[0]<flow.shape[0] and pos[1]<flow.shape[1] and pos[0]>0 and pos[1]>0:
#             output[pos[0],pos[1]] = 
#     return invflow
