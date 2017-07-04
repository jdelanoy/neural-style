import flow_io
import inspect
import os
import importlib
reload(flow_io)
from matplotlib import pyplot as plt
from skimage import io, transform
import numpy as np

func=inspect.getmembers(flow_io,inspect.isfunction)

d={nom:f for nom,f in func if nom.startswith('gen_flow')}

img_folder='../artistic-videos/example/'
img_pattern='marple8_{:02d}.ppm'
id_max=5

path_flows='../artistic-videos/fake_flows/'
if not os.path.exists(path_flows):
    os.mkdir(path_flows)

image=io.imread(os.path.join(img_folder,img_pattern.format(1)))
height=image.shape[0]
width=image.shape[1]
  
for nom in d:
    print(nom[9:])
    meth=nom[9:]
    #init firectory and image
    if not os.path.exists(path_flows+meth):
        os.mkdir(path_flows+meth)
    os.system('cp '+os.path.join(img_folder,img_pattern.format(1))+' '+os.path.join(path_flows,meth,img_pattern.format(1)))
    #compute flow
    flow=d[nom](height, width,10)
    invflow=flow_io.inverse_flow(flow)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    XY=np.stack((Y,X),0)
    map=XY+invflow.transpose([2,0,1])
    #Apply on all images
    for im in range(1,id_max):
        flow_io.write_flow(flow,os.path.join(path_flows,meth,'forward_{:d}_{:d}.flo'.format(im,im+1)))
        flow_io.write_flow(invflow,os.path.join(path_flows,meth,'backward_{:d}_{:d}.flo'.format(im+1,im)))
        image=io.imread(os.path.join(path_flows,meth,img_pattern.format(im)))
        warped=np.ndarray(image.shape)
        for channel in range(warped.shape[2]):
            warped[:,:,channel]=transform.warp(image[:,:,channel],map)
        conf=np.ndarray(image.shape[:2],image.dtype)       
        conf[:]=255
        io.imsave(os.path.join(path_flows,meth,'reliable_{:d}_{:d}.pgm'.format(im,im+1)),conf)
        conf[invflow[:,:,0]>1000]=0
        io.imsave(os.path.join(path_flows,meth,'reliable_{:d}_{:d}.pgm'.format(im+1,im)),conf)
        io.imsave(os.path.join(path_flows,meth,img_pattern.format(im+1)),warped)
        # plt.subplot(2,2,1)
        # flow_io.visualize_flow(flow, 10)
        # plt.subplot(2,2,2)
        # invflow_visu=invflow
        # invflow_visu[invflow_visu>50]=0
        # flow_io.visualize_flow(invflow, 10)
        # plt.subplot(2,2,3)
        # plt.imshow(image)
        # plt.subplot(2,2,4)
        # plt.imshow(warped)
        # plt.show()

