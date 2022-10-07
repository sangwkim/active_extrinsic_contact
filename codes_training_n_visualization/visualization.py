import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import cv2


class visualization:
    
    def __init__(self, reach, pose_error, height, view_elev=30, view_azim=45, view_center = (0,-25,-52+0), view_radius=30, object_name='rectangle', env_type='hole'):
        
        self.stage = 'no touch'
        self.pose_error = pose_error
        self.pose_trn = (pose_error[0], pose_error[1], 0)
        self.pose_rot = R.from_euler('zyx', (pose_error[2], 0, 0), degrees=False).as_matrix()
        self.height = height
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.view_center = view_center
        self.view_radius = view_radius
        self.object_name = object_name
        self.env_type = env_type
        self.gap = 2
        
        self.plotnum = 0
        self.fig = plt.figure(figsize=(8, 8), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='3d')  
        
        if self.object_name in ['rectangle', 'circle', 'ellipse', 'circle_tight', 'ellipse_tight']:

            self.grp_x = [-17.5, -17.5, -17.5, -17.5, 17.5, 17.5, 17.5, 17.5]
            self.grp_y = [-12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5, -12.5]
            self.grp_z = [-12.5, -12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5]
            self.grp_v = [[0,1,2,3], [4,5,6,7]]
            self.grp_tupleList = list(zip(self.grp_x, self.grp_y, self.grp_z))
            
        elif self.object_name in ['hexagon', 'hexagon_tight']:

            self.grp_x = [-15.156, -15.156, -15.156, -15.156, 15.156, 15.156, 15.156, 15.156]
            self.grp_y = [-12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5, -12.5]
            self.grp_z = [-12.5, -12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5]
            self.grp_v = [[0,1,2,3], [4,5,6,7]]
            self.grp_tupleList = list(zip(self.grp_x, self.grp_y, self.grp_z))                
        
        if 'rectangle' in self.object_name:
            self.obj_x = [-17.5, 17.5, 17.5, -17.5, -17.5, 17.5, 17.5, -17.5]
            self.obj_y = [0, 0, 0, 0, -60, -60, -60, -60]
            self.obj_z = [-25, -25, 25, 25, -25, -25, 25, 25]
            self.obj_v = [[0,1,2,3], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [4,5,6,7]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'hexagon' in self.object_name:
            self.obj_x = [15.156, 15.156, 0, -15.156, -15.156, 0, 15.156, 15.156, 0, -15.156, -15.156, 0]
            self.obj_y = [0, 0, 0, 0, 0, 0, -60, -60, -60, -60, -60, -60]
            self.obj_z = [-8.75, 8.75, 17.5, 8.75, -8.75, -17.5, -8.75, 8.75, 17.5, 8.75, -8.75, -17.5]
            self.obj_v = [[0,1,7,6], [1,2,8,7], [2,3,9,8], [3,4,10,9], [4,5,11,10], [5,0,6,11], [0,1,2,3,4,5], [6,7,8,9,10,11]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'circle' in self.object_name:
            self.obj_x = 2*[17.5 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_y = 40*[0]+40*[-60]
            self.obj_z = 2*[17.5 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                           + [[i for i in range(40)]] + [[i+40 for i in range(40)]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'ellipse' in self.object_name:
            self.obj_x = 2*[17.5 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_y = 40*[0]+40*[-60]
            self.obj_z = 2*[25 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                         + [[i for i in range(40)]] + [[i+40 for i in range(40)]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
                         
        self.cline_x = [-50,50]
        self.cline_y = [0,0]
        self.cline_z = [0,0]
        self.cline_tupleList = list(zip(self.cline_x, self.cline_y, self.cline_z))
        
        if env_type == 'hole':
            if self.object_name in ['rectangle', 'ellipse']:
                self.env_x = [-19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -19.75, -100, 100, 100, -100]
                self.env_y = [-27.25, -27.25, 27.25, 27.25, -27.25, -27.25, 27.25, 27.25, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'hexagon':
                self.env_x = [-17.406, 17.406, 17.406, -17.406, -17.406, 17.406, 17.406, -17.406, -100, 100, 100, -100]
                self.env_y = [-19.75, -19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'circle':
                self.env_x = [-19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -19.75, -100, 100, 100, -100]
                self.env_y = [-19.75, -19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'hexagon_tight':
                self.env_x = 2*[17.406, 17.406, 0, -17.406, -17.406, 0] + [5*x for x in [17.406, 17.406, 0, -17.406, -17.406, 0]]
                self.env_y = 2*[-10.049, 10.049, 20.098, 10.049, -10.049, -20.098] + [5*x for x in [-10.049, 10.049, 20.098, 10.049, -10.049, -20.098]]
                self.env_z = [0, 0, 0, 0, 0, 0, -300, -300, -300, -300, -300, -300, 0, 0, 0, 0, 0, 0]
                self.env_v = [[0,1,7,6], [1,2,8,7], [2,3,9,8], [3,4,10,9], [4,5,11,10], [5,0,6,11], [0,1,13,12], [1,2,14,13], [2,3,15,14], [3,4,16,15], [4,5,17,16], [5,0,12,17]]
            elif self.object_name == 'circle_tight':
                self.env_x = 2*[19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_y = 2*[19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_z = 40*[0]+40*[-300]+40*[0]
                self.env_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                             + [[i,(i+1)%40,(i+1)%40+80,i+80] for i in range(40)] \
                                 + [[i for i in range(40)]]
            elif self.object_name == 'ellipse_tight':
                self.env_x = 2*[19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_y = 2*[27.25 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_z = 40*[0]+40*[-300]+40*[0]
                self.env_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                             + [[i,(i+1)%40,(i+1)%40+80,i+80] for i in range(40)] \
                                 + [[i for i in range(40)]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
        elif env_type == 'wall':
            self.env_x = [-100, -100, -100, 100, 100, 100]
            self.env_y = [1, 1, 100, 1, 1, 100]#[-25, -25, -125, -25, -25, -125]
            self.env_z = [-100, 0, 0, -100, 0, 0]
            self.env_v = [[0,1,4,3], [1,2,5,4]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
            
        self.plot_environment((0,0,-self.height-self.gap), np.eye(3), alpha=0.1) 
        self.plot_cline((0,0,-reach), R.from_euler('zyx',(0,0,-0.5*np.pi)).as_matrix())   
        self.plot_gripper((0,0,0), np.eye(3), alpha=0.1)
        self.plot_object((0,0,-reach), R.from_euler('zyx',(0,0,-0.5*np.pi)).as_matrix(), alpha=0.1)
        self.plot_object((0,0,-self.height), R.from_euler('zyx',(0,0,-0.5*np.pi)).as_matrix(), alpha=0.1)
        
        self.show_env = True
        self.show_grp = True
        self.show_obj_gt = True
        self.show_obj_est = True
        self.show_cline = True
        
        self.images = []
        
        self.plot_show()
        
    def set_show(self, env=True, grp = True, obj_gt=True, obj_est=True, cline=True):
        
        self.show_env = env
        self.show_grp = grp
        self.show_obj_gt = obj_gt
        self.show_obj_est = obj_est
        self.show_cline = cline
        
    def set_viewpoint(self, elev, azim):#elev=0, azim=0): #elev=30, azim=-45):
        
        self.ax.view_init(elev=elev, azim=azim)
        
    def plot_clear(self):
        
        self.ax.clear()
    
    def plot_update(self, grp_trn, grp_rot, obj_trn, obj_rot, cline_trn, cline_rot, obj_gt_trn, obj_gt_rot, alpha=0.25):
        
        self.plotnum += 1
        
        if self.show_env:
            self.plot_environment((0,0,-self.height-self.gap), np.eye(3)) 
        if self.show_cline:
            self.plot_cline(cline_trn, cline_rot) 
        if self.show_grp:
            self.plot_gripper(grp_trn, grp_rot, alpha=alpha)
        if self.show_obj_est:
            self.plot_object(obj_trn, obj_rot, alpha=alpha)
        if self.show_obj_gt:
            self.plot_object(obj_gt_trn, obj_gt_rot, alpha=alpha, fc='g')
        
        self.plot_show()
        
    def plot_gripper(self, trn, rot, s=0, lw=1, alpha=0.25, fc='b', ec='r'):
        
        tupleList = [rot.dot(point) + trn for point in self.grp_tupleList]
        tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        
        poly3d = self.form_poly(tupleList, self.grp_v)
        self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
        self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_object(self, trn, rot, s=0, lw=1, alpha=0.25, fc='b', ec='r'):
        
        tupleList = [rot.dot(point) + trn for point in self.obj_tupleList]
        tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        
        if self.object_name == 'rectangle':        
            poly3d = self.form_poly(tupleList, self.obj_v)
        elif self.object_name == 'hexagon_tight':
            poly3d = self.form_poly(tupleList, self.obj_v[:6])
            poly3d_ = self.form_poly(tupleList, self.obj_v[6:])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0, alpha=alpha))
        elif self.object_name == 'circle_tight':
            poly3d = self.form_poly(tupleList, self.obj_v[-2:])
            poly3d_ = self.form_poly(tupleList, self.obj_v[:-2])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0.1, alpha=alpha))
        elif self.object_name == 'ellipse_tight':
            poly3d = self.form_poly(tupleList, self.obj_v[-2:])
            poly3d_ = self.form_poly(tupleList, self.obj_v[:-2])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0.1, alpha=alpha))
        self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
        self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_environment(self, trn, rot, s=0, lw=0.2, alpha=0.25,ec='y',fc='y'):
        
        tupleList = [rot.dot(point) + trn for point in self.env_tupleList]
        
        if self.env_type == 'hole':
            if self.object_name == 'rectangle':
                poly3d = self.form_poly(tupleList, self.env_v[:4])
                poly3d_ = self.form_poly(tupleList, self.env_v[4:])
            elif self.object_name == 'hexagon_tight':
                poly3d = self.form_poly(tupleList, self.env_v[:6])
                poly3d_ = self.form_poly(tupleList, self.env_v[6:])
            elif self.object_name == 'circle_tight':
                poly3d = self.form_poly(tupleList, self.env_v[-1:])
                poly3d_ = self.form_poly(tupleList, self.env_v[:-1])
            elif self.object_name == 'ellipse_tight':
                poly3d = self.form_poly(tupleList, self.env_v[-1:])
                poly3d_ = self.form_poly(tupleList, self.env_v[:-1])
            self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
            self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=0.12))
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0, alpha=alpha))
        elif self.env_type == 'wall':
            poly3d = self.form_poly(tupleList, self.env_v)
            self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
            self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_cline(self, trn, rot, linestyle='-', lw=4, c='r', alpha=1):
        
        tupleList = [rot.dot(point) + trn for point in self.cline_tupleList]
        tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        points = np.array(tupleList)
        
        self.ax.plot3D(points[:,0], points[:,1], points[:,2], linestyle, lw=lw, c=c, alpha=alpha)
        
    def plot_coordinate_axis(self, trn, rot, linestyle='-', lw=2, c='r', alpha=1, scale=25, ar=.3):
        
        trn = self.pose_rot.dot(trn) + self.pose_trn
        rot = self.pose_rot.dot(rot)
        
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,0], scale*rot[1,0], scale*rot[2,0], color=c, lw=lw)#, arrow_length_ratio=ar)
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,1], scale*rot[1,1], scale*rot[2,1], color=c, lw=lw)
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,2], scale*rot[1,2], scale*rot[2,2], color=c, lw=lw)
    
    def form_poly(self, tupleList, vertices):
        
        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        return poly3d    
    
    def set_axes_equal(self, plot_center=(0,-25,-52+0), plot_radius=30):

        x_middle = plot_center[0]
        y_middle = plot_center[1]
        z_middle = plot_center[2]

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - 0.85*plot_radius, z_middle + 0.85*plot_radius])
    
    def plot_show(self):
        
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.axis('off')
        self.set_axes_equal(self.view_center, self.view_radius)
        self.set_viewpoint(self.view_elev, self.view_azim)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.savefig(f'./plots/plot_{self.plotnum}.png')
        image = cv2.imread(f'./plots/plot_{self.plotnum}.png')  
        image =cv2.putText(img=np.copy(image), text=self.stage, org=(260,70),fontFace=1, fontScale=1.5, color=(0,200,0), thickness=2)
        cv2.imwrite(f'./plots/plot_{self.plotnum}.png', image)
        self.images.append(imageio.imread(f'./plots/plot_{self.plotnum}.png'))