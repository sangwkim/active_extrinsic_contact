#!/usr/bin/env python
"""
This code is the simulation environment for the reinforcement learning policy training
"""

import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class insertion_simulator:
    def __init__(self,
                 object_list,
                 max_step,
                 noise_on=True,
                 mode_detection_prob=0.05,
                 false_positive_prob=0.05,
                 false_estimation_prob=0.05,
                 pose_noise=np.array([0.4, 0.4, 0.2 / 180 * np.pi]),
                 estimation_noise=np.array([4, 4. / 180 * np.pi]),
                 mode_detection_noise=np.array([12., 15. / 180 * np.pi]),
                 false_estimation_noise=np.array([20., 60. / 180 * np.pi]),
                 max_error=np.array([12, 12, 15. / 180 * np.pi]),
                 success_reward=2.,
                 isActive=True):

        self.object_list = object_list

        self.noise_on = noise_on
        self.mode_detection_prob = mode_detection_prob
        self.false_positive_prob = false_positive_prob
        self.false_estimation_prob = false_estimation_prob
        self.false_estimation_noise = false_estimation_noise
        self.estimation_noise = estimation_noise
        self.mode_detection_noise = mode_detection_noise
        self.pose_noise = pose_noise

        self.isActive = isActive
        self.hole_circle = np.array([[19.75, 19.75], [-19.75, 19.75],
                                     [-19.75, -19.75], [19.75, -19.75]])
        self.hole_hexagon = np.array([[17.406, 19.75], [-17.406, 19.75],
                                      [-17.406, -19.75], [17.406, -19.75]])
        self.hole_ellipse = np.array([[19.75, 27.25], [-19.75, 27.25],
                                      [-19.75, -27.25], [19.75, -27.25]])
        self.hole_circle_tight = np.array(
            [[19.75 * np.cos(th), 19.75 * np.sin(th)]
             for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
        self.hole_hexagon_tight = (15.156 + 2.25) / 15.156 * np.array(
            [[15.156, -8.75], [15.156, 8.75], [0, 17.5], [-15.156, 8.75],
             [-15.156, -8.75], [0, -17.5]])
        self.hole_ellipse_tight = np.array(
            [[19.75 * np.cos(th), 27.25 * np.sin(th)]
             for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
        self.hole_rectangle = np.array([[19.75, 27.25], [-19.75, 27.25],
                                        [-19.75, -27.25], [19.75, -27.25]])
        self.object_circle = np.array(
            [[17.5 * np.cos(th), 17.5 * np.sin(th)]
             for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
        self.object_hexagon = np.array([[15.156, -8.75], [15.156, 8.75],
                                        [0, 17.5], [-15.156, 8.75],
                                        [-15.156, -8.75], [0, -17.5]])
        self.object_ellipse = np.array(
            [[17.5 * np.cos(th), 25. * np.sin(th)]
             for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
        self.object_rectangle = np.array([[17.5, 25.], [-17.5, 25.],
                                          [-17.5, -25.], [17.5, -25.]])
        self.hole_Dict = {
            'circle': self.hole_circle,
            'hexagon': self.hole_hexagon,
            'ellipse': self.hole_ellipse,
            'rectangle': self.hole_rectangle,
            'circle_tight': self.hole_circle_tight,
            'hexagon_tight': self.hole_hexagon_tight,
            'ellipse_tight': self.hole_ellipse_tight,
        }
        self.object_Dict = {
            'circle': self.object_circle,
            'hexagon': self.object_hexagon,
            'ellipse': self.object_ellipse,
            'rectangle': self.object_rectangle,
            'circle_tight': self.object_circle,
            'hexagon_tight': self.object_hexagon,
            'ellipse_tight': self.object_ellipse,
        }

        self.success_reward = success_reward
        self.step_penalty = success_reward / max_step

        self.max_step = max_step
        self.max_error = max_error

        self.object_pose = np.zeros(3)

    def generate_random_polygon(self,
                                scale=10.,
                                num_points=30,
                                tight_hole=True,
                                offset=2.25):

        points = np.clip(np.random.normal(0, 1, (num_points, 2)), -1.5, 1.5)
        points[:, 0] *= np.random.uniform(5, 20)
        points[:, 1] *= np.random.uniform(5, 20)
        #points = scale * np.clip(np.random.normal(0,1,(num_points,2)),-2,2)
        hull = ConvexHull(points)
        self.object_polygon = np.array(
            [points[vertex] for vertex in hull.vertices])
        if tight_hole:
            self.hole_polygon = self.polygon_offset(self.object_polygon,
                                                    offset)
        else:
            min_x = np.min(self.object_polygon[:, 0] - offset)
            max_x = np.max(self.object_polygon[:, 0] + offset)
            min_y = np.min(self.object_polygon[:, 1] - offset)
            max_y = np.max(self.object_polygon[:, 1] + offset)
            self.hole_polygon = np.array([[min_x, min_y], [max_x, min_y],
                                          [max_x, max_y], [min_x, max_y]])

    def polygon_offset(self, polygon, offset):

        equations = []
        for i in range(len(polygon)):
            A = np.array([[polygon[i - 1][0], polygon[i - 1][1]],
                          [polygon[i][0], polygon[i][1]]])
            Y = np.array([[1], [1]])
            X = np.linalg.inv(A).dot(Y)
            equation = np.array([X[0, 0], X[1, 0], -1])
            equation[2] -= np.linalg.norm(equation[[0, 1]]) * offset
            equations.append(equation)

        points_offset = []
        for i in range(len(equations)):
            A = np.array([[equations[i - 1][0], equations[i - 1][1]],
                          [equations[i][0], equations[i][1]]])
            Y = np.array([[-equations[i - 1][2]], [-equations[i][2]]])
            X = np.linalg.inv(A).dot(Y)
            points_offset.append(X.squeeze())
        points_offset = np.array(points_offset)

        return points_offset

    def pose_convert(self, object_pose):
        dx, dy, th = object_pose
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return R.dot(self.object_polygon.T).T + np.array([dx, dy])

    def reset(self, object_name=None):

        if object_name == None:
            if self.object_list == None:
                object_name = "random"
            else:
                object_name = np.random.choice(self.object_list)

        if object_name == "random":
            if np.random.uniform() < 1.:
                self.generate_random_polygon(tight_hole=True)
            else:
                self.generate_random_polygon(tight_hole=False)
        else:
            self.hole_polygon = self.hole_Dict[object_name]
            self.object_polygon = self.object_Dict[object_name]

        self.step_num = 0

        while True:
            self.object_pose = np.random.uniform(-1, 1,
                                                 size=3) * self.max_error
            if self.noise_on:
                ct_line, Success, tilt_side = self.insert(
                    self.object_pose + np.random.normal(scale=self.pose_noise))
            else:
                ct_line, Success, tilt_side = self.insert(self.object_pose)
            if Success == False:
                break

        self.object_pose_init = self.object_pose.copy()
        self.rot = np.array([[
            np.cos(self.object_pose_init[2]), -np.sin(self.object_pose_init[2])
        ], [
            np.sin(self.object_pose_init[2]),
            np.cos(self.object_pose_init[2])
        ]])

        if ct_line is None or tilt_side == 0:
            if np.random.uniform() < self.false_positive_prob:
                r = np.random.normal(scale=15)
                th = np.random.uniform(-np.pi, np.pi)
                tilt_side = np.random.choice([-1, 1], p=[0.5, 0.5])
                mode_detect = np.random.choice([-1, 1], p=[0.8, 0.2])
                return np.array(
                    [r, np.cos(th),
                     np.sin(th), tilt_side,
                     mode_detect]), self.object_pose, object_name
            else:
                return np.array([0, 0, 0, 0, 0]), self.object_pose, object_name

        p_1, p_2 = ct_line[0], ct_line[1]

        r = ((self.object_pose[0] - p_1[0])*(p_2[1]-p_1[1]) - (self.object_pose[1] - p_1[1])*(p_2[0] - p_1[0])) \
                         / ((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)**0.5
        r = abs(r)
        th = np.arctan2(
            -(p_2[0] - p_1[0]) *
            ((p_1[0] - self.object_pose[0]) * (p_2[1] - p_1[1]) -
             (p_1[1] - self.object_pose[1]) * (p_2[0] - p_1[0])),
            (p_2[1] - p_1[1]) *
            ((p_1[0] - self.object_pose[0]) * (p_2[1] - p_1[1]) -
             (p_1[1] - self.object_pose[1]) * (p_2[0] - p_1[0])))
        th -= self.object_pose[2]

        # false negative rate follows sigmoid function w/
        # 5%  at r*tilt_side = -10
        # 50% at r*tilt_side = -25
        k = np.log(19) / 15
        if np.random.uniform() > 1 / (1 + np.exp(-k * (r * tilt_side + 25))):
            return np.array([0, 0, 0, 0, 0]), self.object_pose, object_name

        if self.noise_on:
            if np.random.uniform() < self.mode_detection_prob:
                mode_detect = 1.
                if np.random.uniform() < self.false_estimation_prob:
                    noise = self.false_estimation_noise.copy()
                else:
                    noise = self.mode_detection_noise.copy()
            else:
                mode_detect = -1.
                if np.random.uniform() < self.false_estimation_prob:
                    noise = self.false_estimation_noise.copy()
                else:
                    noise = self.estimation_noise.copy()

            r += np.random.normal(scale=noise[0])
            th += np.random.normal(scale=noise[1])
            if r < 0:
                r *= -1
                th += np.pi
                tilt_side *= -1
        else:
            mode_detect = 0

        return np.array([r, np.cos(th),
                         np.sin(th), tilt_side,
                         mode_detect]), self.object_pose, object_name

    def step(self, action):

        error_norm_ = np.linalg.norm(self.object_pose / self.max_error)

        self.step_num += 1

        self.object_pose[:2] = self.object_pose_init[:2] + self.rot.dot(
            action[:2])
        self.object_pose[2] = self.object_pose_init[2] + action[2]
        if self.noise_on:
            ct_line, Success, tilt_side = self.insert(
                self.object_pose + np.random.normal(scale=self.pose_noise))
        else:
            ct_line, Success, tilt_side = self.insert(self.object_pose)

        error_norm = np.linalg.norm(self.object_pose / self.max_error)

        done = False

        reward = (error_norm_ - error_norm) / (3**0.5)
        if Success == True:
            reward += self.success_reward
            done = True
        else:
            reward -= self.step_penalty

        if self.step_num >= self.max_step:
            done = True

        if ct_line is None or tilt_side == 0:
            if np.random.uniform() < self.false_positive_prob:
                r = np.random.normal(scale=15)
                th = np.random.uniform(-np.pi, np.pi)
                tilt_side = np.random.choice([-1, 1], p=[0.5, 0.5])
                mode_detect = np.random.choice([-1, 1], p=[0.8, 0.2])
                return np.array(
                    [r, np.cos(th),
                     np.sin(th), tilt_side,
                     mode_detect]), self.object_pose, reward, done
            else:
                return np.array([0, 0, 0, 0,
                                 0]), self.object_pose, reward, done

        p_1, p_2 = ct_line[0], ct_line[1]
        r = ((self.object_pose[0] - p_1[0])*(p_2[1]-p_1[1]) - (self.object_pose[1] - p_1[1])*(p_2[0] - p_1[0])) \
                         / ((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)**0.5
        r = abs(r)
        th = np.arctan2(
            -(p_2[0] - p_1[0]) *
            ((p_1[0] - self.object_pose[0]) * (p_2[1] - p_1[1]) -
             (p_1[1] - self.object_pose[1]) * (p_2[0] - p_1[0])),
            (p_2[1] - p_1[1]) *
            ((p_1[0] - self.object_pose[0]) * (p_2[1] - p_1[1]) -
             (p_1[1] - self.object_pose[1]) * (p_2[0] - p_1[0])))
        th -= self.object_pose[2]

        if self.noise_on:
            if np.random.uniform() < self.mode_detection_prob:
                mode_detect = 1.
                if np.random.uniform() < self.false_estimation_prob:
                    noise = self.false_estimation_noise.copy()
                else:
                    noise = self.mode_detection_noise.copy()
            else:
                mode_detect = -1.
                if np.random.uniform() < self.false_estimation_prob:
                    noise = self.false_estimation_noise.copy()
                else:
                    noise = self.estimation_noise.copy()

            r += np.random.normal(scale=noise[0])
            th += np.random.normal(scale=noise[1])
            if r < 0:
                r *= -1
                th += np.pi
                tilt_side *= -1
        else:
            mode_detect = 0

        return np.array([r, np.cos(th),
                         np.sin(th), tilt_side,
                         mode_detect]), self.object_pose, reward, done

    def insert(self, object_pose):
        self.obj_poly = self.pose_convert(object_pose)

        pts_inside = []
        for i, pt in enumerate(self.obj_poly):
            if Polygon(self.hole_polygon).contains(Point(pt)):
                pts_inside.append(i)
        if len(pts_inside) == 0:
            return None, False, 0
        if len(pts_inside) == len(self.obj_poly):
            return None, True, 0

        in_conseq_pts = []
        exclude_list = []
        for idx in pts_inside:
            if not idx in exclude_list:
                exclude_list.append(idx)
                temp = [idx]
                for i in range(len(self.obj_poly)):
                    if (idx - i - 1) % len(self.obj_poly) in pts_inside:
                        exclude_list.append((idx - i - 1) % len(self.obj_poly))
                        temp.insert(0, (idx - i - 1) % len(self.obj_poly))
                    else:
                        break
                for i in range(len(self.obj_poly)):
                    if (idx + i + 1) % len(self.obj_poly) in pts_inside:
                        exclude_list.append((idx + i + 1) % len(self.obj_poly))
                        temp.append((idx + i + 1) % len(self.obj_poly))
                    else:
                        break
                in_conseq_pts.append(temp)

        ct_line = []
        center_dist_min = np.inf

        for pts in in_conseq_pts:

            po_1 = self.obj_poly[pts[0]]
            po_2 = self.obj_poly[pts[0] - 1]
            r_ = 0
            for j in range(len(self.hole_polygon)):
                ph_1 = self.hole_polygon[j - 1]
                ph_2 = self.hole_polygon[j]
                if not (po_1[0] - po_2[0]) * (ph_1[1] - ph_2[1]) - (
                        po_1[1] - po_2[1]) * (ph_1[0] - ph_2[0]) == 0:
                    r = (ph_2[0]*(ph_1[1]-ph_2[1]) - ph_2[1]*(ph_1[0]-ph_2[0]) - po_2[0]*(ph_1[1]-ph_2[1]) + po_2[1]*(ph_1[0]-ph_2[0])) \
                        / ((po_1[0]-po_2[0])*(ph_1[1]-ph_2[1]) - (po_1[1]-po_2[1])*(ph_1[0]-ph_2[0]))
                    if r_ < r < 1:
                        r_ = r
                        p_1 = r * po_1 + (1 - r) * po_2

            po_1 = self.obj_poly[pts[-1]]
            po_2 = self.obj_poly[(pts[-1] + 1) % len(self.obj_poly)]
            r_ = 0
            for j in range(len(self.hole_polygon)):
                ph_1 = self.hole_polygon[j - 1]
                ph_2 = self.hole_polygon[j]
                if not (po_1[0] - po_2[0]) * (ph_1[1] - ph_2[1]) - (
                        po_1[1] - po_2[1]) * (ph_1[0] - ph_2[0]) == 0:
                    r = (ph_2[0]*(ph_1[1]-ph_2[1]) - ph_2[1]*(ph_1[0]-ph_2[0]) - po_2[0]*(ph_1[1]-ph_2[1]) + po_2[1]*(ph_1[0]-ph_2[0])) \
                        / ((po_1[0]-po_2[0])*(ph_1[1]-ph_2[1]) - (po_1[1]-po_2[1])*(ph_1[0]-ph_2[0]))
                    if r_ < r < 1:
                        r_ = r
                        p_2 = r * po_1 + (1 - r) * po_2

            distance_to_obj_center = ((object_pose[0] - p_1[0])*(p_2[1]-p_1[1]) - (object_pose[1] - p_1[1])*(p_2[0] - p_1[0])) \
                         / ((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)**0.5
            distance_to_an_inside_point = ((po_1[0] - p_1[0])*(p_2[1]-p_1[1]) - (po_1[1] - p_1[1])*(p_2[0] - p_1[0])) \
                         / ((p_2[0]-p_1[0])**2+(p_2[1]-p_1[1])**2)**0.5

            if self.isActive:
                if abs(distance_to_obj_center) < center_dist_min:
                    ct_line = np.array([p_1, p_2])
                    center_dist_min = abs(distance_to_obj_center)
                    if distance_to_an_inside_point * distance_to_obj_center > 0:
                        tilt_side = +1
                    else:
                        tilt_side = -1
            else:
                if distance_to_an_inside_point * distance_to_obj_center > 0:
                    if abs(distance_to_obj_center) > 5:
                        if abs(distance_to_obj_center) < center_dist_min:
                            ct_line = np.array([p_1, p_2])
                            center_dist_min = abs(distance_to_obj_center)
                            if distance_to_an_inside_point * distance_to_obj_center > 0:
                                tilt_side = +1
                            else:
                                tilt_side = -1

        return ct_line, False, tilt_side

    def plot_hole_object(self, object_pose, verbose=True):
        obj_poly = self.pose_convert(object_pose)
        ct_line, done, tilt_side = self.insert((object_pose))
        if verbose:
            print(ct_line)
        plt.figure()
        plt.fill(self.hole_polygon[:, 0],
                 self.hole_polygon[:, 1],
                 facecolor="none",
                 edgecolor='black')
        plt.fill(obj_poly[:, 0],
                 obj_poly[:, 1],
                 facecolor="none",
                 edgecolor='black')
        if not ct_line is None:
            plt.plot(ct_line[:, 0], ct_line[:, 1])
        plt.axis('equal')
        plt.xlim([-50, 50])
        plt.ylim([-30, 30])
        plt.grid()
        plt.show()


if __name__ == "__main__":
    isim = insertion_simulator("rectangle", isActive=True)
    isim.plot_hole_object(np.array([-10, -10, -10 * np.pi / 180]))