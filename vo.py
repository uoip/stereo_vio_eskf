import numpy as np 
import cv2
import g2o

from collections import defaultdict

from queue import Queue
from threading import Thread



class Frame(object):
    def __init__(self, image, timestamp, cam=None, pose=None):
        self.image = image
        self.timestamp = timestamp
        self.cam = cam
        self.pose = pose
        
        self.keypoints = None
        self.descriptors = None

    def extract(self, detector, extractor):
        self.keypoints = detector.detect(self.image)
        self.keypoints, self.descriptors = extractor.compute(
            self.image, self.keypoints)


class StereoFrame(object):
    def __init__(self, 
            image_left, image_right, timestamp, cam=None, pose=None):
        self.image = image_left
        self.timestamp = timestamp
        self.cam = None
        self.pose = pose

        self.left = Frame(image_left, timestamp)
        self.right = Frame(image_right, timestamp)

        if cam is not None:
            self.set_camera(cam)            

        self.triangulated_keypoints = []
        self.triangulated_descriptors = []
        self.triangulated_points = []

    def set_camera(self, cam):
        self.cam = cam
        self.left.cam = cam.left_cam
        self.right.cam = cam.right_cam

    def extract(self, detector, extractor):
        t2 = Thread(
            target=self.right.extract, args=(detector, extractor))
        t2.start()
        self.left.extract(detector, extractor)
        t2.join()

    def set_triangulated(self, i, point):
        self.triangulated_keypoints.append(self.left.keypoints[i])
        self.triangulated_descriptors.append(self.left.descriptors[i])
        self.triangulated_points.append(point)
        
    @property
    def keypoints(self):
        return self.left.keypoints
    @property
    def descriptors(self):
        return self.left.descriptors
        


class StereoVO(object):
    def __init__(self, cam, config, show=False):
        self.cam = cam
        self.config = config
        self.show = show

        self.detector = config.detector
        self.extractor = config.extractor
        self.matcher = config.matcher

        self.keyframe = None
        self.candidates = []

    def track(self, frame):
        frame.set_camera(self.cam)
        frame.extract(self.detector, self.extractor)

        try:
            self.triangulate(frame)
        except:
            return None

        if self.keyframe is None:
            self.keyframe = frame
            return None

        timestamp = self.keyframe.timestamp

        pts3, pts2, matches = self.match(self.keyframe, frame)
        if len(matches) < self.config.min_good_matches:
            self.restart_tracking(frame)
            return None

        # transform points in keyfram to current frame
        T, inliers = solve_pnp_ransac(pts3, pts2, frame.cam.intrinsic_matrix)

        if self.show:
            img3 = cv2.drawMatches(
                self.keyframe.image,
                self.keyframe.triangulated_keypoints,
                frame.image,
                frame.keypoints,
                [matches[i] for i in inliers],
                None,
                flags=2)
            cv2.imshow('inliers', img3);cv2.waitKey(1)

        if T is None or len(inliers) < self.config.min_inliers:
            self.restart_tracking(frame)
            return None

        # if self.keyframe.pose is not None:
        #     frame.pose = self.keyframe.pose * T.inverse()
        
        self.candidates.append(frame)
        return (timestamp, frame.timestamp, T.inverse())

    def restart_tracking(self, frame):
        if len(self.candidates) > 4:
            self.keyframe = self.candidates[-2]
        if len(self.candidates) > 0:
            self.keyframe = self.candidates[-1]
        else:
            self.keyframe = frame
        self.candidates.clear()


    def match(self, query_frame, match_frame):
        # TODO: use predicted pose from IMU or motion model
        matches = self.matcher.match(
            np.array(query_frame.triangulated_descriptors), 
            np.array(match_frame.descriptors))

        distances = defaultdict(lambda: float('inf'))
        good = dict()
        for m in matches:
            pt = query_frame.triangulated_keypoints[m.queryIdx].pt
            id = (int(pt[0] / self.config.cell_size), 
                int(pt[1] / self.config.cell_size))
            if m.distance > min(self.config.matching_distance, distances[id]):
                continue
            good[id] = m
            distances[id] = m.distance
        good = list(good.values())

        pts3 = []
        pts2 = []
        for m in good:
            pts3.append(query_frame.triangulated_points[m.queryIdx])
            pts2.append(match_frame.keypoints[m.trainIdx].pt)
        return pts3, pts2, good


    def triangulate(self, frame):
        matches = self.matcher.match(
            frame.left.descriptors, frame.right.descriptors)
        assert len(matches) > self.config.disparity_matches

        good = []
        for i, m in enumerate(matches):
            query_pt = frame.left.keypoints[m.queryIdx].pt
            match_pt = frame.right.keypoints[m.trainIdx].pt
            dx = abs(query_pt[0] - match_pt[0])
            dy = abs(query_pt[1] - match_pt[1])
            if dx == 0:
                continue
            depth = frame.cam.focal_baseline / dx
            if (dy <= self.config.epipolar_range and 
                self.config.min_depth <= depth <= self.config.max_depth):
                good.append(m)

                point = np.zeros(3)
                point[2] = depth
                point[0] = (query_pt[0] - frame.cam.cx) * depth / frame.cam.fx
                point[1] = (query_pt[1] - frame.cam.cy) * depth / frame.cam.fy
                frame.set_triangulated(m.queryIdx, point)

        assert len(good) > self.config.disparity_matches


def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d), np.array(pts), 
            intrinsic_matrix, None, None, None, 
            False, 50, 2.0, 0.9, None)
    if inliers is None or len(inliers) < 5:
        return None, None

    T = g2o.Isometry3d(cv2.Rodrigues(rvec)[0], tvec)
    return T, inliers.ravel()




if __name__ == '__main__':
    import time

    from config import VOConfig, IMUConfig
    from dataset import EuRoCDataset, DataPublisher

    path = 'path/to/your/EuRoC_MAV_dataset/V1_01_easy/'
    dataset = EuRoCDataset(path)

    vo = StereoVO(dataset.cam, VOConfig(), True)

    img_queue = Queue()

    duration = 20
    img_publisher = DataPublisher(
        dataset.stereo, dataset.starttime, img_queue, duration)
    now = time.time()
    img_publisher.start(now)

    timestamps = []
    while True:
        x = img_queue.get()
        if x is None:
            break
        timestamp, (img0, img1) = x
        frame = StereoFrame(img0, img1, timestamp)
        vo.track(frame)
        timestamps.append(timestamp)

    print(f'elapsed {time.time() - now}s')
    print(f'dataset time interval: {timestamps[-1]} -> {timestamps[0]} '
          f'({timestamps[-1]-timestamps[0]}s)')