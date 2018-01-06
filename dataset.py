import numpy as np
import cv2
import os
import time

from collections import defaultdict

from queue import Queue
from threading import Thread, Lock



class GroundTruthReader(object):
    def __init__(self, path, scaler):
        self.scaler = scaler   # convert timestamp from ns to second
        self.path = path

    def parse(self, line):
        # line: timestamp, 
        # p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], 
        # q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], 
        # v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], 
        # b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], 
        # b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        p = np.array(line[1:4])
        q = np.array(line[4:8])
        v = np.array(line[8:11])
        bw = np.array(line[11:14])
        ba = np.array(line[14:17])
        return timestamp, (p, q, v, bw, ba)

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                yield self.parse(line)


class IMUDataReader(object):
    def __init__(self, path, scaler):
        self.scaler = scaler
        self.path = path

    def parse(self, line):
        # line: timestamp [ns],
        # w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],  
        # a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        wm = np.array(line[1:4])
        am = np.array(line[4:7])
        return timestamp, (wm, am)

    @property
    def starttime(self):
        for _ in self:
            return _[0]

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                yield self.parse(line)



class Camera(object):
    def __init__(self, 
            width, height,
            intrinsic_matrix, 
            undistort_rectify=False,
            extrinsic_matrix=None,
            distortion_coeffs=None,
            rectification_matrix=None,
            projection_matrix=None):

        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rectification_matrix = rectification_matrix
        self.projection_matrix = projection_matrix
        self.undistort_rectify = undistort_rectify
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        if undistort_rectify:
            self.remap = cv2.initUndistortRectifyMap(
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs,
                R=self.rectification_matrix,
                newCameraMatrix=self.projection_matrix,
                size=(width, height),
                m1type=cv2.CV_8U)
        else:
            self.remap = None

    def rectify(self, img):
        if self.remap is None:
            return img
        else:
            return cv2.remap(img, *self.remap, cv2.INTER_LINEAR)
        


class StereoCamera(object):
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        self.width = left_cam.width
        self.height = left_cam.height
        self.intrinsic_matrix = left_cam.intrinsic_matrix
        self.extrinsic_matrix = left_cam.extrinsic_matrix
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy
        self.baseline = abs(right_cam.projection_matrix[0, 3] / 
            right_cam.projection_matrix[0, 0])
        self.focal_baseline = self.fx * self.baseline



class ImageReader(object):
    def __init__(self, ids, timestamps, cam):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10   # 10 images ahead of current index
        self.wait = 1.5    # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.wait:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        if not self.thread_started:
            self.thread_started = True
            self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def starttime(self):
        return self.timestamps[0]



class Stereo(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        for l, r in zip(self.left, self.right):
            t0, img0 = l   # t0: timestamp, in second
            t1, img1 = r
            assert abs(t0 - t1) < 0.01, 'unaligned stereo pair'
            yield t0, (img0, img1)

    def __len__(self):
        return len(self.left)

    @property
    def starttime(self):
        return self.left.starttime

        
    

class EuRoCDataset(object):   # Stereo + IMU
    '''
    path example: 'path/to/your/EuRoC Mav Dataset/MH_01_easy'
    '''
    def __init__(self, path, rectify=True):
        self.left_cam = Camera(
            width=752, height=480,
            intrinsic_matrix = np.array([
                [458.654, 0.000000, 367.215], 
                [0.000000, 457.296, 248.375], 
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs = np.array(
                [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.000000]),
            rectification_matrix = np.array([
                [0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]]),
            projection_matrix = np.array([
                [435.2046959714599, 0, 367.4517211914062, 0],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0., 0, 1, 0]]),
            extrinsic_matrix = np.array([
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0]])
        )  
        self.right_cam = Camera(
            width=752, height=480,
            intrinsic_matrix = np.array([
                [457.587, 0.000000, 379.999], 
                [0.000000, 456.134, 255.238], 
                [0.000000, 0.000000, 1.000000]]),
            undistort_rectify=rectify,
            distortion_coeffs = np.array(
                [-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]),
            rectification_matrix = np.array([
                [0.9999633526194376, -0.003625811871560086, 0.007755443660172947],
                [0.003680398547259526, 0.9999684752771629, -0.007035845251224894],
                [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]]),
            projection_matrix = np.array([
                [435.2046959714599, 0, 367.4517211914062, -47.90639384423901],
                [0, 435.2046959714599, 252.2008514404297, 0],
                [0, 0, 1, 0]]),
            extrinsic_matrix = np.array([
                [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
                [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
                [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
                [0.0, 0.0, 0.0, 1.0]])
        ) 
        
        path = os.path.expanduser(path)
        self.left = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam0', 'data')), 
            self.left_cam)
        self.right = ImageReader(
            *self.list_imgs(os.path.join(path, 'mav0', 'cam1', 'data')), 
            self.right_cam)
        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        self.imu = IMUDataReader(os.path.join(
            path, 'mav0', 'imu0', 'data.csv'), 1e-9)
        self.stereo = Stereo(self.left, self.right)
        self.cam = StereoCamera(self.left_cam, self.right_cam)

        self.groundtruth = GroundTruthReader(os.path.join(
            path, 'mav0', 'state_groundtruth_estimate0', 'data.csv'), 1e-9)

        # self.cam2imu = self.left_cam.extrinsic_matrix
            
    @property
    def starttime(self):
        return max(self.imu.starttime, self.stereo.starttime)

    def list_imgs(self, dir):
        xs = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        xs = sorted(xs, key=lambda x:float(x[:-4]))
        timestamps = [float(_[:-4]) * 1e-9 for _ in xs]
        return [os.path.join(dir, _) for _ in xs], timestamps



# simulate the online environment
class DataPublisher(object):
    def __init__(self, dataset, dataset_starttime, 
            out_queue, duration=float('inf'), ratio=1.): 
        self.dataset = dataset
        self.dataset_starttime = dataset_starttime
        self.out_queue = out_queue
        self.duration = duration
        self.ratio = ratio
        self.starttime = None
        self.started = False
        self.stopped = False

        self.publish_thread = Thread(target=self.publish)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()

    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
        self.out_queue.put(None)

    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                timestamp, data = next(dataset)
            except StopIteration:
                self.out_queue.put(None)
                return

            interval = timestamp - self.dataset_starttime
            if interval < 0:
                continue
            while (time.time() - self.starttime) * self.ratio < interval + 1e-3:
                time.sleep(1e-3)   # assumption: data frequency < 1000hz
                if self.stopped:
                    return
            self.out_queue.put((timestamp, data))

            if interval > self.duration:
                self.out_queue.put(None)
                return




if __name__ == '__main__':
    path = 'path/to/your/EuRoC_MAV_dataset/MH_01_easy/'
    dataset = EuRoCDataset(path, rectify=True)

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    duration = 0
    imu_publisher = DataPublisher(
        dataset.imu, dataset.starttime, imu_queue, duration)
    img_publisher = DataPublisher(
        dataset.stereo, dataset.starttime, img_queue, duration)
    gt_publisher = DataPublisher(
        dataset.groundtruth, dataset.starttime, gt_queue, duration)


    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    # gt_publisher.start(now)


    def print_msg(in_queue, source):
        while True:
            x = in_queue.get()
            if x is None:
                return
            timestamp, data = x
            print(timestamp, source)
    t2 = Thread(target=print_msg, args=(imu_queue, 'imu'))
    t3 = Thread(target=print_msg, args=(gt_queue, 'groundtruth'))
    t2.start()
    t3.start()

    timestamps = []
    while True:
        x = img_queue.get()
        if x is None:
            break
        timestamp, (img0, img1) = x
        print(timestamp, 'image')
        # cv2.imshow('left', np.hstack([img0, img1]))
        # cv2.waitKey(1)
        timestamps.append(timestamp)

    imu_publisher.stop()
    img_publisher.stop()
    gt_publisher.stop()
    t2.join()
    t3.join()

    print()
    print(f'elapsed {time.time() - now}s')
    print(f'dataset time interval: {timestamps[-1]} -> {timestamps[0]}'
        f'  ({timestamps[-1]-timestamps[0]}s)')
    print('Please check if IMU and image are time aligned')