import numpy as np
import g2o

import time
from queue import Queue
from threading import Thread

from eskf import ESKF
from vo import StereoFrame, StereoVO



class VIO(object):
    def __init__(self, imu_queue, img_queue, cam, imu_config, vo_config, gt_queue=None, viewer=None):
        self.vo_queue = Queue()
        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.gt_queue = gt_queue
        self.stopped = False
        self.viewer = viewer

        self.cam2imu = g2o.Isometry3d(cam.extrinsic_matrix)

        self.vo = StereoVO(cam, vo_config)
        self.eskf = ESKF(self.cam2imu, imu_config)
        
        Thread(target=self.maintain_vo).start()
        Thread(target=self.maintain_imu).start()
        Thread(target=self.maintain_vio).start()
        # Thread(target=self.maintain_gt).start()
        
    def stop(self):
        self.stopped = True
        self.eskf.stop()
        self.vo_queue.put(None)
        self.img_queue.put(None)
        self.imu_queue.put(None)

    def maintain_gt(self):
        while not self.stopped:
            x = self.gt_queue.get()
            if x is None:
                return
            timestamp, data = x
        
    def maintain_vo(self):
        while not self.stopped:
            x = self.img_queue.get()
            while not self.img_queue.empty():  # to reduce delay
                x = self.img_queue.get()
            if x is None:
                self.vo_queue.put(None)
                return
            timestamp, (img0, img1) = x
            if self.viewer is not None:
                self.viewer.update_image(img0)

            frame = StereoFrame(img0, img1, timestamp)
            result = self.vo.track(frame)
            if result is None:
                continue
            # if np.random.random() < 0.7:  # test
            #     continue
            self.vo_queue.put(result)

    def maintain_imu(self):
        while not self.stopped:
            x = self.imu_queue.get()
            if x is None:
                return
            self.eskf.process(x)  # x: (timestamp, (wm, am))

            if self.viewer is not None:
                self.viewer.update_pose(self.eskf.current_pose)

    def maintain_vio(self):
        while not self.stopped:
            x = self.vo_queue.get()
            if x is None:
                return
            self.eskf.measure(x)




if __name__ == '__main__':
    from dataset import EuRoCDataset, DataPublisher
    from config import VOConfig, IMUConfig

    if False:
        from viewer import Viewer
        viewer = Viewer()
    else:
        viewer = None


    path = 'path/to/your/EuRoC_MAV_dataset/MH_01_easy/'
    dataset = EuRoCDataset(path)
    
    img_queue = Queue()
    imu_queue = Queue()
    # gt_queue = Queue()

    vio = VIO(
        imu_queue, img_queue, dataset.cam, 
        IMUConfig(), VOConfig(),
        viewer)

    # If image processing is slow, set ratio a smaller value, make 
    # vo and imu time aligned
    ratio = 0.5
    duration = 100
    imu_publisher = DataPublisher(
        dataset.imu, dataset.starttime, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, dataset.starttime, img_queue, duration, ratio)
    # gt_publisher = DataPublisher(
    #     dataset.groundtruth, dataset.starttime, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    # gt_publisher.start(now)