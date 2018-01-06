import numpy as np 
import cv2



class VOConfig(object):
    def __init__(self, feature='GFTT'):
        self.epipolar_range = 1.0   # pixel
        self.max_depth = 20    # meter
        self.min_depth = 0.01  # meter

        if feature == 'GFTT':
            self.detector = cv2.GFTTDetector_create(
                maxCorners=500, qualityLevel=0.01, minDistance=9, blockSize=9)
            self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif feature == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=8, 
                edgeThreshold=31, patchSize=31)
            self.extractor = self.detector
        else:
            raise NotImplementedError
        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.dist_std_scale = 0.4

        self.disparity_matches = 40
        self.min_good_matches = 40
        self.matching_distance = 25
        self.min_inliers = 40
        self.restart_tracking = 3
        
        self.max_update_time = 0.45

        self.cell_size = 15
        



class IMUConfig(object):
    def __init__(self):
        self.gravity = np.array([9.80665, 0, 0])
        self.imu_max_reading_time = 0.06     # sec
        self.imu_max_reading_accel = 50.0    # m / sec^2
        self.imu_max_reading_gyro = 50.0     # rad / sec
        self.max_update_time = 0.5           # sec
        self.max_vo_imu_delay = 0.1          # sec
        self.update_error_bound = 0.1

        self.accel_noise_density = 2.0000e-3
        self.accel_random_walk = 3.0000e-3;
        self.gyro_noise_density = 1.6968e-4
        self.gyro_random_walk = 1.9393e-5
        self.initial_process_covariance = 1e-8
        self.state_buffer_size_in_sec = 60   # How old of state obj will be eliminated (in sec)

        self.using_fixed_vo_covriance = True
        self.vo_fixedstd_q = 1e-4
        self.vo_fixedstd_p = 1e-5
        self.vo_fixedcov_q = self.vo_fixedstd_q * self.vo_fixedstd_q
        self.vo_fixedcov_p = self.vo_fixedstd_p * self.vo_fixedstd_p