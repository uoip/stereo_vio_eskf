import numpy as np
import g2o
import time


class ESKFState(object):
    def __init__(self, config):
        # State: p, v, q, ba, bw
        self.q  = np.array([1.0, 0.0, 0.0, 0.0])   # [w, x, y, z]
        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.ba = np.zeros(3)
        self.bw = np.zeros(3)
        self.P = np.identity(15) * config.initial_process_covariance

        # Reading
        self.wm = np.zeros(3)   # angular_velocity
        self.am = np.zeros(3)   # linear_acceleration
        self.timestamp = None   # in second



# ESKF: Error State Kalman Filter
class ESKF(object):
    def __init__(self, cam2imu, config):
        self.cam2imu = cam2imu
        self.config = config

        self.state_buffer = []
        self.current_pose = None

        self.is_processing = False
        self.is_measuring = False

        # utils
        self.Gc = np.zeros((15, 12))
        self.Gc[3:, :] = np.identity(12)
        self.Gc.setflags(write=False)

        self.H = np.zeros((6, 15))
        self.H[ :3,  :3] = np.identity(3)
        self.H[3:6, 6:9] = np.identity(3)
        self.H.setflags(write=False)

        self.cam2imu_t = self.cam2imu.translation()
        self.cam2imu_q = self.cam2imu.rotation()

    def init(self, p, q, timestamp):
        init_state = ESKFState(self.config)
        init_state.p = np.zeros(3)#p
        init_state.q = np.array([1.0, 0.0, 0.0, 0.0])#q
        init_state.timestamp = timestamp
        self.state_buffer.append(init_state)

    def initialize(self, state, use_prev_state=False):
        if use_prev_state and len(self.state_buffer):
            init_state = self.state_buffer[-1]
            init_state.timestamp = state.timestamp
            self.state_buffer.append(init_state)
            return
        init_state = ESKFState(self.config)
        init_state.timestamp = state.timestamp
        init_state = state
        self.state_buffer.append(init_state)

    def process(self, imu_msg):
        while self.is_measuring:
            time.sleep(1e-4)
        self.is_processing = True

        timestamp, data = imu_msg
        state_new = ESKFState(self.config)
        state_new.wm = data[0]
        state_new.am = data[1]
        state_new.timestamp = timestamp

        if len(self.state_buffer) == 0:
            self.initialize(state_new)
            return

        # ========================
        # === ESKF Propagation ===
        # ========================

        state_old = self.state_buffer[-1]
        dt = state_new.timestamp - state_old.timestamp

        if dt < 0:
            return

        if dt >= self.config.imu_max_reading_time:
            print('[Warning] The interval between readings is too big')
            raise RuntimeError
            # self.initialize(state_new, True)
            # return

        if np.linalg.norm(state_new.am) >= self.config.imu_max_reading_accel:
            print('[Warning] The accelerometer readings is too big')
            state_new.am = state_old.am

        if np.linalg.norm(state_new.wm) >= self.config.imu_max_reading_gyro:
            print('[Warning] The gyroscope readings is too big')
            state_new.wm = state_old.wm

        self.propagate(state_old, state_new)
        self.state_buffer.append(state_new)
        self.shrink_buffer()

        print(state_new.timestamp, '\n p   ', state_new.p, state_new.p-self.state_buffer[0].p, '\n q   ', state_new.q, '\n ba  ', state_new.ba, '\n bw  ', state_new.bw)
        print(' v   ', state_new.v)
            
        self.current_pose = g2o.Isometry3d(
            g2o.Quaternion(state_new.q), state_new.p)
        self.is_processing = False


    def measure(self, vo_msg):
        timestamp_from, timestamp_to, vo_T = vo_msg
        assert timestamp_from < timestamp_to

        # if timestamp_from < self.state_buffer[0].timestamp + 3:  # test
        #     return

        vo_q = vo_T.rotation()
        vo_t = vo_T.translation()

        tic = time.time()
        while (self.is_processing or len(self.state_buffer) < 2 or 
            self.state_buffer[-1].timestamp < timestamp_to):
            time.sleep(1e-4)
            if time.time() - tic > 1:
                print('[Update] Something went wrong with IMU !!')
                self.is_measuring = False
                return

        self.is_measuring = True
        if self.state_buffer[0].timestamp > timestamp_from:
            print('[Update] Keyframe time is older than the first state,'
                'drop this update')
            self.is_measuring = False
            return

        if self.state_buffer[-1].timestamp - timestamp_to > self.config.max_vo_imu_delay:
            print('[Update] Delay between imu and vo is too large, drop this update')
            self.is_measuring = False
            return
            
        state_from = self.get_closest_state(timestamp_from)
        state_to = self.get_closest_state(timestamp_to)
        if state_from is None or state_to is None:
            print('[Update] Nearest state not found')
            self.is_measuring = False
            return

        from_q = quaternion(state_from.q)
        to_q = quaternion(state_to.q)

        odom_q = self.cam2imu_q * vo_q * self.cam2imu_q.conjugate()
        odom_q.normalize()
        odom_t = self.cam2imu_q * vo_t + self.cam2imu_t

        meas_q = from_q * odom_q
        meas_q.normalize()
        meas_p = from_q * odom_t + state_from.p

        R = np.identity(6)
        R[:3, :3] = self.config.vo_fixedcov_p * np.identity(3)
        R[3:, 3:] = self.config.vo_fixedcov_q * np.identity(3)

        # ========================
        # === Update procedure ===
        # ========================
        # ## Step1: Calculate the residual

        residual_p = meas_p - state_to.p
        residual_q = meas_q * to_q.conjugate()
        
        residual_th = 2 * residual_q.vec() / residual_q.w()   # q ~= [1, 0.5*theta]
        residual = np.concatenate([residual_p, residual_th])

        # ## Step2: Compute the Innovation, Kalman gain and Correction state
        S = self.H @ state_to.P @ self.H.T + R    # (6, 6)
        S_inv = np.linalg.inv(S)

        K = state_to.P @ self.H.T @ S_inv    # (15, 6)
        update = K.dot(residual)

        # ##Step3: Update the state
        state_to.p  += update[0:3]
        state_to.v  += update[3:6]
        state_to.ba += update[9:12]
        state_to.bw += update[12:15]

        q_update = np.array([1, *(0.5 * update[6:9])])
        state_to.q = quatmat(state_to.q).dot(q_update)
        state_to.q /= np.linalg.norm(state_to.q)

        # covariance: P k+1|k+1 = (I d −KH)P k+1|k (I d −KH) T +KRK T
        KH = np.identity(15) - K @ self.H    # (15, 15)
        state_to.P = KH @ state_to.P @ KH.T + K @ R @ K.T
        # Make sure P stays symmetric
        state_to.P = (state_to.P + state_to.P.T) / 2.

        # ##Step4: Repropagate State & Covariance
        assert state_to in self.state_buffer
        idx = self.state_buffer.index(state_to)
        if idx < len(self.state_buffer) - 1:
            for i in range(idx, len(self.state_buffer) - 1):
                self.propagate(self.state_buffer[i], self.state_buffer[i+1])

        self.current_pose = g2o.Isometry3d(
            g2o.Quaternion(self.state_buffer[-1].q), self.state_buffer[-1].p)
        self.is_measuring = False
        

    def propagate(self, state_old, state_new):           # update state_new
        dt = state_new.timestamp - state_old.timestamp

        # ## Step1: Propagate the bias (under guassian assumption, they should be constant)
        state_new.ba = state_old.ba
        state_new.bw = state_old.bw

        # state_new.ba = np.clip(state_new.ba, -20, 20)
        # state_new.bw = np.clip(state_new.bw, -10, 10)

        # ## Step2: Estimate the a and w by IMU reading
        ah_new = state_new.am - state_new.ba
        ah_old = state_old.am - state_old.ba
        wh_new = state_new.wm - state_new.bw
        wh_old = state_old.wm - state_old.bw
        w_new  = np.array([0.0, *wh_new])
        w_old  = np.array([0.0, *wh_old])
        omega_new = quatmat(w_new)
        omega_old = quatmat(w_old)
        w_ave = (w_new + w_old) / 2.

        # ## Step3: Propagate q v and p
        # For q - First order quaternion integration (reference to MSF-EKF)
        # We used Hamilton definition for quaternion: q
        div = 1
        exp_w = np.array([1.0, 0.0, 0.0, 0.0])
        w_ave = w_ave * 0.5 * dt
        
        for i in range(1, 4):
            div *= i
            exp_w += w_ave / div
            w_ave = quatmat(w_ave).dot(w_ave)

        state_new.q = quatmat(state_old.q).dot(
            exp_w + (omega_old.dot(w_new) - omega_new.dot(w_old)*dt*dt/48.0))
        state_new.q /= np.linalg.norm(state_new.q)

        # For v
        # Get the average value between the old and new state
        a = (rotmat(state_new.q).dot(ah_new) + rotmat(state_old.q).dot(ah_old))/2.
        # a = state_new.am
        state_new.v = state_old.v + (a - self.config.gravity) * dt

        # For p
        state_new.p = state_old.p + (state_old.v + state_new.v)/2. * dt

        # ## Step4: Propagate state covariance matrix
        # State: p, v, q, ba, bw
        #
        # Key equation:
        # P_new = Fd*P_old*Fd' + Gc*Qd*Gc'
        #
        # Fd was computed based on MSF-EKF (ETHZ) method which was reference to:
        # Stephan Weiss and Roland Siegwart.
        # Real-Time Metric State Estimation for Modular Vision-Inertial Systems.
        # IEEE International Conference on Robotics and Automation. Shanghai, China, 2011
        #
        # Fi and Q matrix adopt the equation listed in:
        # Quaternion kinematics for the error-state KF, Joan Sola, 2016
        # (And also: Stochastic Models, Estimation and Control, p.171)

        # STD of noise sources (all in continuous state)
        na  = self.config.accel_noise_density
        nba = self.config.accel_random_walk
        nw  = self.config.gyro_noise_density
        nbw = self.config.gyro_random_walk

        # Bias corrected IMU readings
        a_sk = skew(ah_new)
        w_sk = skew(wh_new)
        w_sk2 = w_sk @ w_sk
        eye = np.identity(3)
        C_eq = rotmat(state_new.q)

        # Construct the matrix Fd
        dt2_2 = dt * dt / 2.
        dt3_6 = dt2_2 * dt / 3.
        dt4_24 = dt3_6 * dt / 4.
        dt5_120 = dt4_24 * dt / 5.

        Ca3 = C_eq @ a_sk
        A = Ca3 @ (-dt2_2 * eye + dt3_6 * w_sk - dt4_24 * w_sk2)
        B = Ca3 @ (dt3_6 * eye - dt4_24 * w_sk + dt5_120 * w_sk2)
        D = -A 
        E = eye - dt * w_sk + dt2_2 * w_sk2
        F = -dt * eye + dt2_2 * w_sk - dt3_6 * w_sk2
        C = Ca3 @ F

        Fd = np.identity(15)
        Fd[:3,    3:6] = dt * eye
        Fd[:3,    6:9] = A
        Fd[:3,   9:12] = -C_eq * dt2_2
        Fd[:3,  12:15] = B

        Fd[3:6,   6:9] = C
        Fd[3:6,  9:12] = -C_eq * dt
        Fd[3:6, 12:15] = D

        Fd[6:9,   6:9] = E
        Fd[6:9, 12:15] = F

        # Construct the matrix Qd
        Qd = np.identity(12)
        Qd[:3,     :3] = nba*nba * dt*dt * eye
        Qd[3:6,   3:6] = nbw*nbw * dt*dt * eye
        Qd[6:9,   6:9] = na*na * dt * eye
        Qd[9:12, 9:12] = nw*nw * dt * eye

        # Propagate the covariance matrix P_new = Fd*P_old*Fd' + Gc*Qd*Gc'
        Gc = self.Gc
        state_new.P = Fd @ state_old.P @ Fd.T + Gc @ Qd @ Gc.T


    def shrink_buffer(self):
        if len(self.state_buffer) > 0:
            now = self.state_buffer[-1].timestamp
        for state in self.state_buffer:
            if now - state.timestamp > self.config.state_buffer_size_in_sec:
                self.state_buffer.remove(state)
            else:
                break

    def get_closest_state(self, timestamp):
        interval = float('inf')
        closest = None
        for state in self.state_buffer[::-1]:
            dt = abs(timestamp - state.timestamp)
            if dt < interval and dt < self.config.imu_max_reading_time / 2:
                interval = dt
                closest = state
            elif state.timestamp < timestamp:
                break
        return closest




def skew(vec):
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def quatmat(q):
    # left-quaternion-product matrix
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w]])

def rotmat(q):
    q = g2o.Quaternion(q)
    q.normalize()
    return q.matrix()

def quaternion(q):
    q = g2o.Quaternion(q)
    q.normalize()
    return q