# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    """A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space (x, y,
    a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step (vectorized version).
        update: Run the Kalman filter correction step.
        gating_distance: Compute the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.
        """
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create a track from an unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            mean (np.ndarray): Mean vector (8-dimensional) of the new track. Unobserved velocities are initialized to 0.
            covariance (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time
                step.

        Returns:
            mean (np.ndarray): Mean vector of the predicted state.
            covariance (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            confidence (float, optional): Detection confidence; when set, scales measurement noise by max(1 -
                confidence, 0.05) (NSA-Kalman).

        Returns:
            mean (np.ndarray): Projected mean of the given state estimate.
            covariance (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:  # NSA-Kalman: scale measurement noise by detection confidence (StrongSORT)
            innovation_cov *= max(1.0 - float(confidence), 0.05)

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.random.rand(10, 8)  # 10 object states
            >>> covariance = np.random.rand(10, 8, 8)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float | None = None
    ):
        """Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.
            confidence (float, optional): Detection confidence; when set, scales measurement noise by max(1 -
                confidence, 0.05) (NSA-Kalman).

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        kalman_gain = np.linalg.solve(projected_cov, np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (np.ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (np.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (np.ndarray): An (N, 4) matrix of N measurements, each in format (x, y, a, h) where (x, y) is
                the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to box center position
                only.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where (x, y)
    is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities. The
    object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step in a vectorized manner.
        update: Run the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and
                height.

        Returns:
            mean (np.ndarray): Mean vector (8 dimensional) of the new track. Unobserved velocities are initialized to 0.
            covariance (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.      0.      0.      0.      0.      0.      0.      0.    ]
             [ 0.     16.      0.      0.      0.      0.      0.      0.    ]
             [ 0.      0.      4.      0.      0.      0.      0.      0.    ]
             [ 0.      0.      0.     16.      0.      0.      0.      0.    ]
             [ 0.      0.      0.      0.      1.5625  0.      0.      0.    ]
             [ 0.      0.      0.      0.      0.      6.25    0.      0.    ]
             [ 0.      0.      0.      0.      0.      0.      1.5625  0.    ]
             [ 0.      0.      0.      0.      0.      0.      0.      6.25  ]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time
                step.

        Returns:
            mean (np.ndarray): Mean vector of the predicted state.
            covariance (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            confidence (float, optional): Detection confidence; when set, scales measurement noise by max(1 -
                confidence, 0.05) (NSA-Kalman).

        Returns:
            mean (np.ndarray): Projected mean of the given state estimate.
            covariance (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:  # NSA-Kalman: scale measurement noise by detection confidence (StrongSORT)
            innovation_cov *= max(1.0 - float(confidence), 0.05)

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = np.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = np.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float | None = None
    ):
        """Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.
            confidence (float, optional): Detection confidence; when set, scales measurement noise by max(1 -
                confidence, 0.05) (NSA-Kalman).

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement, confidence=confidence)


def resolve_angle(
    measured_angle: float,
    ref_angle: float,
    vx: float,
    vy: float,
    period: float,
    motion_thresh: float,
) -> float:
    """在角度的周期性模糊候选集合 {measured_angle + k*period, k=...,-1,0,1,...} 中，
    选出与"参考方向"最匹配的等价角度，返回值与 measured_angle 相差 period 的整数倍。

    参考方向的选取：若目标运动速度 sqrt(vx**2+vy**2) 超过 motion_thresh（车辆确实在运动），
    用运动航向 atan2(vy, vx) 作为参考——车辆朝向和运动方向物理上高度相关，比单纯"离预测角度
    最近"更可靠（尤其是丢失多帧后预测角度本身可能已经跑偏的情况）。速度太小（接近静止）时
    航向不可靠，退化为直接用 ref_angle（比如卡尔曼预测角度，或上一帧观测角度）做最近邻选择。

    Args:
        measured_angle (float): 本次观测到的原始角度（弧度），可能存在 k*period 的整数倍歧义。
        ref_angle (float): 静止时退化使用的参考角度（弧度），如预测角度或上一帧角度。
        vx (float): 目标当前 x 方向速度（像素/帧）。
        vy (float): 目标当前 y 方向速度（像素/帧）。
        period (float): 角度的物理周期（弧度），矩形对称场景通常是 pi。
        motion_thresh (float): 判定"是否在运动"的速度阈值（像素/帧），越小越倾向使用航向参考。

    Returns:
        (float): 消歧后的角度（弧度），与 measured_angle 模 period 同余，且离参考方向最近。
    """
    speed = float(np.hypot(vx, vy))
    target = float(np.arctan2(vy, vx)) if speed > motion_thresh else float(ref_angle)
    diff = float(measured_angle) - target
    diff_wrapped = (diff + period / 2.0) % period - period / 2.0
    return target + diff_wrapped


class KalmanFilterXYWHA(KalmanFilterXYWH):
    """在 KalmanFilterXYWH 的 (x, y, w, h) 基础上扩展一个角度维度，用于旋转框(OBB)跟踪。

    状态向量为 10 维：(x, y, w, h, angle, vx, vy, vw, vh, vangle)，观测向量为 5 维：
    (x, y, w, h, angle)。角度不做归一化/裁剪，内部允许无界累积（例如车辆持续转圈时角度可以
    超过 ±360°），只在 update() 接收新观测时，通过 resolve_angle() 把原始角度解到离参考方向
    最近的等价分支，避免角度在周期边界（如 179°→-179°）附近产生错误的"跳变"修正。

    角度维度的不确定度（position/velocity std）使用固定弧度常数，不像 x/y/w/h 那样按框尺寸
    缩放——角度误差和物体大小无关。
    """

    def __init__(
        self,
        angle_period: float = np.pi,
        motion_static_threshold: float = 2.0,
        std_weight_angle_position: float = np.deg2rad(5.0),
        std_weight_angle_velocity: float = np.deg2rad(1.0),
    ):
        """Initialize the 10-dimensional (x, y, w, h, angle, ...) Kalman filter model matrices.

        Args:
            angle_period (float): 角度物理周期（弧度）。矩形180°对称场景用 pi，默认 pi。
            motion_static_threshold (float): 判定"是否在运动"的速度阈值（像素/帧），
                用于 resolve_angle() 里选择用运动航向还是参考角度做角度消歧。
            std_weight_angle_position (float): 角度位置不确定度（弧度），不随框尺寸缩放。
            std_weight_angle_velocity (float): 角速度不确定度（弧度/帧），不随框尺寸缩放。
        """
        super().__init__()  # 先按父类初始化 _std_weight_position / _std_weight_velocity 等标量权重

        ndim, dt = 5, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self.angle_period = angle_period
        self.motion_static_threshold = motion_static_threshold
        self._std_angle_position = std_weight_angle_position
        self._std_angle_velocity = std_weight_angle_velocity

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement (x, y, w, h, angle)."""
        mean_pos = np.asarray(measurement, dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_angle_position,
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_angle_velocity,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step for the 10-dimensional state."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_angle_position,
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_angle_velocity,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """Project the 10-dimensional state distribution to the 5-dimensional measurement space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_angle_position,
        ]
        innovation_cov = np.diag(np.square(std))
        if confidence is not None:
            innovation_cov *= max(1.0 - float(confidence), 0.05)

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorized prediction step for multiple 10-dimensional object states."""
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_angle_position * np.ones_like(mean[:, 2]),
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_angle_velocity * np.ones_like(mean[:, 2]),
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, confidence: float | None = None
    ):
        """Run Kalman filter correction step, first resolving the angle's periodic ambiguity.

        Args:
            mean (np.ndarray): The predicted state's mean vector (10 dimensional).
            covariance (np.ndarray): The state's covariance matrix (10x10 dimensional).
            measurement (np.ndarray): The 5 dimensional measurement vector (x, y, w, h, angle). The angle
                component may be off by an integer multiple of `angle_period` from the "true" branch; it is
                resolved against the track's current heading (from vx, vy) or predicted angle before the
                standard linear-Gaussian update runs.
            confidence (float, optional): Detection confidence; when set, scales measurement noise (NSA-Kalman).

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.
        """
        measurement = np.array(measurement, dtype=float, copy=True)
        predicted_angle = mean[4]
        vx, vy = mean[5], mean[6]
        measurement[4] = resolve_angle(
            measured_angle=measurement[4],
            ref_angle=predicted_angle,
            vx=vx,
            vy=vy,
            period=self.angle_period,
            motion_thresh=self.motion_static_threshold,
        )
        return super().update(mean, covariance, measurement, confidence=confidence)
