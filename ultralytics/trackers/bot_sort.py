# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import parse_bboxes
from .utils.virtual_trajectory import VirtualTrajectoryConfig, build_virtual_trajectory


class ReidRecoveryConfig:
    """Tunable parameters for the ReID-guided third association pass.

    Attributes:
        MIN_GAP_FOR_RECOVERY (int): Minimum consecutive lost frames before a track is even considered
            for this pass. Short gaps are already handled fine by the normal IoU association; this
            pass exists specifically for the long-gap case where straight-line drift breaks IoU.
        ACCEPT_SIM_THRESH (float): Minimum cosine similarity between a Lost track's smoothed appearance
            feature and a candidate detection's feature to accept the pair as a genuine recovery.
            Chosen from empirical measurement on this project's real IDSW cases: same-vehicle
            (pre-loss vs. reappearance) similarity was never below ~0.65-0.70, while different-vehicle
            similarity in the same scene never exceeded ~0.57-0.59 — a threshold of 0.65 sits cleanly
            in that gap.
        MAX_SPEED_PX_PER_FRAME (float): Appearance similarity alone is not sufficient — on a scene with
            many simultaneously Lost tracks, some unrelated vehicle's embedding will occasionally clear
            `ACCEPT_SIM_THRESH` by chance (confirmed in practice: a newly-entering vehicle got matched to
            a Lost track last seen ~1000px away on the opposite side of the frame). This caps how far a
            candidate's last known position may be from the new detection, scaled by how long it's been
            lost (`gap * MAX_SPEED_PX_PER_FRAME`), using a generous real-world vehicle speed bound. This
            project's genuine long-gap recoveries never exceeded ~6.2 px/frame of actual displacement
            (measured from confirmed IDSW cases), so 20 px/frame leaves ample headroom for real
            (including curved-road) motion while still rejecting cross-frame teleportation.
        EDGE_MARGIN_PX (float): A Lost track whose last observation sits within this many pixels of the
            frame boundary is a candidate for the "actually drove off-screen" exclusion below. A plain
            "last position is near the edge" filter was tried and measured unsafe: on this footage 91%
            of ALL long-gap recoveries (legit ones included) have an edge-adjacent last position, because
            the chronic-occlusion ramp corner that motivates this whole recovery pass happens to sit right
            at the frame boundary too — position alone can't tell "drove away for good" from "still on the
            ramp, just undetected". See `EDGE_EXIT_MIN_SPEED` for the signal that actually discriminates.
        MIN_SIZE_IOU (float): Rotation-independent size-consistency gate — compares only (long side, short
            side) of the candidate's last-observed box against the new detection's box (both reduced to
            their own long/short edges first, so a 90-degree labeling ambiguity doesn't matter), ignoring
            position and orientation entirely. Orientation is deliberately excluded: a vehicle recovered
            after a long gap on a curved ramp has genuinely rotated with the road, so requiring angle
            agreement would penalize exactly the curved-road recoveries this pass exists for. Physical
            size doesn't change with heading, so it stays valid there. Measured on this project's real
            "chimera" mismatches (a Lost track's old id wrongly reactivated by a different real vehicle):
            some pairs (similarly-sized sedans) aren't caught by this alone, but clear size mismatches
            (e.g. 0.44 for a confirmed bad case) are — set below the ~0.61-0.95 range measured on confirmed
            genuine recoveries, so it only rejects clearly-mismatched sizes, not ordinary detection noise.
        EDGE_EXIT_MIN_SPEED (float): Only skip an edge-adjacent Lost track as a recovery candidate if its
            Kalman velocity *at the moment it was lost* also had an outward component (toward the nearest
            boundary) exceeding this speed (px/frame) — i.e. it was actually heading off-screen, not just
            entering, idling, or drifting parallel to the edge. Measured on this footage: restricting to
            direction-aware exclusion still catches 10/12 of the confirmed edge-zombie IDSW cases while
            only affecting 77% of all-recoveries instead of 91%, and the ones it does affect mostly have a
            visibly ballistic outward trajectory (multi-px/frame straight toward the boundary), not the
            near-zero/parallel drift of a track idling near an entry point.
    """

    MIN_GAP_FOR_RECOVERY = 30
    ACCEPT_SIM_THRESH = 0.65
    MAX_SPEED_PX_PER_FRAME = 20.0
    EDGE_MARGIN_PX = 100.0
    EDGE_EXIT_MIN_SPEED = 0.5
    MIN_SIZE_IOU = 0.5


class BOTrack(STrack):
    """An extended version of the STrack class for YOLO, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None):
        """Initialize a BOTrack object with feature-smoothing state and a Kalman filter.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

        # Snapshot of (mean, covariance, frame_id) taken right after the most recent *real*
        # correction (activate/update/re_activate), used as the t1 anchor for virtual trajectory
        # replay when this track goes Lost and is later re-matched. `predict()`-only steps never
        # touch these, so they always point at the last genuinely observed state.
        self.last_observed_mean: np.ndarray | None = None
        self.last_observed_covariance: np.ndarray | None = None
        self.last_observed_frame: int | None = None

    def _cache_last_observed(self) -> None:
        """Snapshot the current (post-correction) mean/covariance/frame_id as the new t1 anchor."""
        self.last_observed_mean = self.mean.copy()
        self.last_observed_covariance = self.covariance.copy()
        self.last_observed_frame = self.frame_id

    def activate(self, kalman_filter, frame_id: int) -> None:
        """Activate a new tracklet and cache its state as the initial last-observed anchor."""
        super().activate(kalman_filter, frame_id)
        self._cache_last_observed()

    def update_features(self, feat: np.ndarray) -> None:
        """Update the current feature and its exponential-moving-average smoothed feature."""
        curr, smooth = smooth_feature(feat, self.smooth_feat, self.alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def predict(self) -> None:
        """Predict the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0  # vw
            mean_state[7] = 0  # vh
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a track with updated features and optionally assign a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)
        self._cache_last_observed()

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        """Update the track with new detection information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)
        self._cache_last_observed()

    @property
    def tlwh(self) -> np.ndarray:
        """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0  # vw
                multi_mean[i][7] = 0  # vh
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[BOTrack], H: np.ndarray) -> None:
        """Apply global motion compensation correctly for the 8-dimensional XYWH Kalman state.

        The generic `multi_gmc` helper in `utils.stracks` assumes a state made of four consecutive
        2-d pairs (e.g. `(x,y),(w,h),(vx,vy),(vw,vh)`) and would rotate `(w, h)` along with position,
        which is wrong for a width/height pair. This variant only rotates the position `(x, y)` and
        velocity `(vx, vy)` blocks; `(w, h)` and `(vw, vh)` are left untouched.

        Args:
            stracks (list[BOTrack]): Tracks to warp in place.
            H (np.ndarray): 2x3 affine homography mapping the previous frame to the current one.
        """
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        t = H[:2, 2]

        state_dim = multi_mean.shape[1]
        R_full = np.eye(state_dim, dtype=np.float32)
        R_full[:2, :2] = R  # rotate position (x, y)
        vel_offset = state_dim // 2
        R_full[vel_offset : vel_offset + 2, vel_offset : vel_offset + 2] = R  # rotate velocity (vx, vy)

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R_full.dot(mean)
            mean[:2] += t
            cov = R_full.dot(cov).dot(R_full.transpose())
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detection results and optional image for ReID.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict the mean and covariance of multiple object tracks using a shared Kalman filter.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args)
        >>> bot_sort.init_track(results, img)
        >>> bot_sort.multi_predict(tracks)

    Notes:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
        """
        super().__init__(args)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = build_encoder(args.with_reid, args.model, getattr(args, "device", None))
        self.frame_shape: tuple[int, int] | None = None  # (h, w) of the most recent frame, for edge-exit checks

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states."""
        return KalmanFilterXYWH()

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        """Apply GMC using BOTrack's dimension-aware multi_gmc (not the generic width/height-corrupting one)."""
        if img is not None:
            self.frame_shape = img.shape[:2]
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results_high.xyxy)
            except Exception as e:
                LOGGER.warning(f"GMC failed, falling back to identity: {e}")
                warp = np.eye(2, 3)
            BOTrack.multi_gmc(strack_pool, warp)
            BOTrack.multi_gmc(unconfirmed, warp)

    def _apply_match(self, track: BOTrack, det: BOTrack, activated: list[BOTrack], refind: list[BOTrack]) -> None:
        """Update or re-activate a matched track, bridging Lost gaps with a virtual trajectory first.

        For a re-matched Lost track whose gap is long enough (`VirtualTrajectoryConfig.
        MIN_LOST_FRAMES_FOR_VIRTUAL`), this replays predict()+update() through a sequence of smooth
        LI+GPR pseudo-observations for the lost frames before the real re_activate() runs, instead of
        letting the track coast on pure Kalman prediction across the whole gap.
        """
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated.append(track)
            return
        self._maybe_inject_virtual_trajectory(track, det)
        track.re_activate(det, self.frame_id, new_id=False)
        refind.append(track)

    def _maybe_inject_virtual_trajectory(self, track: BOTrack, det: BOTrack) -> None:
        """Rewind `track` to its last real observation and replay virtual predict+update through the gap.

        The Kalman state here is plain `(x,y,w,h,vx,vy,vw,vh)` (no angle dimension), so only the
        4-dim xywh position/size is bridged with LI+GPR; angle is not part of the KF state in this
        configuration and is simply carried over directly from the real detection at re_activate().
        """
        cfg = VirtualTrajectoryConfig
        t1 = track.last_observed_frame
        t2 = self.frame_id
        if t1 is None or track.last_observed_mean is None:
            return
        lost_frame_indices = list(range(t1 + 1, t2))
        if len(lost_frame_indices) < cfg.MIN_LOST_FRAMES_FOR_VIRTUAL:
            return

        z_t1 = track.last_observed_mean[:4]
        z_t2 = track.tlwh_to_xywh(det.tlwh)
        vx, vy = track.last_observed_mean[4], track.last_observed_mean[5]

        virtual_obs = build_virtual_trajectory(
            z_t1=z_t1,
            z_t2=z_t2,
            t1=t1,
            t2=t2,
            lost_frame_indices=lost_frame_indices,
            angle_dim_index=None,
            gpr_length_scale=cfg.GPR_LENGTH_SCALE,
            gpr_noise_level=cfg.GPR_NOISE_LEVEL,
            vx=vx,
            vy=vy,
            motion_static_threshold=cfg.MOTION_STATIC_THRESHOLD,
        )

        track.mean = track.last_observed_mean.copy()
        track.covariance = track.last_observed_covariance.copy()
        for obs in virtual_obs:
            track.predict()  # frozen vw/vh, same as any other Lost-state predict step
            track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, obs)
        track.predict()  # advance to t2; re_activate()'s update() assumes an already-predicted mean

    def _is_exiting_frame(self, track: BOTrack) -> bool:
        """Check whether `track` was last seen heading off-screen, not just idling/entering near an edge.

        A Lost track near the frame boundary is ambiguous by position alone (see `ReidRecoveryConfig.
        EDGE_MARGIN_PX`). What actually distinguishes "drove away for good" is the direction of travel at
        the moment it was lost: a track whose Kalman velocity pointed outward, toward the nearest
        boundary, faster than `EDGE_EXIT_MIN_SPEED`, was genuinely leaving the scene. A track that was
        entering, idling, or moving roughly parallel to the boundary is not.
        """
        if track.last_observed_mean is None or self.frame_shape is None:
            return False
        cfg = ReidRecoveryConfig
        x, y = track.last_observed_mean[0], track.last_observed_mean[1]
        h, w = self.frame_shape
        d_left, d_right, d_top, d_bottom = x, w - x, y, h - y
        d_min = min(d_left, d_right, d_top, d_bottom)
        if d_min >= cfg.EDGE_MARGIN_PX:
            return False
        if d_min == d_left:
            outward = (-1.0, 0.0)
        elif d_min == d_right:
            outward = (1.0, 0.0)
        elif d_min == d_top:
            outward = (0.0, -1.0)
        else:
            outward = (0.0, 1.0)
        vx, vy = track.last_observed_mean[4], track.last_observed_mean[5]
        return (vx * outward[0] + vy * outward[1]) > cfg.EDGE_EXIT_MIN_SPEED

    @staticmethod
    def _size_iou(w1: float, h1: float, w2: float, h2: float) -> float:
        """Rotation-independent size-consistency score: IoU of the two boxes' (long, short) edges only.

        Reducing each box to (max(w,h), min(w,h)) before comparing sidesteps the label ambiguity between
        w/h and a +/-90 degree angle, and deliberately discards orientation entirely — see
        `ReidRecoveryConfig.MIN_SIZE_IOU` for why angle isn't safe to use here.
        """
        long1, short1 = max(w1, h1), min(w1, h1)
        long2, short2 = max(w2, h2), min(w2, h2)
        inter = min(long1, long2) * min(short1, short2)
        area1, area2 = long1 * short1, long2 * short2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _post_first_association(
        self,
        strack_pool: list[BOTrack],
        detections: list[BOTrack],
        u_track: list[int],
        u_detection: list[int],
        activated: list[BOTrack],
        refind: list[BOTrack],
    ) -> tuple[list[int], list[int]]:
        """Third association pass: recover long-lost tracks via appearance (ReID), not geometry.

        A Lost track can fail the first IoU pass simply because its constant-velocity-predicted
        position has drifted too far over a long gap (e.g. a curved road bends away from a
        straight-line extrapolation). Trying to fix this with a better geometric guess (a road-shape
        prior) turned out to be unsafe on a busy multi-lane road: there is almost always *some*
        recently-lost track sitting near any given position, so a purely geometric recovery pass ends
        up gluing unrelated vehicles together under an old id instead of the intended fix.

        Appearance doesn't have that problem — two different nearby vehicles usually still look
        different. So this pass only fires for tracks lost long enough that IoU-based recovery is
        already known to fail (`MIN_GAP_FOR_RECOVERY`), and only accepts a candidate whose ReID
        cosine similarity to the track's EMA-smoothed appearance feature clears `ACCEPT_SIM_THRESH`,
        subject to a physical-distance sanity bound (`MAX_SPEED_PX_PER_FRAME`), an edge-exit check
        (`_is_exiting_frame`) that drops candidates last seen heading off-screen, and a rotation-independent
        size-consistency check (`_size_iou` vs. `MIN_SIZE_IOU`) that drops candidates whose physical
        dimensions don't match the new detection. If no candidate clears the bar for a given detection, it
        is left untouched and falls through to the normal downstream stages (unconfirmed association, then
        new-track creation) exactly as today.
        """
        if not self.args.with_reid or self.encoder is None:
            return u_track, u_detection

        cfg = ReidRecoveryConfig
        lost_idx = [
            i for i in u_track
            if strack_pool[i].state == TrackState.Lost
            and strack_pool[i].last_observed_frame is not None
            and (self.frame_id - strack_pool[i].last_observed_frame) >= cfg.MIN_GAP_FOR_RECOVERY
            and strack_pool[i].smooth_feat is not None
            and not self._is_exiting_frame(strack_pool[i])
        ]
        det_idx = [j for j in u_detection if detections[j].curr_feat is not None]
        if len(lost_idx) == 0 or len(det_idx) == 0:
            return u_track, u_detection

        lost_tracks = [strack_pool[i] for i in lost_idx]
        cand_dets = [detections[j] for j in det_idx]
        dist = matching.embedding_distance(lost_tracks, cand_dets)  # cosine distance, (len(lost_idx), len(det_idx))

        candidates = []  # (neg_similarity, track_pool_idx, detection_idx) — ascending sort => best match first
        for a, ti in enumerate(lost_idx):
            track = strack_pool[ti]
            gap = self.frame_id - track.last_observed_frame
            max_dist = gap * cfg.MAX_SPEED_PX_PER_FRAME
            lx, ly = track.last_observed_mean[0], track.last_observed_mean[1]
            for b, dj in enumerate(det_idx):
                det = detections[dj]
                if det.cls != track.cls:
                    continue
                sim = 1.0 - float(dist[a, b])
                if sim < cfg.ACCEPT_SIM_THRESH:
                    continue
                dx, dy = det.tlwh[0] + det.tlwh[2] / 2, det.tlwh[1] + det.tlwh[3] / 2
                if float(np.hypot(dx - lx, dy - ly)) > max_dist:
                    continue  # appearance matched, but physically implausible given how long it's been lost
                size_iou = self._size_iou(
                    track.last_observed_mean[2], track.last_observed_mean[3],
                    det.tlwh[2], det.tlwh[3],
                )
                if size_iou < cfg.MIN_SIZE_IOU:
                    continue  # appearance and distance matched, but this is a differently-sized vehicle
                candidates.append((-sim, ti, dj))

        candidates.sort(key=lambda c: c[0])

        import os as _os
        if _os.environ.get("REID_DEBUG_LOG"):
            self._debug_log_candidates(candidates, strack_pool, detections)

        used_tracks, used_dets = set(), set()
        for _, ti, dj in candidates:
            if ti in used_tracks or dj in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(dj)
            track, det = strack_pool[ti], detections[dj]
            self._maybe_inject_virtual_trajectory(track, det)
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)

        if not used_tracks:
            return u_track, u_detection
        return (
            [i for i in u_track if i not in used_tracks],
            [j for j in u_detection if j not in used_dets],
        )

    def _debug_log_candidates(
        self, candidates: list[tuple[float, int, int]], strack_pool: list[BOTrack], detections: list[BOTrack]
    ) -> None:
        """Temporary debug hook: dump every (lost_track, candidate_det) pair considered this frame to a log file.

        Gated behind the ``REID_DEBUG_LOG`` env var so it has zero cost/effect when unset. Used to
        offline-compare the actual greedy pick against a global-optimal (Hungarian) assignment over
        the same candidate set, to check whether greedy sequential acceptance is itself a source of
        wrong chimera matches (as opposed to the underlying signal being ambiguous either way).
        """
        import os as _os

        path = _os.environ["REID_DEBUG_LOG"]
        with open(path, "a", encoding="utf-8") as f:
            for neg_sim, ti, dj in candidates:
                track, det = strack_pool[ti], detections[dj]
                dx, dy = det.tlwh[0] + det.tlwh[2] / 2, det.tlwh[1] + det.tlwh[3] / 2
                f.write(
                    f"{self.frame_id}\t{track.track_id}\t{dj}\t{-neg_sim:.4f}\t{dx:.1f}\t{dy:.1f}\n"
                )

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None and img is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Calculate distances between tracks and detections using IoU only.

        This project deliberately does not blend ReID embedding distance into the normal short-range
        association here, even though `args.with_reid` is on: on this busy multi-lane highway many
        visually-similar vehicles (same color/model) sit close together, and empirically enabling the
        embedding blend on every frame caused rapid, spurious id swaps between adjacent tracks (IDSW
        went from 16 to 76) — far worse than the drift problem it was meant to help with. ReID is still
        useful here, just not for this fine-grained frame-to-frame decision: `with_reid=True` keeps
        `curr_feat`/`smooth_feat` populated so the dedicated long-gap recovery pass in
        `_post_first_association` (which only fires after MIN_GAP_FOR_RECOVERY frames lost, where
        appearance is the more reliable signal) can use them.
        """
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()
