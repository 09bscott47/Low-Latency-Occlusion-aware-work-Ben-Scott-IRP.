# # Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# class DeepOCSort(BaseTracker):
#     def __init__(
#         self,
#         model_weights=None,
#         device='cuda:0',
#         fp16=False,
#         per_class=False,
#         det_thresh=0.3,
#         max_age=30,
#         min_hits=3,
#         iou_threshold=0.3,
#         delta_t=3,
#         asso_func="iou",
#         inertia=0.2,
#         w_association_emb=0.5,
#         alpha_fixed_emb=0.95,
#         aw_param=0.5,
#         embedding_off=False,
#         cmc_off=True,
#         aw_off=False,
#         new_kf_off=False,
#         custom_features=False,
#         **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#         # (M detections X N tracks, final score)
#
#         if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
#             stage1_emb_cost = None
#         else:
#             stage1_emb_cost = dets_embs @ trk_embs.T
#
#         matched, unmatched_dets, unmatched_trks = associate(
#             dets[:, 0:5],
#             trks,
#             self.asso_func,
#             self.iou_threshold,
#             velocities,
#             k_observations,
#             self.inertia,
#             img.shape[1],  # w
#             img.shape[0],  # h
#             stage1_emb_cost,
#             self.w_association_emb,
#             self.aw_off,
#             self.aw_param,
#         )
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(
#                 dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                                       ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                            trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])
##############################################################################
#Above is normal, below is rubbish depth cascade association
##############################################################################
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
#
# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.depth = self.compute_depth(self.get_state()[0])  #######################################
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def compute_depth(self, bbox): ###############################################################
#         return bbox[3]  # bottom of bbox
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# class DeepOCSort(BaseTracker):
#     def __init__(
#             self,
#             model_weights=None,
#             device='cuda:0',
#             fp16=False,
#             per_class=False,
#             det_thresh=0.3,
#             max_age=30,
#             min_hits=3,
#             iou_threshold=0.3,
#             delta_t=3,
#             asso_func="iou",
#             inertia=0.2,
#             w_association_emb=0.5,
#             alpha_fixed_emb=0.95,
#             aw_param=0.5,
#             embedding_off=False,
#             cmc_off=True,
#             aw_off=False,
#             new_kf_off=False,
#             custom_features=False,
#             **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#         # (M detections X N tracks, final score)
#
#         if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
#             stage1_emb_cost = None
#         else:
#             stage1_emb_cost = dets_embs @ trk_embs.T
#
#         # Detections: [x1, y1, x2, y2, score, cls, det_ind]
#         # Track predictions: [x1, y1, x2, y2, 0]
#
# #####################################################
#         def compute_depth(bbox):
#             # Pseudo-depth from bottom of box
#             return bbox[1] + bbox[3]  # y1 + h ≈ y2
#
#         bin_width = 100  # adjust based on dataset scale (pixels)
#
#         # Group detections and tracks into bins
#         depth_bins = {}
#
#         for d_idx, det in enumerate(dets):
#             y1, h = det[1], det[3]
#             depth = compute_depth([0, y1, 0, h])  # use y + h
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {"dets": [], "dets_idx": []})
#             depth_bins[bin_id]["dets"].append(det)
#             depth_bins[bin_id]["dets_idx"].append(d_idx)
#
#         for t_idx, trk in enumerate(trks):
#             y1, h = trk[1], trk[3]
#             depth = compute_depth([0, y1, 0, h])
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks", []).append(trk)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks_idx", []).append(t_idx)
#
#         matched = []
#         unmatched_dets = set(range(len(dets)))
#         unmatched_trks = set(range(len(trks)))
#
#         for bin_data in depth_bins.values():
#             if "dets" not in bin_data or "trks" not in bin_data:
#                 continue  # skip if only dets or only tracks
#
#             dets_bin = np.array(bin_data["dets"])
#             trks_bin = np.array(bin_data["trks"])
#             dets_idx_bin = bin_data["dets_idx"]
#             trks_idx_bin = bin_data["trks_idx"]
#
#             # Filter embeddings for bin
#             dets_embs_bin = dets_embs[dets_idx_bin]
#             trks_embs_bin = trk_embs[trks_idx_bin]
#             k_obs_bin = k_observations[trks_idx_bin]
#             vel_bin = velocities[trks_idx_bin]
#
#             if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#                 emb_cost = None
#             else:
#                 emb_cost = dets_embs_bin @ trks_embs_bin.T
#
#             # Run association in the bin
#             matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#                 dets_bin[:, 0:5],
#                 trks_bin,
#                 self.asso_func,
#                 self.iou_threshold,
#                 vel_bin,
#                 k_obs_bin,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#
#             # Adjust matched indices back to global indices
#             for d_local, t_local in matched_bin:
#                 matched.append((dets_idx_bin[d_local], trks_idx_bin[t_local]))
#                 unmatched_dets.discard(dets_idx_bin[d_local])
#                 unmatched_trks.discard(trks_idx_bin[t_local])
#
#         unmatched_dets = np.array(list(unmatched_dets))
#         unmatched_trks = np.array(list(unmatched_trks))
#
#         ##################################################### removed the below and replaced with the above.
#         # matched, unmatched_dets, unmatched_trks = associate(
#         #     dets[:, 0:5],
#         #     trks,
#         #     self.asso_func,
#         #     self.iou_threshold,
#         #     velocities,
#         #     k_observations,
#         #     self.inertia,
#         #     img.shape[1],  # w
#         #     img.shape[0],  # h
#         #     stage1_emb_cost,
#         #     self.w_association_emb,
#         #     self.aw_off,
#         #     self.aw_param,
#         # )
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(
#                 dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                        ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                     trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])
##############################################################################
#Above is rubbish depth cascade association, below is depth cascade association along with Temporal depth consistency checks and Re-association for depth violations
##############################################################################
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#         # Inside KalmanBoxTracker.__init__
#         self.prev_depth = None ##################################
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.depth = self.compute_depth(self.get_state()[0])  #######################################
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def compute_depth(self, bbox): ###############################################################
#         return bbox[3]  # bottom of bbox
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#         self.prev_depth = self.depth  # store previous depth
#         self.depth = self.compute_depth(self.get_state()[0])  # update current
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# class DeepOCSort(BaseTracker):
#     def __init__(
#             self,
#             model_weights=None,
#             device='cuda:0',
#             fp16=False,
#             per_class=False,
#             det_thresh=0.3,
#             max_age=30,
#             min_hits=3,
#             iou_threshold=0.3,
#             delta_t=3,
#             asso_func="iou",
#             inertia=0.2,
#             w_association_emb=0.5,#0.5
#             alpha_fixed_emb=0.95,
#             aw_param=0.5,
#             embedding_off=False,
#             cmc_off=True,
#             aw_off=False,
#             new_kf_off=False,
#             custom_features=False,
#             **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#         # (M detections X N tracks, final score)
#
#         if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
#             stage1_emb_cost = None
#         else:
#             stage1_emb_cost = dets_embs @ trk_embs.T
#
#         # Detections: [x1, y1, x2, y2, score, cls, det_ind]
#         # Track predictions: [x1, y1, x2, y2, 0]
#
# #####################################################
#         def compute_depth(bbox):
#             # Pseudo-depth from bottom of box
#             return bbox[1] + bbox[3]  # y1 + h ≈ y2
#
#         bin_width = 100  # adjust based on dataset scale (pixels)
#
#         # Group detections and tracks into bins
#         depth_bins = {}
#
#         for d_idx, det in enumerate(dets):
#             y1, h = det[1], det[3]
#             depth = compute_depth([0, y1, 0, h])  # use y + h
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {"dets": [], "dets_idx": []})
#             depth_bins[bin_id]["dets"].append(det)
#             depth_bins[bin_id]["dets_idx"].append(d_idx)
#
#         for t_idx, trk in enumerate(trks):
#             y1, h = trk[1], trk[3]
#             depth = compute_depth([0, y1, 0, h])
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks", []).append(trk)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks_idx", []).append(t_idx)
#
#         matched = []
#         unmatched_dets = set(range(len(dets)))
#         unmatched_trks = set(range(len(trks)))
#
#         for bin_data in depth_bins.values():
#             if "dets" not in bin_data or "trks" not in bin_data:
#                 continue  # skip if only dets or only tracks
#
#             dets_bin = np.array(bin_data["dets"])
#             trks_bin = np.array(bin_data["trks"])
#             dets_idx_bin = bin_data["dets_idx"]
#             trks_idx_bin = bin_data["trks_idx"]
#
#             # Filter embeddings for bin
#             dets_embs_bin = dets_embs[dets_idx_bin]
#             trks_embs_bin = trk_embs[trks_idx_bin]
#             k_obs_bin = k_observations[trks_idx_bin]
#             vel_bin = velocities[trks_idx_bin]
#
#             if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#                 emb_cost = None
#             else:
#                 emb_cost = dets_embs_bin @ trks_embs_bin.T
#
#             # Run association in the bin
#             matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#                 dets_bin[:, 0:5],
#                 trks_bin,
#                 self.asso_func,
#                 self.iou_threshold,
#                 vel_bin,
#                 k_obs_bin,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#
#             # Adjust matched indices back to global indices
#             for d_local, t_local in matched_bin:
#                 matched.append((dets_idx_bin[d_local], trks_idx_bin[t_local]))
#                 unmatched_dets.discard(dets_idx_bin[d_local])
#                 unmatched_trks.discard(trks_idx_bin[t_local])
#
#         unmatched_dets = np.array(list(unmatched_dets), dtype=int)
#         # unmatched_trks = np.array(list(unmatched_trks))
#         unmatched_trks = np.array(list(unmatched_trks), dtype=int)
#
#         ##################################################### removed the below and replaced with the above.
#         # matched, unmatched_dets, unmatched_trks = associate(
#         #     dets[:, 0:5],
#         #     trks,
#         #     self.asso_func,
#         #     self.iou_threshold,
#         #     velocities,
#         #     k_observations,
#         #     self.inertia,
#         #     img.shape[1],  # w
#         #     img.shape[0],  # h
#         #     stage1_emb_cost,
#         #     self.w_association_emb,
#         #     self.aw_off,
#         #     self.aw_param,
#         # )
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(
#                 dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         #############################################################################
#         # Below is looking at which violates rules
#         #############################################################################
#         depth_violations = []
#         trk_recheck = []
#         det_recheck = []
#
#         for d_idx, t_idx in matched:
#             track = self.active_tracks[t_idx]
#             det = dets[d_idx]
#
#             # Current pseudo-depth (e.g., y + h)
#             z_depth = det[1] + det[3]
#             t_depth = track.depth
#             prev_depth = track.prev_depth
#             track.prev_depth = track.depth###############################
#             # Store current depth
#             track.depth = z_depth
#
#             # Simple violation condition: large depth swap
#             if prev_depth is not None:
#                 depth_diff = abs(z_depth - prev_depth)
#                 if depth_diff > 80:  # This is a tunable threshold in pixels
#                     # Optionally add a smarter verification rule here
#                     trk_recheck.append(t_idx)
#                     det_recheck.append(d_idx)
#                     print("Depth violation triggered!")
#
#         # Remove invalid matches
#         matched = [pair for pair in matched if not (pair[0] in det_recheck or pair[1] in trk_recheck)]
#
#         # Store violating pairs CAN GET RID OF - BEN
#         violating_pairs = list(zip(det_recheck, trk_recheck))
#         print(f"[CHECK] Violating pairs before recheck: {violating_pairs}")
#
#         # Mark recheck sets as unmatched
#         unmatched_dets = np.concatenate([unmatched_dets, np.array(det_recheck, dtype=int)], axis=0)
#         unmatched_trks = np.concatenate([unmatched_trks, np.array(trk_recheck, dtype=int)], axis=0)
#
#         ############################################################################
#         # Here is Reassociation for Depth-Violating Tracks (Recheck)
#         ############################################################################
#         if len(det_recheck) > 0 and len(trk_recheck) > 0:
#             print(f"[RECHECK] Depth violations detected — {len(det_recheck)} dets, {len(trk_recheck)} tracks") #############################
#             # Get subset
#             recheck_dets = dets[det_recheck]
#             recheck_dets_embs = dets_embs[det_recheck]
#             recheck_trks = trks[trk_recheck]
#             recheck_trks_embs = trk_embs[trk_recheck]
#             recheck_kobs = k_observations[trk_recheck]
#             recheck_vels = velocities[trk_recheck]
#
#             # if self.embedding_off or recheck_dets_embs.shape[0] == 0 or recheck_trks_embs.shape[0] == 0:
#             #     emb_cost_recheck = None
#             # else:
#             #     emb_cost_recheck = recheck_dets_embs @ recheck_trks_embs.T
#
#                 #####################################################################
#             # for d_local, t_local in zip(range(len(det_recheck)), range(len(trk_recheck))):
#             #     if det_recheck[d_local] == trk_recheck[t_local]:  # exact same pair
#             #         if emb_cost_recheck is not None:
#             #             emb_cost_recheck[d_local, t_local] = np.inf
#                 #####################################################################
#             ######################################
#             # Build emb_cost_recheck and mask invalid pairs
#             if self.embedding_off or recheck_dets_embs.shape[0] == 0 or recheck_trks_embs.shape[0] == 0:
#                 emb_cost_recheck = None
#             else:
#                 emb_cost_recheck = recheck_dets_embs @ recheck_trks_embs.T
#
#                 # Explicitly mask the violating pairs so they can't be reselected
#                 for d_local, d_idx in enumerate(det_recheck):
#                     for t_local, t_idx in enumerate(trk_recheck):
#                         if (d_idx, t_idx) in violating_pairs:
#                             emb_cost_recheck[d_local, t_local] = +100000000  # or -1e9 if associate can't handle -inf
#                             print("1e-9")
#             ###########################
#             re_matched, rem_unmatched_dets, rem_unmatched_trks = associate(
#                 recheck_dets[:, 0:5],
#                 recheck_trks,
#                 self.asso_func,
#                 self.iou_threshold,
#                 recheck_vels,
#                 recheck_kobs,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost_recheck,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#             print(f"[RECHECK] Re-associated {len(re_matched)} pairs out of {len(det_recheck)}")
# #######################################
#             valid_re_matched = []
#             for d_local, t_local in re_matched:
#                 global_d_idx = det_recheck[d_local]
#                 global_t_idx = trk_recheck[t_local]
#
#                 print(f"[CHECK] Reassociated pair: ({global_d_idx}, {global_t_idx})")
#                 if (global_d_idx, global_t_idx) in violating_pairs:
#                     print("[BLOCKED] SAME MATCH RE-ASSOCIATED! 🚨 Rejected.")
#                     continue
#
#                 matched.append((global_d_idx, global_t_idx))
#                 unmatched_dets = unmatched_dets[unmatched_dets != global_d_idx]
#                 unmatched_trks = unmatched_trks[unmatched_trks != global_t_idx]
#             # print(f"[SUMMARY] {len(valid_re_matched)} re-matches accepted (non-violating)")
#             # print(f"[SUMMARY] {len(re_matched) - len(valid_re_matched)} re-matches blocked due to depth")
#             for pair in violating_pairs:
#                 if pair in matched:
#                     print("fucks sake")
#                 assert pair not in matched, f"[FAIL] Violating pair {pair} was accepted again!"
#
#             ##############################
#             # # Map local to global indices
#             # for d_local, t_local in re_matched:
#             #     global_d_idx = det_recheck[d_local]
#             #     global_t_idx = trk_recheck[t_local]
#             #     matched.append((global_d_idx, global_t_idx))
#             #
#             #     print(f"[CHECK] Reassociated pair: ({global_d_idx}, {global_t_idx})")
#             #
#             #     # 🚨 Check if it matches the original violating pair
#             #     if (global_d_idx, global_t_idx) in violating_pairs:
#             #         print("[WARNING] SAME MATCH RE-ASSOCIATED! 🚨 This should NOT happen.")
# ################################################
#             # Map local to global indices
#             # for d_local, t_local in re_matched:
#             #     global_d_idx = det_recheck[d_local]
#             #     global_t_idx = trk_recheck[t_local]
#             #     matched.append((global_d_idx, global_t_idx))
#             #     unmatched_dets = unmatched_dets[unmatched_dets != global_d_idx]
#             #     unmatched_trks = unmatched_trks[unmatched_trks != global_t_idx]
#
#         ###############################################################################
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                        ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                     trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])

#######################################################################
#######################################################################
##### Below is with proper rules
# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#         # Inside KalmanBoxTracker.__init__
#         self.prev_depth = None ##################################
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.depth = self.compute_depth(self.get_state()[0])  #######################################
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def compute_depth(self, bbox): ###############################################################
#         return bbox[3]  # bottom of bbox
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#         self.prev_depth = self.depth  # store previous depth
#         self.depth = self.compute_depth(self.get_state()[0])  # update current
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# class DeepOCSort(BaseTracker):
#     def __init__(
#             self,
#             model_weights=None,
#             device='cuda:0',
#             fp16=False,
#             per_class=False,
#             det_thresh=0.3,
#             max_age=30,
#             min_hits=3,
#             iou_threshold=0.3,
#             delta_t=3,
#             asso_func="iou",
#             inertia=0.2,
#             w_association_emb=0.5,#0.5
#             alpha_fixed_emb=0.95,
#             aw_param=0.5,
#             embedding_off=False,
#             cmc_off=True,
#             aw_off=False,
#             new_kf_off=False,
#             custom_features=False,
#             **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#         # (M detections X N tracks, final score)
#
#         if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
#             stage1_emb_cost = None
#         else:
#             stage1_emb_cost = dets_embs @ trk_embs.T
#
#         # Detections: [x1, y1, x2, y2, score, cls, det_ind]
#         # Track predictions: [x1, y1, x2, y2, 0]
#
# #####################################################
#         def compute_depth(bbox):
#             # Pseudo-depth from bottom of box
#             return bbox[1] + bbox[3]  # y1 + h ≈ y2
#
#         bin_width = 100  # adjust based on dataset scale (pixels)
#
#         # Group detections and tracks into bins
#         depth_bins = {}
#
#         for d_idx, det in enumerate(dets):
#             y1, h = det[1], det[3]
#             depth = compute_depth([0, y1, 0, h])  # use y + h
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {"dets": [], "dets_idx": []})
#             depth_bins[bin_id]["dets"].append(det)
#             depth_bins[bin_id]["dets_idx"].append(d_idx)
#
#         for t_idx, trk in enumerate(trks):
#             y1, h = trk[1], trk[3]
#             depth = compute_depth([0, y1, 0, h])
#             bin_id = int(depth // bin_width)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks", []).append(trk)
#             depth_bins.setdefault(bin_id, {}).setdefault("trks_idx", []).append(t_idx)
#
#         matched = []
#         unmatched_dets = set(range(len(dets)))
#         unmatched_trks = set(range(len(trks)))
#
#         for bin_data in depth_bins.values():
#             if "dets" not in bin_data or "trks" not in bin_data:
#                 continue  # skip if only dets or only tracks
#
#             dets_bin = np.array(bin_data["dets"])
#             trks_bin = np.array(bin_data["trks"])
#             dets_idx_bin = bin_data["dets_idx"]
#             trks_idx_bin = bin_data["trks_idx"]
#
#             # Filter embeddings for bin
#             dets_embs_bin = dets_embs[dets_idx_bin]
#             trks_embs_bin = trk_embs[trks_idx_bin]
#             k_obs_bin = k_observations[trks_idx_bin]
#             vel_bin = velocities[trks_idx_bin]
#
#             if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#                 emb_cost = None
#             else:
#                 emb_cost = dets_embs_bin @ trks_embs_bin.T
#
#             # Run association in the bin
#             matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#                 dets_bin[:, 0:5],
#                 trks_bin,
#                 self.asso_func,
#                 self.iou_threshold,
#                 vel_bin,
#                 k_obs_bin,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#
#             # Adjust matched indices back to global indices
#             for d_local, t_local in matched_bin:
#                 matched.append((dets_idx_bin[d_local], trks_idx_bin[t_local]))
#                 unmatched_dets.discard(dets_idx_bin[d_local])
#                 unmatched_trks.discard(trks_idx_bin[t_local])
#
#         unmatched_dets = np.array(list(unmatched_dets), dtype=int)
#         # unmatched_trks = np.array(list(unmatched_trks))
#         unmatched_trks = np.array(list(unmatched_trks), dtype=int)
#
#         ##################################################### removed the below and replaced with the above.
#         # matched, unmatched_dets, unmatched_trks = associate(
#         #     dets[:, 0:5],
#         #     trks,
#         #     self.asso_func,
#         #     self.iou_threshold,
#         #     velocities,
#         #     k_observations,
#         #     self.inertia,
#         #     img.shape[1],  # w
#         #     img.shape[0],  # h
#         #     stage1_emb_cost,
#         #     self.w_association_emb,
#         #     self.aw_off,
#         #     self.aw_param,
#         # )
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(
#                 dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         #############################################################################
#         # Below is looking at which violates rules
#         #############################################################################
#         import torch
#         from torchvision.ops import box_iou
#         if len(self.active_tracks) >= 2:
#             last_obs_boxes = np.array([trk.last_observation[:4] for trk in self.active_tracks], ndmin=2)
#
#             # Convert to [x1, y1, x2, y2]
#             last_obs_boxes_xyxy = last_obs_boxes.copy()
#             last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]
#             last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]
#
#             last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
#             iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#
#             overlap_thresh = 0.5
#             occluding_pairs = [
#                 (i, j) for i in range(len(self.active_tracks)) for j in range(i + 1, len(self.active_tracks))
#                 if iou_matrix[i, j] > overlap_thresh
#             ]
#         else:
#             occluding_pairs = []
#
#         # Get raw last observations
#         #last_obs_boxes = np.array([trk.last_observation[:4] for trk in self.active_tracks])  # ensure shape [N, 4]
#         # last_obs_boxes = np.array([trk.last_observation[:4] for trk in self.active_tracks], ndmin=2)
#
#         # # Convert [x, y, w, h] -> [x1, y1, x2, y2]
#         # last_obs_boxes_xyxy = last_obs_boxes.copy()
#         # last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]  # x2 = x + w
#         # last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]  # y2 = y + h
#         #
#         # last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
#
#         # Only compute IoU if there are at least two boxes
#         # if last_obs_boxes_tensor.shape[0] >= 2:
#         #     iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#         # else:
#         #     iou_matrix = torch.zeros((0, 0))
#
#         # from yolox.tracker.matching import iou_distance
#         # from ultralytics.utils.metrics import bbox_iou
#         # from torchvision.ops import box_iou
#         # import torch
#         # # Example: compute pairwise IoU between all track boxes from last frame
#         # last_obs_boxes = np.array([trk.last_observation for trk in self.active_tracks])
#         # # Ensure correct shape
#         # if last_obs_boxes.ndim == 1:
#         #     last_obs_boxes = np.expand_dims(last_obs_boxes, axis=0)
#         #
#         # if last_obs_boxes.shape[0] >= 2:  # Only compute if we have at least 2 boxes
#         #     last_obs_boxes_tensor = torch.tensor(last_obs_boxes, dtype=torch.float32)
#         #     iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#         # else:
#         #     iou_matrix = torch.zeros((0, 0))  # Skip or return dummy
#         # iou_matrix = iou_distance(last_obs_boxes, last_obs_boxes)
#         # last_obs_boxes_tensor = torch.tensor(last_obs_boxes, dtype=torch.float32)
#         # iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#
#         # Threshold to find overlapping pairs
#         # overlap_thresh = 0.5
#         # occluding_pairs = [
#         #     (i, j) for i in range(len(self.active_tracks)) for j in range(i + 1, len(self.active_tracks))
#         #     if iou_matrix[i, j] > overlap_thresh
#         # ]
#
#         ###########################################################################
#         # depth_violations = []
#         # trk_recheck = []
#         # det_recheck = []
#         #
#         # for d_idx, t_idx in matched:
#         #     track = self.active_tracks[t_idx]
#         #     det = dets[d_idx]
#         #
#         #     # Current pseudo-depth (e.g., y + h)
#         #     z_depth = det[1] + det[3]
#         #     t_depth = track.depth
#         #     prev_depth = track.prev_depth
#         #     track.prev_depth = track.depth###############################
#         #     # Store current depth
#         #     track.depth = z_depth
#         #
#         #     # Simple violation condition: large depth swap
#         #     if prev_depth is not None:
#         #         depth_diff = abs(z_depth - prev_depth)
#         #         if depth_diff > 80:  # This is a tunable threshold in pixels
#         #             # Optionally add a smarter verification rule here
#         #             trk_recheck.append(t_idx)
#         #             det_recheck.append(d_idx)
#         #             print("Depth violation triggered!")
#         #
#         # # Remove invalid matches
#         # matched = [pair for pair in matched if not (pair[0] in det_recheck or pair[1] in trk_recheck)]
#         #
#         # # Store violating pairs CAN GET RID OF - BEN
#         # violating_pairs = list(zip(det_recheck, trk_recheck))
#         # print(f"[CHECK] Violating pairs before recheck: {violating_pairs}")
#
#         # Mark recheck sets as unmatched
#         # unmatched_dets = np.concatenate([unmatched_dets, np.array(det_recheck, dtype=int)], axis=0)
#         # unmatched_trks = np.concatenate([unmatched_trks, np.array(trk_recheck, dtype=int)], axis=0)
#         """
#         Hello - changed this was pass
#         I am not sure what to put for margin.
#         """
# ##################################################### 07/07
#         margin = 10#30
#         # Track previous depths
#         prev_depths = [trk.prev_depth for trk in self.active_tracks]
#         # Match map: track_idx -> det_idx
#         track_to_det = {t_idx: d_idx for d_idx, t_idx in matched}
#
#         for tA, tB in occluding_pairs:
#             if tA not in track_to_det or tB not in track_to_det:
#                 continue  # one wasn't matched this frame
#
#             dA, dB = track_to_det[tA], track_to_det[tB]
#             detA_depth = dets[dA][1] + dets[dA][3]
#             detB_depth = dets[dB][1] + dets[dB][3]
#
#             prevA = prev_depths[tA]
#             prevB = prev_depths[tB]
#
#            # print(f"Checking pair tA={tA}, tB={tB}")
#             #print(f"prevA={prevA}, prevB={prevB}, currA={detA_depth}, currB={detB_depth}")
#
#             # Check if the previous depth relationship flipped
#             if prevA is not None and prevB is not None:
#                 if prevA > prevB + margin and detA_depth < detB_depth - margin:
#                     #print(f"[FLIP DETECTED] Track {tA} (was front) now behind Track {tB}")
#                     # Try to swap the matches if it's a clean 2-way switch
#                     if (dA, tB) in matched and (dB, tA) in matched:
#                         matched.remove((dA, tB))
#                         matched.remove((dB, tA))
#                         matched.append((dA, tA))
#                         matched.append((dB, tB))
#                         print(f"!!!!!!!!!!!!!!!!!!!!![FIX] Swapped matches to restore front-back order")
#                     # else:
#                     #     print(f"[SKIP] Not a clean switch, can't flip safely")
#
#         ###################################################
#         ############################################################################
#         # Here is Reassociation for Depth-Violating Tracks (Recheck)
#         ############################################################################
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                        ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                     trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])
##############################################################################
#Above is whatever - rules and depth cascade I think, below is PROPER depth cascade association
##############################################################################
### Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.depth = self.compute_depth(self.get_state()[0])  #######################################
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def compute_depth(self, bbox): ###############################################################
#         return bbox[3]  # bottom of bbox
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# class DeepOCSort(BaseTracker):
#     def __init__(
#             self,
#             model_weights=None,
#             device='cuda:0',
#             fp16=False,
#             per_class=False,
#             det_thresh=0.3,
#             max_age=30,
#             min_hits=3,
#             iou_threshold=0.3,
#             delta_t=3,
#             asso_func="iou",
#             inertia=0.2,
#             w_association_emb=0.5,
#             alpha_fixed_emb=0.95,
#             aw_param=0.5,
#             embedding_off=False,
#             cmc_off=True,
#             aw_off=False,
#             new_kf_off=False,
#             custom_features=False,
#             **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#         # (M detections X N tracks, final score)
#
#         # if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
#         #     stage1_emb_cost = None
#         # else:
#         #     stage1_emb_cost = dets_embs @ trk_embs.T
#
#         # Detections: [x1, y1, x2, y2, score, cls, det_ind]
#         # Track predictions: [x1, y1, x2, y2, 0]
#
# #####################################################
#         # def compute_depth(bbox):
#         #     # Pseudo-depth from bottom of box
#         #     return bbox[1] + bbox[3]  # y1 + h ≈ y2
#         #
#         # bin_width = 100  # adjust based on dataset scale (pixels)
#         #
#         # # Group detections and tracks into bins
#         # depth_bins = {}
#         #
#         # for d_idx, det in enumerate(dets):
#         #     y1, h = det[1], det[3]
#         #     depth = compute_depth([0, y1, 0, h])  # use y + h
#         #     bin_id = int(depth // bin_width)
#         #     depth_bins.setdefault(bin_id, {"dets": [], "dets_idx": []})
#         #     depth_bins[bin_id]["dets"].append(det)
#         #     depth_bins[bin_id]["dets_idx"].append(d_idx)
#         #
#         # for t_idx, trk in enumerate(trks):
#         #     y1, h = trk[1], trk[3]
#         #     depth = compute_depth([0, y1, 0, h])
#         #     bin_id = int(depth // bin_width)
#         #     depth_bins.setdefault(bin_id, {}).setdefault("trks", []).append(trk)
#         #     depth_bins.setdefault(bin_id, {}).setdefault("trks_idx", []).append(t_idx)
#         #
#         # matched = []
#         # unmatched_dets = set(range(len(dets)))
#         # unmatched_trks = set(range(len(trks)))
#         #
#         # for bin_data in depth_bins.values():
#         #     if "dets" not in bin_data or "trks" not in bin_data:
#         #         continue  # skip if only dets or only tracks
#         #
#         #     dets_bin = np.array(bin_data["dets"])
#         #     trks_bin = np.array(bin_data["trks"])
#         #     dets_idx_bin = bin_data["dets_idx"]
#         #     trks_idx_bin = bin_data["trks_idx"]
#         #
#         #     # Filter embeddings for bin
#         #     dets_embs_bin = dets_embs[dets_idx_bin]
#         #     trks_embs_bin = trk_embs[trks_idx_bin]
#         #     k_obs_bin = k_observations[trks_idx_bin]
#         #     vel_bin = velocities[trks_idx_bin]
#         #
#         #     if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#         #         emb_cost = None
#         #     else:
#         #         emb_cost = dets_embs_bin @ trks_embs_bin.T
#         #
#         #     # Run association in the bin
#         #     matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#         #         dets_bin[:, 0:5],
#         #         trks_bin,
#         #         self.asso_func,
#         #         self.iou_threshold,
#         #         vel_bin,
#         #         k_obs_bin,
#         #         self.inertia,
#         #         self.width,
#         #         self.height,
#         #         emb_cost,
#         #         self.w_association_emb,
#         #         self.aw_off,
#         #         self.aw_param,
#         #     )
#         #
#         #     # Adjust matched indices back to global indices
#         #     for d_local, t_local in matched_bin:
#         #         matched.append((dets_idx_bin[d_local], trks_idx_bin[t_local]))
#         #         unmatched_dets.discard(dets_idx_bin[d_local])
#         #         unmatched_trks.discard(trks_idx_bin[t_local])
#         #
#         # unmatched_dets = np.array(list(unmatched_dets))
#         # unmatched_trks = np.array(list(unmatched_trks))
#         #
#         # ##################################################### removed the below and replaced with the above.
#         # # matched, unmatched_dets, unmatched_trks = associate(
#         # #     dets[:, 0:5],
#         # #     trks,
#         # #     self.asso_func,
#         # #     self.iou_threshold,
#         # #     velocities,
#         # #     k_observations,
#         # #     self.inertia,
#         # #     img.shape[1],  # w
#         # #     img.shape[0],  # h
#         # #     stage1_emb_cost,
#         # #     self.w_association_emb,
#         # #     self.aw_off,
#         # #     self.aw_param,
#         # # )
#         # for m in matched:
#         #     self.active_tracks[m[1]].update(dets[m[0], :])
#         #     self.active_tracks[m[1]].update_emb(
#         #         dets_embs[m[0]], alpha=dets_alpha[m[0]])
# #########################################################################
#         def compute_depth(bbox, img_height):
#             return img_height - (bbox[1] + bbox[3])  # bottom of box to bottom of image
#
#         k_levels = 5  # number of depth levels (can tune)
#         img_height = self.height
#
#         # Compute pseudo-depths for dets and trks
#         det_depths = np.array([compute_depth(d, img_height) for d in dets])
#         trk_depths = np.array([compute_depth(t, img_height) for t in trks])
#
#         # depth_min = min(det_depths.min(), trk_depths.min())
#         # depth_max = max(det_depths.max(), trk_depths.max())
#         if len(det_depths) == 0 and len(trk_depths) == 0:
#             return [], np.array([]), np.array([])  # Nothing to match
#
#         elif len(det_depths) == 0:
#             depth_min = trk_depths.min()
#             depth_max = trk_depths.max()
#         elif len(trk_depths) == 0:
#             depth_min = det_depths.min()
#             depth_max = det_depths.max()
#         else:
#             depth_min = min(det_depths.min(), trk_depths.min())
#             depth_max = max(det_depths.max(), trk_depths.max())
#
#         depth_bins = np.linspace(depth_min, depth_max, k_levels + 1)
#
#         # Assign dets/trks to bins
#         dets_bins = [[] for _ in range(k_levels)]
#         trks_bins = [[] for _ in range(k_levels)]
#         dets_idx_bins = [[] for _ in range(k_levels)]
#         trks_idx_bins = [[] for _ in range(k_levels)]
#
#         for i, d in enumerate(dets):
#             bin_id = np.searchsorted(depth_bins, det_depths[i], side='right') - 1
#             bin_id = np.clip(bin_id, 0, k_levels - 1)
#             dets_bins[bin_id].append(d)
#             dets_idx_bins[bin_id].append(i)
#
#         for i, t in enumerate(trks):
#             bin_id = np.searchsorted(depth_bins, trk_depths[i], side='right') - 1
#             bin_id = np.clip(bin_id, 0, k_levels - 1)
#             trks_bins[bin_id].append(t)
#             trks_idx_bins[bin_id].append(i)
#
#         matched = []
#         unmatched_dets = set(range(len(dets)))
#         unmatched_trks = set(range(len(trks)))
#
#         carry_dets, carry_dets_idx = [], []
#         carry_trks, carry_trks_idx = [], []
#
#         for bin_id in range(k_levels):
#             dets_bin = np.array(dets_bins[bin_id] + carry_dets)
#             trks_bin = np.array(trks_bins[bin_id] + carry_trks)
#             dets_idx_bin = dets_idx_bins[bin_id] + carry_dets_idx
#             trks_idx_bin = trks_idx_bins[bin_id] + carry_trks_idx
#
#             if len(dets_bin) == 0 or len(trks_bin) == 0:
#                 carry_dets, carry_dets_idx = dets_bin.tolist(), dets_idx_bin
#                 carry_trks, carry_trks_idx = trks_bin.tolist(), trks_idx_bin
#                 continue
#
#             dets_embs_bin = dets_embs[dets_idx_bin]
#             trks_embs_bin = trk_embs[trks_idx_bin]
#             k_obs_bin = k_observations[trks_idx_bin]
#             vel_bin = velocities[trks_idx_bin]
#
#             if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#                 emb_cost = None
#             else:
#                 emb_cost = dets_embs_bin @ trks_embs_bin.T
#
#             matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#                 dets_bin[:, 0:5],
#                 trks_bin,
#                 self.asso_func,
#                 self.iou_threshold,
#                 vel_bin,
#                 k_obs_bin,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#
#             for d_local, t_local in matched_bin:
#                 d_global = dets_idx_bin[d_local]
#                 t_global = trks_idx_bin[t_local]
#                 matched.append((d_global, t_global))
#                 unmatched_dets.discard(d_global)
#                 unmatched_trks.discard(t_global)
#
#             # Carry unmatched forward
#             carry_dets = [dets_bin[i] for i in unmatched_d_bin]
#             carry_dets_idx = [dets_idx_bin[i] for i in unmatched_d_bin]
#             carry_trks = [trks_bin[i] for i in unmatched_t_bin]
#             carry_trks_idx = [trks_idx_bin[i] for i in unmatched_t_bin]
#
#         unmatched_dets = np.array(list(unmatched_dets))
#         unmatched_trks = np.array(list(unmatched_trks))
#
#         # Update matched tracks
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                        ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                     trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])


##############################################################################
# Above is PROPER depth, below is PROPER depth cascade association and rules
##############################################################################
### Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
#
# import numpy as np
# from collections import deque
#
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
# from boxmot.motion.cmc import get_cmc_method
# from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
# from boxmot.utils.association import associate, linear_assignment
# from boxmot.utils.iou import get_asso_func
# from boxmot.trackers.basetracker import BaseTracker
# from boxmot.utils import PerClassDecorator
#
#
# def k_previous_obs(observations, cur_age, k):
#     if len(observations) == 0:
#         return [-1, -1, -1, -1, -1]
#     for i in range(k):
#         dt = k - i
#         if cur_age - dt in observations:
#             return observations[cur_age - dt]
#     max_age = max(observations.keys())
#     return observations[max_age]
#
#
# def convert_bbox_to_z(bbox):
#     """
#     Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#       [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#       the aspect ratio
#     """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     s = w * h  # scale is just area
#     r = w / float(h + 1e-6)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_bbox_to_z_new(bbox):
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.0
#     y = bbox[1] + h / 2.0
#     return np.array([x, y, w, h]).reshape((4, 1))
#
#
# def convert_x_to_bbox_new(x):
#     x, y, w, h = x.reshape(-1)[:4]
#     return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#     Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#       [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#     """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score is None:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))
#
#
# def speed_direction(bbox1, bbox2):
#     cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
#     cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
#     speed = np.array([cy2 - cy1, cx2 - cx1])
#     norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
#     return speed / norm
#
#
# def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
#     Q = np.diag(
#         ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
#          (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
#     )
#     return Q
#
#
# def new_kf_measurement_noise(w, h, m=1 / 20):
#     w_var = (m * w) ** 2
#     h_var = (m * h) ** 2
#     R = np.diag((w_var, h_var, w_var, h_var))
#     return R
#
#
# class KalmanBoxTracker(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.
#     """
#
#     count = 0
#
#     def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
#         """
#         Initialises a tracker using initial bounding box.
#
#         """
#         # define constant velocity model
#         self.max_obs = max_obs
#         self.new_kf = new_kf
#         bbox = det[0:5]
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#         # Inside KalmanBoxTracker.__init__
#         self.prev_depth = None ################################## 20/07
#         # self.flip_violating_ids = set() #######################21/07
#
#         if new_kf:
#             self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
#             self.kf.F = np.array(
#                 [
#                     # x y w h x' y' w' h'
#                     [1, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0, 0],
#                 ]
#             )
#             _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
#             self.kf.P = new_kf_process_noise(w, h)
#             self.kf.P[:4, :4] *= 4
#             self.kf.P[4:, 4:] *= 100
#             # Process and measurement uncertainty happen in functions
#             self.bbox_to_z_func = convert_bbox_to_z_new
#             self.x_to_bbox_func = convert_x_to_bbox_new
#         else:
#             self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
#             self.kf.F = np.array(
#                 [
#                     # x  y  s  r  x' y' s'
#                     [1, 0, 0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0, 0, 1],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                 ]
#             )
#             self.kf.H = np.array(
#                 [
#                     [1, 0, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                 ]
#             )
#             self.kf.R[2:, 2:] *= 10.0
#             # give high uncertainty to the unobservable initial velocities
#             self.kf.P[4:, 4:] *= 1000.0
#             self.kf.P *= 10.0
#             self.kf.Q[-1, -1] *= 0.01
#             self.kf.Q[4:, 4:] *= 0.01
#             self.bbox_to_z_func = convert_bbox_to_z
#             self.x_to_bbox_func = convert_x_to_bbox
#
#         self.kf.x[:4] = self.bbox_to_z_func(bbox)
#
#         self.depth = self.compute_depth(self.get_state()[0])  #######################################
#         # print(self.get_state()[0])
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = deque([], maxlen=self.max_obs)
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
#         let's bear it for now.
#         """
#         # Used for OCR
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         # Used to output track after min_hits reached
#         self.features = deque([], maxlen=self.max_obs)
#         # Used for velocity
#         self.observations = dict()
#         self.velocity = None
#         self.delta_t = delta_t
#         self.history_observations = deque([], maxlen=self.max_obs)
#
#         self.emb = emb
#
#         self.frozen = False
#
#     def compute_depth(self, bbox):  ###############################################################
#         #print(f"[DEPTH DEBUG] bbox = {bbox}")
#         return bbox[3]  # bottom of bbox
#
#     def update(self, det):
#         """
#         Updates the state vector with observed bbox.
#         """
#
#         if det is not None:
#             bbox = det[0:5]
#
#             self.conf = det[4]
#             self.cls = det[5]
#             self.det_ind = det[6]
#             self.frozen = False
#
#             if self.last_observation.sum() >= 0:  # no previous observation
#                 previous_box = None
#                 for dt in range(self.delta_t, 0, -1):
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#
#             # ✅ Fix: Update prev_depth BEFORE overwriting depth ########## 22/07
#             #####################################
#             # self.prev_depth = self.depth
#             # self.depth = self.compute_depth(self.get_state()[0])
#             self.prev_depth = self.compute_depth(self.last_observation)  # based on actual observed bbox
#
#             ################################################ 23/07 (night of 22)
#             # print("STATE",self.get_state()[0])
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)
#
#             self.time_since_update = 0
#             self.hits += 1
#             self.hit_streak += 1
#
#             if self.new_kf:
#                 R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#                 self.kf.update(self.bbox_to_z_func(bbox), R=R)
#             else:
#                 self.kf.update(self.bbox_to_z_func(bbox))
#         else:
#             self.kf.update(det)
#             self.frozen = True
#             self.prev_depth = self.depth  # store previous depth #####################20/07#############################
#             self.depth = self.compute_depth(self.get_state()[0])  # update current #######20/07##########
#
#     def update_emb(self, emb, alpha=0.9):
#         self.emb = alpha * self.emb + (1 - alpha) * emb
#         self.emb /= np.linalg.norm(self.emb)
#
#     def get_emb(self):
#         # self.features.append(self.emb)
#         return self.emb
#
#     def apply_affine_correction(self, affine):
#         m = affine[:, :2]
#         t = affine[:, 2].reshape(2, 1)
#         # For OCR
#         if self.last_observation.sum() > 0:
#             ps = self.last_observation[:4].reshape(2, 2).T
#             ps = m @ ps + t
#             self.last_observation[:4] = ps.T.reshape(-1)
#
#         # Apply to each box in the range of velocity computation
#         for dt in range(self.delta_t, -1, -1):
#             if self.age - dt in self.observations:
#                 ps = self.observations[self.age - dt][:4].reshape(2, 2).T
#                 ps = m @ ps + t
#                 self.observations[self.age - dt][:4] = ps.T.reshape(-1)
#
#         # Also need to change kf state, but might be frozen
#         self.kf.apply_affine_correction(m, t, self.new_kf)
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         # Don't allow negative bounding boxes
#         if self.new_kf:
#             if self.kf.x[2] + self.kf.x[6] <= 0:
#                 self.kf.x[6] = 0
#             if self.kf.x[3] + self.kf.x[7] <= 0:
#                 self.kf.x[7] = 0
#
#             # Stop velocity, will update in kf during OOS
#             if self.frozen:
#                 self.kf.x[6] = self.kf.x[7] = 0
#             Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
#         else:
#             if (self.kf.x[6] + self.kf.x[2]) <= 0:
#                 self.kf.x[6] *= 0.0
#             Q = None
#
#         self.kf.predict(Q=Q)
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.x_to_bbox_func(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.x_to_bbox_func(self.kf.x)
#
#     def mahalanobis(self, bbox):
#         """Should be run after a predict() call for accuracy."""
#         return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
#
#
# import cv2
# import time
# class DeepOCSort(BaseTracker):
#     def __init__(
#             self,
#             model_weights=None,
#             device='cuda:0',
#             fp16=False,
#             per_class=False,
#             det_thresh=0.3,
#             max_age=30,
#             min_hits=3,
#             iou_threshold=0.3,
#             delta_t=3,
#             asso_func="iou",
#             inertia=0.2,
#             w_association_emb=0.5,
#             alpha_fixed_emb=0.95,
#             aw_param=0.5,
#             embedding_off=False,
#             cmc_off=True,
#             aw_off=False,
#             new_kf_off=False,
#             custom_features=False,
#             **kwargs
#     ):
#         super().__init__(max_age=max_age)
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.det_thresh = det_thresh
#         self.delta_t = delta_t
#         self.asso_func = get_asso_func(asso_func)
#         self.inertia = inertia
#         self.w_association_emb = w_association_emb
#         self.alpha_fixed_emb = alpha_fixed_emb
#         self.aw_param = aw_param
#         self.per_class = per_class
#         self.custom_features = custom_features
#         self.flip_violating_ids = set() ################21/07
#         self.occlud_actv_trk_ids = set()
#         self.depth_fixes=0
#         KalmanBoxTracker.count = 1
#
#         if not self.custom_features:
#             assert model_weights is not None, "Model weights must be provided for custom features"
#
#             rab = ReidAutoBackend(
#                 weights=model_weights, device=device, half=fp16
#             )
#
#             self.model = rab.get_backend()
#
#         # "similarity transforms using feature point extraction, optical flow, and RANSAC"
#         self.cmc = get_cmc_method('sof')()
#         self.embedding_off = embedding_off
#         self.cmc_off = cmc_off
#         self.aw_off = aw_off
#         self.new_kf_off = new_kf_off
#         self.removed_tracks = []
#
#     @PerClassDecorator
#     def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
#         """
#         Params:
#           dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#         Requires: this method must be called once for each frame even with empty detections
#         (use np.empty((0, 5)) for frames without detections).
#         Returns the a similar array, where the last column is the object ID.
#         NOTE: The number of objects returned may differ from the number of detections provided.
#         """
#         # dets, s, c = dets.data
#         # print(dets, s, c)
#         assert isinstance(
#             dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
#         assert isinstance(
#             img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
#         assert len(
#             dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
#         assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
#
#         self.frame_count += 1
#         self.height, self.width = img.shape[:2]
#
#         scores = dets[:, 4]
#         dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
#         assert dets.shape[1] == 7
#
#         remain_inds = scores > self.det_thresh
#
#         dets = dets[remain_inds]
#
#         # appearance descriptor extraction
#         if self.embedding_off or dets.shape[0] == 0:
#             dets_embs = np.ones((dets.shape[0], 1))
#         elif embs is not None:
#             dets_embs = embs
#         else:
#             # (Ndets x ReID_DIM) [34 x 512]
#             # dets_embs = self.model.get_features(dets[:, 0:4], img)
#             # Generate with 1 if no embedding
#             dets_embs = np.ones((dets.shape[0], 1))
#
#         # CMC
#         if not self.cmc_off:
#             print(f'\nUsing CMC\n')
#             transform = self.cmc.apply(img, dets[:, :4])
#             for trk in self.active_tracks:
#                 trk.apply_affine_correction(transform)
#
#         trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
#         af = self.alpha_fixed_emb
#         # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
#         dets_alpha = af + (1 - af) * (1 - trust)
#
#         # get predicted locations from existing trackers.
#         trks = np.zeros((len(self.active_tracks), 5))
#         trk_embs = []
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.active_tracks[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#             else:
#                 trk_embs.append(self.active_tracks[t].get_emb())
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#
#         if len(trk_embs) > 0:
#             trk_embs = np.vstack(trk_embs)
#         else:
#             trk_embs = np.array(trk_embs)
#
#         for t in reversed(to_del):
#             self.active_tracks.pop(t)
#
#         velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
#             (0, 0)) for trk in self.active_tracks])
#         last_boxes = np.array(
#             [trk.last_observation for trk in self.active_tracks])
#         k_observations = np.array([k_previous_obs(
#             trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
#
#         """
#             First round of association
#         """
#
#         def compute_depth(bbox, img_height):
#             return bbox[3]
#             #return img_height - (bbox[1] + bbox[3])  # bottom of box to bottom of image
#
#         k_levels = 5  # number of depth levels (can tune)
#         img_height = self.height
#         # print(self.height,self.width)
#
#         # Compute pseudo-depths for dets and trks
#         det_depths = np.array([compute_depth(d, img_height) for d in dets])
#         trk_depths = np.array([compute_depth(t, img_height) for t in trks])
#
#         # depth_min = min(det_depths.min(), trk_depths.min())
#         # depth_max = max(det_depths.max(), trk_depths.max())
#         if len(det_depths) == 0 and len(trk_depths) == 0:
#             return [], np.array([]), np.array([])  # Nothing to match
#
#         elif len(det_depths) == 0:
#             depth_min = trk_depths.min()
#             depth_max = trk_depths.max()
#         elif len(trk_depths) == 0:
#             depth_min = det_depths.min()
#             depth_max = det_depths.max()
#         else:
#             depth_min = min(det_depths.min(), trk_depths.min())
#             depth_max = max(det_depths.max(), trk_depths.max())
#
#         depth_bins = np.linspace(depth_min, depth_max, k_levels + 1)
#
#         # Assign dets/trks to bins
#         dets_bins = [[] for _ in range(k_levels)]
#         trks_bins = [[] for _ in range(k_levels)]
#         dets_idx_bins = [[] for _ in range(k_levels)]
#         trks_idx_bins = [[] for _ in range(k_levels)]
#
#         for i, d in enumerate(dets):
#             bin_id = np.searchsorted(depth_bins, det_depths[i], side='right') - 1
#             bin_id = np.clip(bin_id, 0, k_levels - 1)
#             dets_bins[bin_id].append(d)
#             dets_idx_bins[bin_id].append(i)
#
#         for i, t in enumerate(trks):
#             bin_id = np.searchsorted(depth_bins, trk_depths[i], side='right') - 1
#             bin_id = np.clip(bin_id, 0, k_levels - 1)
#             trks_bins[bin_id].append(t)
#             trks_idx_bins[bin_id].append(i)
#
#         matched = []
#         unmatched_dets = set(range(len(dets)))
#         unmatched_trks = set(range(len(trks)))
#
#         carry_dets, carry_dets_idx = [], []
#         carry_trks, carry_trks_idx = [], []
#
#         for bin_id in range(k_levels):
#             dets_bin = np.array(dets_bins[bin_id] + carry_dets)
#             trks_bin = np.array(trks_bins[bin_id] + carry_trks)
#             dets_idx_bin = dets_idx_bins[bin_id] + carry_dets_idx
#             trks_idx_bin = trks_idx_bins[bin_id] + carry_trks_idx
#
#             if len(dets_bin) == 0 or len(trks_bin) == 0:
#                 carry_dets, carry_dets_idx = dets_bin.tolist(), dets_idx_bin
#                 carry_trks, carry_trks_idx = trks_bin.tolist(), trks_idx_bin
#                 continue
#
#             dets_embs_bin = dets_embs[dets_idx_bin]
#             trks_embs_bin = trk_embs[trks_idx_bin]
#             k_obs_bin = k_observations[trks_idx_bin]
#             vel_bin = velocities[trks_idx_bin]
#
#             if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
#                 emb_cost = None
#             else:
#                 emb_cost = dets_embs_bin @ trks_embs_bin.T
#
#             matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
#                 dets_bin[:, 0:5],
#                 trks_bin,
#                 self.asso_func,
#                 self.iou_threshold,
#                 vel_bin,
#                 k_obs_bin,
#                 self.inertia,
#                 self.width,
#                 self.height,
#                 emb_cost,
#                 self.w_association_emb,
#                 self.aw_off,
#                 self.aw_param,
#             )
#
#             for d_local, t_local in matched_bin:
#                 d_global = dets_idx_bin[d_local]
#                 t_global = trks_idx_bin[t_local]
#                 matched.append((d_global, t_global))
#                 unmatched_dets.discard(d_global)
#                 unmatched_trks.discard(t_global)
#
#             # Carry unmatched forward
#             carry_dets = [dets_bin[i] for i in unmatched_d_bin]
#             carry_dets_idx = [dets_idx_bin[i] for i in unmatched_d_bin]
#             carry_trks = [trks_bin[i] for i in unmatched_t_bin]
#             carry_trks_idx = [trks_idx_bin[i] for i in unmatched_t_bin]
#
#         unmatched_dets = np.array(list(unmatched_dets))
#         unmatched_trks = np.array(list(unmatched_trks))
#
#         # Update matched tracks
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         #print("matched",matched)
#         #############################################################################
#         # Below is looking at which violates rules
#         #############################################################################
#         # self.flip_violating_ids = set() ###############21/07
#         import torch
#         from torchvision.ops import box_iou
#         # Map: index in active_tracks -> track ID #####################21/07
#         # Only use tracks that are visualized (i.e., with enough history) #######22/07
#         visible_tracks = [trk for trk in self.active_tracks if len(trk.history_observations) > 2]
#         index_to_id = {i: trk.id for i, trk in enumerate(self.active_tracks)}
#         ################
#         if len(visible_tracks) >= 2:
#             last_obs_boxes = np.array([trk.last_observation[:4] for trk in visible_tracks], ndmin=2)
#
#             # Convert to [x1, y1, x2, y2]
#             last_obs_boxes_xyxy = last_obs_boxes.copy()
#             # last_obs_boxes_xyxy = np.zeros_like(last_obs_boxes)  #########22/07
#             # last_obs_boxes_xyxy[:, 0] = last_obs_boxes[:, 0]
#             # last_obs_boxes_xyxy[:, 1] = last_obs_boxes[:, 1]
#             # last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]
#             # last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]
#
#             ###############
#           #  debug_img = img.copy()  # make a copy to avoid overwriting the main visualization
#
#
#
#             # for i, box in enumerate(last_obs_boxes_xyxy):
#             #     x1, y1, x2, y2 = map(int, box)
#             #     trk_id = self.active_tracks[i].id
#             #     color = (255, 0, 255)  # magenta
#             #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
#             #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             #
#             #
#             # cv2.imshow('Debug Boxes', debug_img)
#             # cv2.waitKey(2)
#          #   scale_factor = 0.5  # Adjust this as needed
#
#             # for i, box in enumerate(last_obs_boxes_xyxy):
#             #     x1, y1, x2, y2 = map(int, box)
#             #     trk_id = self.active_tracks[i].id
#             #     color = (255, 0, 255)  # magenta
#             #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
#             #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#             # for trk, box in zip(self.active_tracks, last_obs_boxes_xyxy):
#             #     x1, y1, x2, y2 = map(int, box)
#             #     trk_id = trk.id
#             #     color = (255, 0, 255)  # magenta
#             #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
#             #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             #
#             # # Resize for display
#             # resized_debug_img = cv2.resize(debug_img, (0, 0), fx=scale_factor, fy=scale_factor)
#             # cv2.imshow('Debug Boxes', resized_debug_img)
#             # cv2.waitKey(0)  # Wait until a key is pressed
#             # cv2.destroyAllWindows()
#
#             ######################
#             last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
#             iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#
#             overlap_thresh = 0.15
#             occluding_pairs = [
#                 (index_to_id[i], index_to_id[j])
#                 for i in range(len(visible_tracks))
#                 for j in range(i + 1, len(visible_tracks))
#                 if iou_matrix[i, j] > overlap_thresh
#             ]
#         else:
#             occluding_pairs = []
#
#         #################
#
#         # if len(self.active_tracks) >= 2:
#         #     last_obs_boxes = np.array([trk.last_observation[:4] for trk in self.active_tracks], ndmin=2)
#         #
#         #     # Convert to [x1, y1, x2, y2]
#         #     # last_obs_boxes_xyxy = last_obs_boxes.copy()
#         #     last_obs_boxes_xyxy = np.zeros_like(last_obs_boxes)
#         #     last_obs_boxes_xyxy[:, 0] = last_obs_boxes[:, 0]
#         #     last_obs_boxes_xyxy[:, 1] = last_obs_boxes[:, 1]
#         #     last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]
#         #     last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]
#         #
#         #     last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
#         #     iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
#         #
#         #     overlap_thresh = 0.6
#         #     # occluding_pairs = [
#         #     #     (i, j) for i in range(len(self.active_tracks)) for j in range(i + 1, len(self.active_tracks))
#         #     #     if iou_matrix[i, j] > overlap_thresh
#         #     # ]
#         #     occluding_pairs = [
#         #         (index_to_id[i], index_to_id[j])
#         #         for i in range(len(self.active_tracks))
#         #         for j in range(i + 1, len(self.active_tracks))
#         #         if iou_matrix[i, j] > overlap_thresh
#         #     ]
#         #
#         # else:
#         #     occluding_pairs = []
#
#         """
#                 Hello - changed this was pass
#                 I am not sure what to put for margin.
#         """
#         ##################################################### 07/07
#         margin = 50  # 30
#         # Track previous depths
#         # prev_depths = [trk.prev_depth for trk in self.active_tracks]
#         # Match map: track_idx -> det_idx
#         index_to_id = {i: trk.id for i, trk in enumerate(self.active_tracks)}
#         track_to_det = {
#             index_to_id[t_idx]: d_idx
#             for d_idx, t_idx in matched
#             if t_idx in index_to_id
#         }
#         # track_to_det = {t_idx: d_idx for d_idx, t_idx in matched}############ no +1 originally
#         #print("track to det",track_to_det)
#
#         #print(occluding_pairs)
#         #print("[DEBUG] Occluding track ID pairs:")
#         for i, j in occluding_pairs:
#             #print(f"({i}, {j})")
#             self.occlud_actv_trk_ids.update([i, j])  #################### 21/07
#
#         # if len(occluding_pairs)>0:
#         #     print("[DEBUG] IOU matrix:")
#         #     print(iou_matrix.numpy())  # if you're using torch tensors
#         #     time.sleep(10)
#
#         ######################## 22/07
#         id_to_track = {trk.id: trk for trk in self.active_tracks}
#         track_to_det = {t_id: d_id for d_id, t_id in matched}
#
#         # Assert that all track IDs in matched exist in active_tracks
#         # for _, tid in matched:
#         #     assert tid in id_to_track, f"Matched track ID {tid} not in active tracks!"
#         # print(track_to_det)
#         # print("Active track IDs:", [trk.id for trk in self.active_tracks])
#         # for trk in self.active_tracks:
#         #     print(f"Track ID: {trk.id}, Last Obs: {trk.last_observation}, Prev Depth: {trk.prev_depth}")
#         # print("before occlud loop")
#         #time.sleep(10)
#
#         for tA, tB in occluding_pairs:
#             if tA not in track_to_det or tB not in track_to_det:
#                 continue  # not matched this frame
#
#             trkA = id_to_track[tA]
#             trkB = id_to_track[tB]
#             # Previous depths
#             prevA = trkA.prev_depth
#             prevB = trkB.prev_depth
#
#             State_A = trkA.get_state()[0]
#             State_B = trkB.get_state()[0]
#             # print("State A", State_A, "State B", State_B )
#             # Get detection indices
#             dA = track_to_det[tA]
#             dB = track_to_det[tB]
#
#             # Now you’re safe to calculate depths etc.
#             # print(dets[dA])
#             ####################
#             # for trk in self.active_tracks:
#             #     print(f"Track ID: {trk.id}, Last Obs: {trk.last_observation}, Prev Depth: {trk.prev_depth}")
#             #     import matplotlib.pyplot as plt
#             #     import matplotlib.patches as patches
#             #
#             #     # img: your current frame (numpy array, shape [H, W, 3])
#             #     # dets[dA]: detection bbox
#             #     # trk.last_observation: track's previous bbox
#             #
#             #     fig, ax = plt.subplots(figsize=(10, 8))
#             #     ax.imshow(img)
#             #
#             #     # Draw detection box (in RED)
#             #     x1, y1, x2, y2 = dets[dA][:4]
#             #     rect_det = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none',
#             #                                  label='Detection')
#             #     ax.add_patch(rect_det)
#             #
#             #     # Draw tracker box (in GREEN)
#             #     x1t, y1t, x2t, y2t = trk.last_observation[:4]
#             #     rect_trk = patches.Rectangle((x1t, y1t), x2t - x1t, y2t - y1t, linewidth=2, edgecolor='g',
#             #                                  facecolor='none', label='Track')
#             #     ax.add_patch(rect_trk)
#             #
#             #     # Optional: add labels
#             #     ax.text(x1, y1 - 10, 'Det', color='r')
#             #     ax.text(x1t, y1t - 10, 'Track', color='g')
#             #
#             #     plt.title(f"Det vs Track (dA={dA}, tA={tA})")
#             #     plt.legend()
#             #     plt.axis('off')
#             #     plt.show()
#
#             ####################################
#             # time.sleep(60)
#             detA_depth = dets[dA][3] #dets[dA][1] + dets[dA][3]
#             detB_depth = dets[dB][3] #dets[dB][1] + dets[dB][3]
#
#             # print(f"Checking pair tA={tA}, tB={tB}")
#             # print(f"prevA={prevA}, prevB={prevB}, currA={detA_depth}, currB={detB_depth}")
#
#             # Check if the previous depth relationship flipped
#             if prevA is not None and prevB is not None:
#                 # print("hello2")
#                 # print("prevA", prevA)
#                 # print("prevB", prevB)
#                 # print("detA_depth", detA_depth)
#                 # print("detB_depth", detB_depth)
#                 #time.sleep(10)
#                 # if prevA > prevB + margin and detA_depth < detB_depth - margin:
#                 if (
#                         (prevA > prevB + margin and detA_depth < detB_depth - margin)
#                         or
#                         (prevB > prevA + margin and detB_depth < detA_depth - margin)
#                 ):
#
#                     # trkA, trkB = self.active_tracks[tA], self.active_tracks[tB]
#                     print(f"[FLIP DETECTED] Track {trkA.id} (was front) now behind Track {trkB.id}")
#
#                     # print(f"[FLIP DETECTED] Track {tA} (was front) now behind Track {tB}")
#                     # print(f"[FLIP DETECTED] Det {dA} (was front) now behind Det {dB}")
#                     self.flip_violating_ids.update([trkA.id, trkB.id])  #################### 21/07
#                     # print(f"[FLIP] Violating IDs: {self.flip_violating_ids}")
#
#                     ###########################################
#                     # import matplotlib.pyplot as plt
#                     # import matplotlib.patches as patches
#                     #
#                     # fig, ax = plt.subplots(figsize=(12, 9))
#                     # ax.imshow(img)
#                     #
#                     # # Detection A - Red
#                     # xa1, ya1, xa2, ya2 = dets[dA][:4]
#                     # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
#                     #                                linewidth=2, edgecolor='red', facecolor='none'))
#                     # ax.text(xa1, ya1 - 5, 'detA', color='red')
#                     #
#                     # # Detection B - Orange
#                     # xb1, yb1, xb2, yb2 = dets[dB][:4]
#                     # ax.add_patch(patches.Rectangle((xb1, yb1), xb2 - xb1, yb2 - yb1,
#                     #                                linewidth=2, edgecolor='orange', facecolor='none'))
#                     # ax.text(xb1, yb1 - 5, 'detB', color='orange')
#                     #
#                     # # Track A - Green
#                     # xta1, yta1, xta2, yta2 = trkA.last_observation[:4]
#                     # ax.add_patch(patches.Rectangle((xta1, yta1), xta2 - xta1, yta2 - yta1,
#                     #                                linewidth=2, edgecolor='green', facecolor='none'))
#                     # ax.text(xta1, yta1 - 5, f'trkA (ID={trkA.id})', color='green')
#                     #
#                     # # Track B - Blue
#                     # xtb1, ytb1, xtb2, ytb2 = trkB.last_observation[:4]
#                     # ax.add_patch(patches.Rectangle((xtb1, ytb1), xtb2 - xtb1, ytb2 - ytb1,
#                     #                                linewidth=2, edgecolor='blue', facecolor='none'))
#                     # ax.text(xtb1, ytb1 - 5, f'trkB (ID={trkB.id})', color='black')
#                     #
#                     # #Kalman State A
#                     # xa1, ya1, xa2, ya2 = State_A[:4]
#                     # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
#                     #                                linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-'))
#                     # ax.text(xa1, ya1 - 5, 'StateA', color='yellow')
#                     # # Kalman State B
#                     # xa1, ya1, xa2, ya2 = State_B[:4]
#                     # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
#                     #                                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-'))
#                     # ax.text(xa1, ya1 - 5, 'StateB', color='cyan')
#                     #
#                     #
#                     #
#                     # ax.set_title(f"Flip Check: dA={dA}, dB={dB}, tA={tA}, tB={tB}")
#                     # ax.axis('off')
#                     # print(track_to_det)
#                     # plt.show()
#
#                     ###########################################
#
#                     # Try to swap the matches if it's a clean 2-way switch
#                     # if (dA, tB) in track_to_det and (dB, tA) in track_to_det:
#                     # if (dA, tB) in matched and (dB, tA) in matched: # dont want to use matched!
#                     #     matched.remove((dA, tB))
#                     #     matched.remove((dB, tA))
#                     #     matched.append((dA, tA))
#                     #     matched.append((dB, tB))
#                     # time.sleep(10)
#                     if (
#                             tA in track_to_det and tB in track_to_det and
#                             track_to_det[tA] == dA and
#                             track_to_det[tB] == dB
#                     ):
#
#                         # Flip the detection assignments
#                         track_to_det[tA], track_to_det[tB] = dB, dA
#
#                         # If you still need the `matched` list updated as well:
#                         matched = [(d, t) for t, d in track_to_det.items()]
#                         # print(f"!!!!!!!!!!!!!!!!!!!!![FIX] Swapped matches to restore front-back order")
#                         self.depth_fixes+=1
#                         print(self.depth_fixes)
#                         # time.sleep(5)
#
#                     # else:
#                     #     print(f"[SKIP] Not a clean switch, can't flip safely")
#
#         ###########################
#
#
#         # for tA, tB in occluding_pairs:
#         #     if tA not in track_to_det or tB not in track_to_det:
#         #         continue  # one wasn't matched this frame
#         #     print("hello1")
#         #     # dA, dB = track_to_det[tA], track_to_det[tB]
#         #     # detA_depth = dets[dA][1] + dets[dA][3]
#         #     # detB_depth = dets[dB][1] + dets[dB][3]
#         #     #
#         #     # prevA = prev_depths[tA]
#         #     # prevB = prev_depths[tB]
#         #     trkA = next(trk for trk in self.active_tracks if trk.id == tA)
#         #     trkB = next(trk for trk in self.active_tracks if trk.id == tB)
#         #
#         #     # Previous depths
#         #     prevA = trkA.prev_depth
#         #     prevB = trkB.prev_depth
#         #
#         #     # Get index in active_tracks (if needed for track_to_det)
#         #     idxA = self.active_tracks.index(trkA)
#         #     idxB = self.active_tracks.index(trkB)
#         #
#         #     dA = track_to_det.get(tA)
#         #     dB = track_to_det.get(tB)
#         #     detA_depth = dets[dA][1] + dets[dA][3]
#         #     detB_depth = dets[dB][1] + dets[dB][3]
#         #
#         #     # print(f"Checking pair tA={tA}, tB={tB}")
#         #     # print(f"prevA={prevA}, prevB={prevB}, currA={detA_depth}, currB={detB_depth}")
#         #
#         #     # Check if the previous depth relationship flipped
#         #     if prevA is not None and prevB is not None:
#         #         print("hello2")
#         #         print("prevA",prevA)
#         #         print("prevB",prevB)
#         #         print("detA_depth",detA_depth)
#         #         print("detB_depth",detB_depth)
#         #         time.sleep(10)
#         #         if prevA > prevB + margin and detA_depth < detB_depth - margin:
#         #             # trkA, trkB = self.active_tracks[tA], self.active_tracks[tB]
#         #             print(f"[FLIP DETECTED] Track {trkA.id} (was front) now behind Track {trkB.id}")
#         #
#         #             print(f"[FLIP DETECTED] Track {tA} (was front) now behind Track {tB}")
#         #             print(f"[FLIP DETECTED] Det {dA} (was front) now behind Det {dB}")
#         #             self.flip_violating_ids.update([trkA.id, trkB.id]) #################### 21/07
#         #             print(f"[FLIP] Violating IDs: {self.flip_violating_ids}")
#         #
#         #
#         #             # Try to swap the matches if it's a clean 2-way switch
#         #             if (dA, tB) in matched and (dB, tA) in matched:
#         #                 matched.remove((dA, tB))
#         #                 matched.remove((dB, tA))
#         #                 matched.append((dA, tA))
#         #                 matched.append((dB, tB))
#         #                 print(f"!!!!!!!!!!!!!!!!!!!!![FIX] Swapped matches to restore front-back order")
#         ##             else:
#         ##                 print(f"[SKIP] Not a clean switch, can't flip safely")
#
#         ###################################################
#         ############################################################################
#         # Here is Reassociation for Depth-Violating Tracks (Recheck)
#         ############################################################################
#         # Update matched tracks #########22/07
#         for m in matched:
#             self.active_tracks[m[1]].update(dets[m[0], :])
#             self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
#
#         """
#             Second round of associaton by OCR
#         """
#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_dets_embs = dets_embs[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             left_trks_embs = trk_embs[unmatched_trks]
#
#             iou_left = self.asso_func(left_dets, left_trks)
#             # TODO: is better without this
#             emb_cost_left = left_dets_embs @ left_trks_embs.T
#             if self.embedding_off:
#                 emb_cost_left = np.zeros_like(emb_cost_left)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                 NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                 get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                 uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]
#                                        ], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.active_tracks[trk_ind].update(dets[det_ind, :])
#                     self.active_tracks[trk_ind].update_emb(
#                         dets_embs[det_ind], alpha=dets_alpha[det_ind])
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(
#                     unmatched_dets, np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(
#                     unmatched_trks, np.array(to_remove_trk_indices))
#
#         for m in unmatched_trks:
#             self.active_tracks[m].update(None)
#
#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(
#                 dets[i],
#                 delta_t=self.delta_t,
#                 emb=dets_embs[i],
#                 alpha=dets_alpha[i],
#                 new_kf=not self.new_kf_off,
#                 max_obs=self.max_obs
#             )
#             self.active_tracks.append(trk)
#         i = len(self.active_tracks)
#         for trk in reversed(self.active_tracks):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 """
#                 this is optional to use the recent observation or the kalman filter prediction,
#                 we didn't notice significant difference here
#                 """
#                 d = trk.last_observation[:4]
#
#             '''
#             # self.frame_count <= self.min_hits
#             This allows for all detections to be included in the initial frames
#             (before the tracker has seen enough frames to confirm tracks).
#             '''
#             # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id], [trk.conf], [
#                     trk.cls], [trk.det_ind])).reshape(1, -1))
#
#             i -= 1
#             # remove dead tracklet
#             if trk.time_since_update > self.max_age:
#                 self.active_tracks.pop(i)
#                 self.removed_tracks.append(trk.id)
#         # self.flip_violating_ids = set()
#
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.array([])

#######################################
# Below is just rules and normal
####################################
# # Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
### Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
#
import numpy as np
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.deepocsort_kf import KalmanFilter
from boxmot.utils.association import associate, linear_assignment
from boxmot.utils.iou import get_asso_func
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
         (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, max_obs=50):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.max_obs = max_obs
        self.new_kf = new_kf
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        # Inside KalmanBoxTracker.__init__
        self.prev_depth = None ################################## 20/07
        # self.flip_violating_ids = set() #######################21/07

        if new_kf:
            self.kf = KalmanFilter(dim_x=8, dim_z=4, max_obs=max_obs)
            self.kf.F = np.array(
                [
                    # x y w h x' y' w' h'
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            # Process and measurement uncertainty happen in functions
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    # x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            # give high uncertainty to the unobservable initial velocities
            self.kf.P[4:, 4:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.depth = self.compute_depth(self.get_state()[0])  #######################################
        # print(self.get_state()[0])

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.features = deque([], maxlen=self.max_obs)
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.history_observations = deque([], maxlen=self.max_obs)

        self.emb = emb

        self.frozen = False

    def compute_depth(self, bbox):  ###############################################################
        #print(f"[DEPTH DEBUG] bbox = {bbox}")
        return bbox[3]  # bottom of bbox

    def update(self, det):
        """
        Updates the state vector with observed bbox.
        """

        if det is not None:
            bbox = det[0:5]

            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            # ✅ Fix: Update prev_depth BEFORE overwriting depth ########## 22/07
            #####################################
            # self.prev_depth = self.depth
            # self.depth = self.compute_depth(self.get_state()[0])
            self.prev_depth = self.compute_depth(self.last_observation)  # based on actual observed bbox

            ################################################ 23/07 (night of 22)
            # print("STATE",self.get_state()[0])
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(det)
            self.frozen = True
            self.prev_depth = self.depth  # store previous depth #####################20/07#############################
            self.depth = self.compute_depth(self.get_state()[0])  # update current #######20/07##########

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        # self.features.append(self.emb)
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0

            # Stop velocity, will update in kf during OOS
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


import cv2
import time
class DeepOCSort(BaseTracker):
    def __init__(
            self,
            model_weights=None,
            device='cuda:0',
            fp16=False,
            per_class=False,
            det_thresh=0.3,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            w_association_emb=0.5,
            alpha_fixed_emb=0.95,
            aw_param=0.5,
            embedding_off=False,
            cmc_off=True,
            aw_off=False,
            new_kf_off=False,
            custom_features=False,
            **kwargs
    ):
        super().__init__(max_age=max_age)
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        self.custom_features = custom_features
        self.flip_violating_ids = set() ################21/07
        self.occlud_actv_trk_ids = set()
        self.depth_fixes=0
        KalmanBoxTracker.count = 1

        if not self.custom_features:
            assert model_weights is not None, "Model weights must be provided for custom features"

            rab = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            )

            self.model = rab.get_backend()

        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = get_cmc_method('sof')()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.removed_tracks = []

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # dets, s, c = dets.data
        # print(dets, s, c)
        assert isinstance(
            dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
        assert len(
            dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        assert dets.shape[1] == 7

        remain_inds = scores > self.det_thresh

        dets = dets[remain_inds]

        # appearance descriptor extraction
        if self.embedding_off or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        elif embs is not None:
            dets_embs = embs
        else:
            # (Ndets x ReID_DIM) [34 x 512]
            # dets_embs = self.model.get_features(dets[:, 0:4], img)
            # Generate with 1 if no embedding
            dets_embs = np.ones((dets.shape[0], 1))

        # CMC
        if not self.cmc_off:
            print(f'\nUsing CMC\n')
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.active_tracks[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
            (0, 0)) for trk in self.active_tracks])
        last_boxes = np.array(
            [trk.last_observation for trk in self.active_tracks])
        k_observations = np.array([k_previous_obs(
            trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])

########################################################################
        """
            First round of association
        """
        # (M detections X N tracks, final score)

        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T

        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.asso_func,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            img.shape[1],  # w
            img.shape[0],  # h
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :])
            self.active_tracks[m[1]].update_emb(
                dets_embs[m[0]], alpha=dets_alpha[m[0]])




############################################################################
        # """
        #     First round of association
        # """

        # def compute_depth(bbox, img_height):
        #     return bbox[3]
        #     #return img_height - (bbox[1] + bbox[3])  # bottom of box to bottom of image
        #
        # k_levels = 5  # number of depth levels (can tune)
        # img_height = self.height
        # # print(self.height,self.width)
        #
        # # Compute pseudo-depths for dets and trks
        # det_depths = np.array([compute_depth(d, img_height) for d in dets])
        # trk_depths = np.array([compute_depth(t, img_height) for t in trks])
        #
        # # depth_min = min(det_depths.min(), trk_depths.min())
        # # depth_max = max(det_depths.max(), trk_depths.max())
        # if len(det_depths) == 0 and len(trk_depths) == 0:
        #     return [], np.array([]), np.array([])  # Nothing to match
        #
        # elif len(det_depths) == 0:
        #     depth_min = trk_depths.min()
        #     depth_max = trk_depths.max()
        # elif len(trk_depths) == 0:
        #     depth_min = det_depths.min()
        #     depth_max = det_depths.max()
        # else:
        #     depth_min = min(det_depths.min(), trk_depths.min())
        #     depth_max = max(det_depths.max(), trk_depths.max())
        #
        # depth_bins = np.linspace(depth_min, depth_max, k_levels + 1)
        #
        # # Assign dets/trks to bins
        # dets_bins = [[] for _ in range(k_levels)]
        # trks_bins = [[] for _ in range(k_levels)]
        # dets_idx_bins = [[] for _ in range(k_levels)]
        # trks_idx_bins = [[] for _ in range(k_levels)]
        #
        # for i, d in enumerate(dets):
        #     bin_id = np.searchsorted(depth_bins, det_depths[i], side='right') - 1
        #     bin_id = np.clip(bin_id, 0, k_levels - 1)
        #     dets_bins[bin_id].append(d)
        #     dets_idx_bins[bin_id].append(i)
        #
        # for i, t in enumerate(trks):
        #     bin_id = np.searchsorted(depth_bins, trk_depths[i], side='right') - 1
        #     bin_id = np.clip(bin_id, 0, k_levels - 1)
        #     trks_bins[bin_id].append(t)
        #     trks_idx_bins[bin_id].append(i)
        #
        # matched = []
        # unmatched_dets = set(range(len(dets)))
        # unmatched_trks = set(range(len(trks)))
        #
        # carry_dets, carry_dets_idx = [], []
        # carry_trks, carry_trks_idx = [], []
        #
        # for bin_id in range(k_levels):
        #     dets_bin = np.array(dets_bins[bin_id] + carry_dets)
        #     trks_bin = np.array(trks_bins[bin_id] + carry_trks)
        #     dets_idx_bin = dets_idx_bins[bin_id] + carry_dets_idx
        #     trks_idx_bin = trks_idx_bins[bin_id] + carry_trks_idx
        #
        #     if len(dets_bin) == 0 or len(trks_bin) == 0:
        #         carry_dets, carry_dets_idx = dets_bin.tolist(), dets_idx_bin
        #         carry_trks, carry_trks_idx = trks_bin.tolist(), trks_idx_bin
        #         continue
        #
        #     dets_embs_bin = dets_embs[dets_idx_bin]
        #     trks_embs_bin = trk_embs[trks_idx_bin]
        #     k_obs_bin = k_observations[trks_idx_bin]
        #     vel_bin = velocities[trks_idx_bin]
        #
        #     if self.embedding_off or dets_embs_bin.shape[0] == 0 or trks_embs_bin.shape[0] == 0:
        #         emb_cost = None
        #     else:
        #         emb_cost = dets_embs_bin @ trks_embs_bin.T
        #
        #     matched_bin, unmatched_d_bin, unmatched_t_bin = associate(
        #         dets_bin[:, 0:5],
        #         trks_bin,
        #         self.asso_func,
        #         self.iou_threshold,
        #         vel_bin,
        #         k_obs_bin,
        #         self.inertia,
        #         self.width,
        #         self.height,
        #         emb_cost,
        #         self.w_association_emb,
        #         self.aw_off,
        #         self.aw_param,
        #     )
        #
        #     for d_local, t_local in matched_bin:
        #         d_global = dets_idx_bin[d_local]
        #         t_global = trks_idx_bin[t_local]
        #         matched.append((d_global, t_global))
        #         unmatched_dets.discard(d_global)
        #         unmatched_trks.discard(t_global)
        #
        #     # Carry unmatched forward
        #     carry_dets = [dets_bin[i] for i in unmatched_d_bin]
        #     carry_dets_idx = [dets_idx_bin[i] for i in unmatched_d_bin]
        #     carry_trks = [trks_bin[i] for i in unmatched_t_bin]
        #     carry_trks_idx = [trks_idx_bin[i] for i in unmatched_t_bin]
        #
        # unmatched_dets = np.array(list(unmatched_dets))
        # unmatched_trks = np.array(list(unmatched_trks))
        #
        # # Update matched tracks
        # for m in matched:
        #     self.active_tracks[m[1]].update(dets[m[0], :])
        #     self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        #print("matched",matched)
        #############################################################################
        # Below is looking at which violates rules
        #############################################################################
        # self.flip_violating_ids = set() ###############21/07
        import torch
        from torchvision.ops import box_iou
        # Map: index in active_tracks -> track ID #####################21/07
        # Only use tracks that are visualized (i.e., with enough history) #######22/07
        visible_tracks = [trk for trk in self.active_tracks if len(trk.history_observations) > 2]
        index_to_id = {i: trk.id for i, trk in enumerate(self.active_tracks)}
        ################
        if len(visible_tracks) >= 2:
            last_obs_boxes = np.array([trk.last_observation[:4] for trk in visible_tracks], ndmin=2)

            # Convert to [x1, y1, x2, y2]
            last_obs_boxes_xyxy = last_obs_boxes.copy()
            # last_obs_boxes_xyxy = np.zeros_like(last_obs_boxes)  #########22/07
            # last_obs_boxes_xyxy[:, 0] = last_obs_boxes[:, 0]
            # last_obs_boxes_xyxy[:, 1] = last_obs_boxes[:, 1]
            # last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]
            # last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]

            ###############
          #  debug_img = img.copy()  # make a copy to avoid overwriting the main visualization



            # for i, box in enumerate(last_obs_boxes_xyxy):
            #     x1, y1, x2, y2 = map(int, box)
            #     trk_id = self.active_tracks[i].id
            #     color = (255, 0, 255)  # magenta
            #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            #
            #
            # cv2.imshow('Debug Boxes', debug_img)
            # cv2.waitKey(2)
         #   scale_factor = 0.5  # Adjust this as needed

            # for i, box in enumerate(last_obs_boxes_xyxy):
            #     x1, y1, x2, y2 = map(int, box)
            #     trk_id = self.active_tracks[i].id
            #     color = (255, 0, 255)  # magenta
            #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # for trk, box in zip(self.active_tracks, last_obs_boxes_xyxy):
            #     x1, y1, x2, y2 = map(int, box)
            #     trk_id = trk.id
            #     color = (255, 0, 255)  # magenta
            #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(debug_img, f'ID: {trk_id}', (x1, y1 - 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            #
            # # Resize for display
            # resized_debug_img = cv2.resize(debug_img, (0, 0), fx=scale_factor, fy=scale_factor)
            # cv2.imshow('Debug Boxes', resized_debug_img)
            # cv2.waitKey(0)  # Wait until a key is pressed
            # cv2.destroyAllWindows()

            ######################
            last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
            iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)

            overlap_thresh = 0.15
            occluding_pairs = [
                (index_to_id[i], index_to_id[j])
                for i in range(len(visible_tracks))
                for j in range(i + 1, len(visible_tracks))
                if iou_matrix[i, j] > overlap_thresh
            ]
        else:
            occluding_pairs = []

        #################

        # if len(self.active_tracks) >= 2:
        #     last_obs_boxes = np.array([trk.last_observation[:4] for trk in self.active_tracks], ndmin=2)
        #
        #     # Convert to [x1, y1, x2, y2]
        #     # last_obs_boxes_xyxy = last_obs_boxes.copy()
        #     last_obs_boxes_xyxy = np.zeros_like(last_obs_boxes)
        #     last_obs_boxes_xyxy[:, 0] = last_obs_boxes[:, 0]
        #     last_obs_boxes_xyxy[:, 1] = last_obs_boxes[:, 1]
        #     last_obs_boxes_xyxy[:, 2] = last_obs_boxes[:, 0] + last_obs_boxes[:, 2]
        #     last_obs_boxes_xyxy[:, 3] = last_obs_boxes[:, 1] + last_obs_boxes[:, 3]
        #
        #     last_obs_boxes_tensor = torch.tensor(last_obs_boxes_xyxy, dtype=torch.float32)
        #     iou_matrix = box_iou(last_obs_boxes_tensor, last_obs_boxes_tensor)
        #
        #     overlap_thresh = 0.6
        #     # occluding_pairs = [
        #     #     (i, j) for i in range(len(self.active_tracks)) for j in range(i + 1, len(self.active_tracks))
        #     #     if iou_matrix[i, j] > overlap_thresh
        #     # ]
        #     occluding_pairs = [
        #         (index_to_id[i], index_to_id[j])
        #         for i in range(len(self.active_tracks))
        #         for j in range(i + 1, len(self.active_tracks))
        #         if iou_matrix[i, j] > overlap_thresh
        #     ]
        #
        # else:
        #     occluding_pairs = []

        """
                Hello - changed this was pass
                I am not sure what to put for margin.
        """
        ##################################################### 07/07
        margin = 1  # 30
        # Track previous depths
        # prev_depths = [trk.prev_depth for trk in self.active_tracks]
        # Match map: track_idx -> det_idx
        index_to_id = {i: trk.id for i, trk in enumerate(self.active_tracks)}
        track_to_det = {
            index_to_id[t_idx]: d_idx
            for d_idx, t_idx in matched
            if t_idx in index_to_id
        }
        # track_to_det = {t_idx: d_idx for d_idx, t_idx in matched}############ no +1 originally
        #print("track to det",track_to_det)

        #print(occluding_pairs)
        #print("[DEBUG] Occluding track ID pairs:")
        for i, j in occluding_pairs:
            #print(f"({i}, {j})")
            self.occlud_actv_trk_ids.update([i, j])  #################### 21/07

        # if len(occluding_pairs)>0:
        #     print("[DEBUG] IOU matrix:")
        #     print(iou_matrix.numpy())  # if you're using torch tensors
        #     time.sleep(10)

        ######################## 22/07
        id_to_track = {trk.id: trk for trk in self.active_tracks}
        track_to_det = {t_id: d_id for d_id, t_id in matched}

        # Assert that all track IDs in matched exist in active_tracks
        # for _, tid in matched:
        #     assert tid in id_to_track, f"Matched track ID {tid} not in active tracks!"
        # print(track_to_det)
        # print("Active track IDs:", [trk.id for trk in self.active_tracks])
        # for trk in self.active_tracks:
        #     print(f"Track ID: {trk.id}, Last Obs: {trk.last_observation}, Prev Depth: {trk.prev_depth}")
        # print("before occlud loop")
        #time.sleep(10)

        for tA, tB in occluding_pairs:
            if tA not in track_to_det or tB not in track_to_det:
                continue  # not matched this frame

            trkA = id_to_track[tA]
            trkB = id_to_track[tB]
            # Previous depths
            prevA = trkA.prev_depth
            prevB = trkB.prev_depth

            State_A = trkA.get_state()[0]
            State_B = trkB.get_state()[0]
            # print("State A", State_A, "State B", State_B )
            # Get detection indices
            dA = track_to_det[tA]
            dB = track_to_det[tB]

            # Now you’re safe to calculate depths etc.
            # print(dets[dA])
            ####################
            # for trk in self.active_tracks:
            #     print(f"Track ID: {trk.id}, Last Obs: {trk.last_observation}, Prev Depth: {trk.prev_depth}")
            #     import matplotlib.pyplot as plt
            #     import matplotlib.patches as patches
            #
            #     # img: your current frame (numpy array, shape [H, W, 3])
            #     # dets[dA]: detection bbox
            #     # trk.last_observation: track's previous bbox
            #
            #     fig, ax = plt.subplots(figsize=(10, 8))
            #     ax.imshow(img)
            #
            #     # Draw detection box (in RED)
            #     x1, y1, x2, y2 = dets[dA][:4]
            #     rect_det = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none',
            #                                  label='Detection')
            #     ax.add_patch(rect_det)
            #
            #     # Draw tracker box (in GREEN)
            #     x1t, y1t, x2t, y2t = trk.last_observation[:4]
            #     rect_trk = patches.Rectangle((x1t, y1t), x2t - x1t, y2t - y1t, linewidth=2, edgecolor='g',
            #                                  facecolor='none', label='Track')
            #     ax.add_patch(rect_trk)
            #
            #     # Optional: add labels
            #     ax.text(x1, y1 - 10, 'Det', color='r')
            #     ax.text(x1t, y1t - 10, 'Track', color='g')
            #
            #     plt.title(f"Det vs Track (dA={dA}, tA={tA})")
            #     plt.legend()
            #     plt.axis('off')
            #     plt.show()

            ####################################
            # time.sleep(60)
            detA_depth = dets[dA][3] #dets[dA][1] + dets[dA][3]
            detB_depth = dets[dB][3] #dets[dB][1] + dets[dB][3]

            # print(f"Checking pair tA={tA}, tB={tB}")
            # print(f"prevA={prevA}, prevB={prevB}, currA={detA_depth}, currB={detB_depth}")

            # Check if the previous depth relationship flipped
            if prevA is not None and prevB is not None:
                # print("hello2")
                # print("prevA", prevA)
                # print("prevB", prevB)
                # print("detA_depth", detA_depth)
                # print("detB_depth", detB_depth)
                #time.sleep(10)
                # if prevA > prevB + margin and detA_depth < detB_depth - margin:
                if (
                        (prevA > prevB + margin and detA_depth < detB_depth - margin)
                        or
                        (prevB > prevA + margin and detB_depth < detA_depth - margin)
                ):

                    # trkA, trkB = self.active_tracks[tA], self.active_tracks[tB]
                    print(f"[FLIP DETECTED] Track {trkA.id} (was front) now behind Track {trkB.id}")

                    # print(f"[FLIP DETECTED] Track {tA} (was front) now behind Track {tB}")
                    # print(f"[FLIP DETECTED] Det {dA} (was front) now behind Det {dB}")
                    self.flip_violating_ids.update([trkA.id, trkB.id])  #################### 21/07
                    # print(f"[FLIP] Violating IDs: {self.flip_violating_ids}")

                    ###########################################
                    # import matplotlib.pyplot as plt
                    # import matplotlib.patches as patches
                    #
                    # fig, ax = plt.subplots(figsize=(12, 9))
                    # ax.imshow(img)
                    #
                    # # Detection A - Red
                    # xa1, ya1, xa2, ya2 = dets[dA][:4]
                    # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
                    #                                linewidth=2, edgecolor='red', facecolor='none'))
                    # ax.text(xa1, ya1 - 5, 'detA', color='red')
                    #
                    # # Detection B - Orange
                    # xb1, yb1, xb2, yb2 = dets[dB][:4]
                    # ax.add_patch(patches.Rectangle((xb1, yb1), xb2 - xb1, yb2 - yb1,
                    #                                linewidth=2, edgecolor='orange', facecolor='none'))
                    # ax.text(xb1, yb1 - 5, 'detB', color='orange')
                    #
                    # # Track A - Green
                    # xta1, yta1, xta2, yta2 = trkA.last_observation[:4]
                    # ax.add_patch(patches.Rectangle((xta1, yta1), xta2 - xta1, yta2 - yta1,
                    #                                linewidth=2, edgecolor='green', facecolor='none'))
                    # ax.text(xta1, yta1 - 5, f'trkA (ID={trkA.id})', color='green')
                    #
                    # # Track B - Blue
                    # xtb1, ytb1, xtb2, ytb2 = trkB.last_observation[:4]
                    # ax.add_patch(patches.Rectangle((xtb1, ytb1), xtb2 - xtb1, ytb2 - ytb1,
                    #                                linewidth=2, edgecolor='blue', facecolor='none'))
                    # ax.text(xtb1, ytb1 - 5, f'trkB (ID={trkB.id})', color='black')
                    #
                    # #Kalman State A
                    # xa1, ya1, xa2, ya2 = State_A[:4]
                    # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
                    #                                linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-'))
                    # ax.text(xa1, ya1 - 5, 'StateA', color='yellow')
                    # # Kalman State B
                    # xa1, ya1, xa2, ya2 = State_B[:4]
                    # ax.add_patch(patches.Rectangle((xa1, ya1), xa2 - xa1, ya2 - ya1,
                    #                                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-'))
                    # ax.text(xa1, ya1 - 5, 'StateB', color='cyan')
                    #
                    #
                    #
                    # ax.set_title(f"Flip Check: dA={dA}, dB={dB}, tA={tA}, tB={tB}")
                    # ax.axis('off')
                    # print(track_to_det)
                    # plt.show()

                    ###########################################

                    # Try to swap the matches if it's a clean 2-way switch
                    # if (dA, tB) in track_to_det and (dB, tA) in track_to_det:
                    # if (dA, tB) in matched and (dB, tA) in matched: # dont want to use matched!
                    #     matched.remove((dA, tB))
                    #     matched.remove((dB, tA))
                    #     matched.append((dA, tA))
                    #     matched.append((dB, tB))
                    # time.sleep(10)
                    if (
                            tA in track_to_det and tB in track_to_det and
                            track_to_det[tA] == dA and
                            track_to_det[tB] == dB
                    ):

                        # Flip the detection assignments
                        track_to_det[tA], track_to_det[tB] = dB, dA

                        # If you still need the `matched` list updated as well:
                        matched = [(d, t) for t, d in track_to_det.items()]
                        # print(f"!!!!!!!!!!!!!!!!!!!!![FIX] Swapped matches to restore front-back order")
                        self.depth_fixes+=1
                        print(self.depth_fixes)
                        # time.sleep(5)

                    # else:
                    #     print(f"[SKIP] Not a clean switch, can't flip safely")

        ###########################


        # for tA, tB in occluding_pairs:
        #     if tA not in track_to_det or tB not in track_to_det:
        #         continue  # one wasn't matched this frame
        #     print("hello1")
        #     # dA, dB = track_to_det[tA], track_to_det[tB]
        #     # detA_depth = dets[dA][1] + dets[dA][3]
        #     # detB_depth = dets[dB][1] + dets[dB][3]
        #     #
        #     # prevA = prev_depths[tA]
        #     # prevB = prev_depths[tB]
        #     trkA = next(trk for trk in self.active_tracks if trk.id == tA)
        #     trkB = next(trk for trk in self.active_tracks if trk.id == tB)
        #
        #     # Previous depths
        #     prevA = trkA.prev_depth
        #     prevB = trkB.prev_depth
        #
        #     # Get index in active_tracks (if needed for track_to_det)
        #     idxA = self.active_tracks.index(trkA)
        #     idxB = self.active_tracks.index(trkB)
        #
        #     dA = track_to_det.get(tA)
        #     dB = track_to_det.get(tB)
        #     detA_depth = dets[dA][1] + dets[dA][3]
        #     detB_depth = dets[dB][1] + dets[dB][3]
        #
        #     # print(f"Checking pair tA={tA}, tB={tB}")
        #     # print(f"prevA={prevA}, prevB={prevB}, currA={detA_depth}, currB={detB_depth}")
        #
        #     # Check if the previous depth relationship flipped
        #     if prevA is not None and prevB is not None:
        #         print("hello2")
        #         print("prevA",prevA)
        #         print("prevB",prevB)
        #         print("detA_depth",detA_depth)
        #         print("detB_depth",detB_depth)
        #         time.sleep(10)
        #         if prevA > prevB + margin and detA_depth < detB_depth - margin:
        #             # trkA, trkB = self.active_tracks[tA], self.active_tracks[tB]
        #             print(f"[FLIP DETECTED] Track {trkA.id} (was front) now behind Track {trkB.id}")
        #
        #             print(f"[FLIP DETECTED] Track {tA} (was front) now behind Track {tB}")
        #             print(f"[FLIP DETECTED] Det {dA} (was front) now behind Det {dB}")
        #             self.flip_violating_ids.update([trkA.id, trkB.id]) #################### 21/07
        #             print(f"[FLIP] Violating IDs: {self.flip_violating_ids}")
        #
        #
        #             # Try to swap the matches if it's a clean 2-way switch
        #             if (dA, tB) in matched and (dB, tA) in matched:
        #                 matched.remove((dA, tB))
        #                 matched.remove((dB, tA))
        #                 matched.append((dA, tA))
        #                 matched.append((dB, tB))
        #                 print(f"!!!!!!!!!!!!!!!!!!!!![FIX] Swapped matches to restore front-back order")
        ##             else:
        ##                 print(f"[SKIP] Not a clean switch, can't flip safely")

        ###################################################
        ############################################################################
        # Here is Reassociation for Depth-Violating Tracks (Recheck)
        ############################################################################
        # Update matched tracks #########22/07
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :])
            self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            # TODO: is better without this
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]
                                       ], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :])
                    self.active_tracks[trk_ind].update_emb(
                        dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                new_kf=not self.new_kf_off,
                max_obs=self.max_obs
            )
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]

            '''
            # self.frame_count <= self.min_hits
            This allows for all detections to be included in the initial frames
            (before the tracker has seen enough frames to confirm tracks).
            '''
            # if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id], [trk.conf], [
                    trk.cls], [trk.det_ind])).reshape(1, -1))

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
                self.removed_tracks.append(trk.id)
        # self.flip_violating_ids = set()

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])