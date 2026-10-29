"""Canonical-space dot templates with translation search and safe adaptation."""
import math
import warnings
import cv2
import numpy as np


class OcclusionScorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dots = np.empty((0, 2), np.float32)
        self.boxes = []
        self.templates = []
        self.unreliable = []
        self.ready = False

    @staticmethod
    def gray(image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(image, (3, 3), 1.0)

    def initialize(self, image, dots, boxes):
        gray = self.gray(image)
        self.dots = np.asarray(dots, np.float32).reshape(-1, 2)
        self.boxes = list(boxes)
        self.templates, self.unreliable = [], []
        for point, box in zip(self.dots, self.boxes):
            _, _, w, h = box
            rt = int(math.ceil(.5*max(w, h))) + self.cfg.template_margin
            patch = self._crop(gray, point, rt)
            self.templates.append(patch.astype(np.float32))
            bad = float(patch.std()) < 3.0
            self.unreliable.append(bad)
            if bad:
                warnings.warn("Low-variance dot template; using normalized absolute difference", RuntimeWarning)
        self.ready = bool(self.templates)

    @staticmethod
    def _crop(gray, point, radius):
        x, y = np.rint(point).astype(int)
        padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        return padded[y:y+2*radius+1, x:x+2*radius+1]

    def score(self, image, allow_update=False, update_mask=None):
        if not self.ready:
            return np.empty(0, np.float32), []
        gray = self.gray(image)
        scores, offsets = [], []
        for i, (point, ref) in enumerate(zip(self.dots, self.templates)):
            rt = ref.shape[0]//2
            rs = rt + (self.cfg.search_margin if self.cfg.use_search_window else 0)
            search = self._crop(gray, point, rs)
            target = ref.astype(np.uint8)
            if search.shape[0] < target.shape[0] or search.shape[1] < target.shape[1]:
                scores.append(-1.0); offsets.append((0, 0)); continue
            if self.unreliable[i]:
                result = cv2.matchTemplate(search.astype(np.float32), ref, cv2.TM_SQDIFF_NORMED)
                _, minimum, _, loc = cv2.minMaxLoc(result)
                score = float(1.0 - min(max(minimum, 0.0), 1.0))
            else:
                result = cv2.matchTemplate(search, target, cv2.TM_CCOEFF_NORMED)
                _, maximum, _, loc = cv2.minMaxLoc(np.nan_to_num(result, nan=-1.0))
                score = float(maximum)
            offset = (loc[0]-rs+rt, loc[1]-rs+rt)
            scores.append(score); offsets.append(offset)
            may_update = allow_update and (update_mask is None or (i < len(update_mask) and update_mask[i]))
            if may_update and self.cfg.use_adaptive_template and score > .85 and max(map(abs, offset)) <= 1:
                patch = self._crop(gray, point + np.asarray(offset), rt).astype(np.float32)
                self.templates[i] = (1-self.cfg.alpha)*ref + self.cfg.alpha*patch
        return np.asarray(scores, np.float32), offsets
