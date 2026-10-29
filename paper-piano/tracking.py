"""Paper quad tracking and dot extraction in a stable canonical frame."""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from detection import order_points, approximate_contour, add_inside_border

CANON_W = 640


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, value, t):
        value = np.asarray(value, dtype=np.float32)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = value.copy(), np.zeros_like(value), t
            return self.x.copy()
        dt = t - self.t
        derivative = (value - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * derivative + (1-a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * value + (1-a) * self.x
        self.t = t
        return self.x.copy()


@dataclass
class TrackResult:
    H_cam2canon: np.ndarray | None
    corners_cam: np.ndarray | None
    confidence: float
    status: str


def find_quads(gray, blur=10, canny1=40, canny2=40, min_area_frac=0.03):
    k = max(1, int(blur)*2 + 1)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (k, k), 0), canny1, canny2, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((4, 4), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        if cv2.contourArea(c) < min_area_frac * gray.shape[0] * gray.shape[1]:
            continue
        q = approximate_contour(c, .01)
        if len(q) == 4 and cv2.isContourConvex(q):
            quads.append(order_points(q.reshape(4, 2)))
    return sorted(quads, key=cv2.contourArea, reverse=True)


def extract_dots(warp_gray, return_mask=False):
    extraction = cv2.GaussianBlur(warp_gray, (11, 11), 2)
    extraction = cv2.adaptiveThreshold(extraction, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    extraction = cv2.morphologyEx(extraction, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    extraction = add_inside_border(extraction, 10)
    extraction = cv2.dilate(extraction, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(extraction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 6 and h > 6:
            items.append(((x+w/2, y+h/2), (x, y, w, h)))
    items.sort(key=lambda item: item[0][0])
    dots=np.asarray([p for p, _ in items], np.float32) if items else np.empty((0, 2), np.float32)
    boxes=[b for _, b in items]
    return (dots,boxes,extraction) if return_mask else (dots,boxes)


def debug_edges(gray, blur=10, canny1=40, canny2=40):
    k=max(1,int(blur)*2+1)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(k,k),0),canny1,canny2,L2gradient=True)
    return cv2.dilate(edges,np.ones((4,4),np.uint8))


class PaperTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.filters = [OneEuroFilter() for _ in range(4)]
        self.corners = None
        self.H = None
        self.miss = 0
        self.reacquire = 0
        self.warmup = 0
        self.latest = None

    @staticmethod
    def transform(corners, width, height):
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
        return cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)

    def update(self, gray, t):
        self.warmup += 1
        quads = find_quads(gray, self.cfg.blur, self.cfg.canny1, self.cfg.canny2)
        self.latest = (gray.copy(), t, quads)
        if self.corners is None:
            if quads:
                self.corners = quads[0]
                self.H = np.eye(3, dtype=np.float64)
                return self._result(self.corners, 1.0, "searching")
            return TrackResult(None, None, 0.0, "searching")
        diag = np.linalg.norm(self.corners[0] - self.corners[2])
        acceptable = []
        for q in quads:
            dist = float(np.mean(np.linalg.norm(q-self.corners, axis=1)))
            old_area = abs(cv2.contourArea(self.corners.astype(np.float32)))
            area = abs(cv2.contourArea(q.astype(np.float32)))
            ratio = area / max(old_area, 1)
            gate = (0.5 <= ratio <= 2.0) if self.miss >= self.cfg.hold_frames else (0.6 <= ratio <= 1.6)
            if dist < max(12.0, .10*diag) and gate:
                acceptable.append((dist, q))
        if acceptable:
            _, q = min(acceptable, key=lambda x: x[0])
            if self.miss >= self.cfg.hold_frames:
                self.reacquire += 1
                if self.reacquire < 3:
                    self.miss += 1
                    return self._result(self.corners, .5, "lost")
            self.miss = self.reacquire = 0
            self.corners = np.asarray([f(p, t) for f, p in zip(self.filters, q)], np.float32)
            # Use the current measurement for the homography; keep filtered
            # corners for association and for short detector dropouts.
            return self._result(q, 1.0, "tracking")
        self.miss += 1
        if self.miss <= self.cfg.hold_frames:
            return self._result(self.corners, .5, "tracking")
        self.reacquire = 0
        return self._result(self.corners, 0.0, "lost")

    def _result(self, corners, confidence, status):
        width = CANON_W
        aspect = (np.linalg.norm(corners[0]-corners[3]) + np.linalg.norm(corners[1]-corners[2])) / max(
            np.linalg.norm(corners[0]-corners[1]) + np.linalg.norm(corners[3]-corners[2]), 1)
        height = max(80, int(round(width*aspect)))
        self.H = self.transform(corners, width, height)
        return TrackResult(self.H.copy(), np.asarray(corners).copy(), confidence, status)
