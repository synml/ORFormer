"""Landmark-aware data augmentation used for training.

Ported from torchlm 0.1.6.10 (MIT License, https://github.com/DefTruth/torchlm), whose
box helpers are based on https://github.com/Paperspace/DataAugmentationForObjectDetection.
Only the transforms ORFormer uses are kept, with the same random calls and results, so
training no longer depends on the unmaintained torchlm package.

Each transform takes and returns (img, landmarks): img is an (H, W, C) uint8 array and
landmarks is an (N, 2) array of (x, y) pixel coordinates.
"""

import random
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

_VISUAL_SIZE = 3  # landmarks are augmented as tiny boxes, like torchlm


class LandmarksOutOfImageError(ValueError):
    """A transform moved some landmarks out of the image; LandmarksCompose skips it."""


class LandmarksCompose:
    """Apply transforms in order, skipping one that pushes landmarks out of the image.

    torchlm skipped a transform on any exception, which hid real bugs (its RandomMask
    always failed on numpy 2); only that expected failure is skipped here.
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            try:
                img, landmarks = t(img, landmarks)
            except LandmarksOutOfImageError:
                continue
        return img, landmarks


class bind:
    """Apply an image-only torchvision transform (e.g. RandomGrayscale) to an HWC array."""

    def __init__(self, transform: torch.nn.Module, prob: float = 1.0) -> None:
        self.transform = transform
        self.prob = prob

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0.0, 1.0) > self.prob:
            return img, landmarks
        chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return self.transform(chw).permute(1, 2, 0).contiguous().numpy(), landmarks


class LandmarksRandomMask:
    """Fill a random rectangle with a random or black color."""

    def __init__(
        self, mask_ratio: float = 0.1, prob: float = 0.5, trans_ratio: float = 0.5
    ) -> None:
        assert 0.02 < mask_ratio < 1.0
        assert 0 < trans_ratio < 1.0
        self.mask_ratio = mask_ratio
        self.prob = prob
        self.trans_ratio = trans_ratio

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0.0, 1.0) > self.prob:
            return img.astype(np.uint8), landmarks.astype(np.float32)

        h, w, c = img.shape
        mask_ratio = np.sqrt(np.random.uniform(0.05, self.mask_ratio, size=1))
        mask_h, mask_w = int(h * mask_ratio[0]), int(w * mask_ratio[0])
        down_w = max(2, int(mask_w * self.trans_ratio))
        up_w = min(int(mask_w * (1 + self.trans_ratio)), w - 2)
        new_mask_w = np.random.randint(min(down_w, up_w), max(down_w, up_w))
        new_mask_h = int(mask_h * mask_w / new_mask_w)

        mask_w = min(max(0.0, new_mask_w), w)
        mask_h = min(max(0.0, new_mask_h), h)
        x0 = np.random.randint(0, w - mask_w + 1)
        y0 = np.random.randint(0, h - mask_h + 1)
        x1, y1 = min(int(x0 + mask_w), w), min(int(y0 + mask_h), h)
        mask_value = np.random.choice(list(range(0, 240, 10)), size=c)
        if np.random.uniform(0.0, 1.0) < 0.35:
            mask_value = [0] * c
        img[y0:y1, x0:x1, :] = mask_value  # in place, like torchlm
        return img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomRotate:
    """Rotate around the image center by an angle drawn from angles fixed at init."""

    def __init__(self, angle: tuple[int, int] = (-10, 10), prob: float = 0.5) -> None:
        low, high = min(angle), max(angle)
        self.prob = prob
        self.choose_angles = np.random.uniform(low, high, size=int(abs(high - low)))

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0.0, 1.0) > self.prob:
            return img.astype(np.uint8), landmarks.astype(np.float32)

        angle = np.random.choice(self.choose_angles)
        boxes = _to_boxes(landmarks)
        h, w = img.shape[:2]
        M = _expanded_rotation(angle, w // 2, h // 2, h, w)
        new_w, new_h = _expanded_size(M, h, w)
        new_img = cv2.warpAffine(img.copy(), M, (new_w, new_h)).astype(np.uint8)

        # rotate the 4 corners of each box, keep top-left and bottom-right
        x1, y1, x2, y2 = (boxes[:, i : i + 1] for i in range(4))
        corners = np.hstack((x1, y1, x2, y1, x1, y2, x2, y2)).reshape(-1, 2)
        corners = np.hstack((corners, np.ones((len(corners), 1))))
        corners = (M @ corners.T).T.reshape(-1, 8)
        rotated = np.zeros_like(boxes)
        rotated[:, [0, 1]] = corners[:, [0, 1]]
        rotated[:, [2, 3]] = corners[:, [6, 7]]

        sx, sy = new_img.shape[1] / w, new_img.shape[0] / h
        new_img = cv2.resize(new_img, (w, h))
        rotated[:, :4] /= [sx, sy, sx, sy]
        return _finish(new_img, _clip_boxes(rotated, w, h), landmarks)


class LandmarksRandomTranslate:
    """Shift the image by a fraction of its size, filling the rest with black."""

    def __init__(
        self,
        translate: tuple[float, float] = (-0.2, 0.2),
        prob: float = 0.5,
        diff: bool = False,
    ) -> None:
        assert len(translate) == 2 and all(-1.0 < t < 1.0 for t in translate)
        self.translate = (min(translate), max(translate))
        self.prob = prob
        self.diff = diff

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0.0, 1.0) > self.prob:
            return img.astype(np.uint8), landmarks.astype(np.float32)

        h, w = img.shape[:2]
        boxes = _to_boxes(landmarks)
        tx = random.uniform(*self.translate)
        ty = random.uniform(*self.translate)
        if not self.diff:
            ty = tx
        dx, dy = int(tx * w), int(ty * h)

        canvas = np.zeros(img.shape, np.uint8)
        canvas[max(0, dy) : min(h, dy + h), max(0, dx) : min(w, dx + w), :] = img[
            max(-dy, 0) : min(h, -dy + h), max(-dx, 0) : min(w, -dx + w), :
        ]
        boxes[:, :4] += [dx, dy, dx, dy]
        return _finish(canvas, _clip_boxes(boxes, w, h), landmarks)


class LandmarksRandomScale:
    """Resize by independent random x/y factors (1 + scale)."""

    def __init__(
        self,
        scale: tuple[float, float] = (-0.4, 0.4),
        prob: float = 0.5,
        diff: bool = True,
    ) -> None:
        assert len(scale) == 2 and all(s > -1.0 for s in scale)
        self.scale = scale
        self.prob = prob
        self.diff = diff

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0.0, 1.0) > self.prob:
            return img.astype(np.uint8), landmarks.astype(np.float32)

        sx = 1 + random.uniform(*self.scale)
        sy = 1 + random.uniform(*self.scale) if self.diff else sx
        new_img = cv2.resize(img, None, fx=sx, fy=sy)
        new_landmarks = landmarks.copy()
        new_landmarks[:, 0] *= sx
        new_landmarks[:, 1] *= sy
        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksResize:
    """Resize to size (w, h); keep_aspect letterboxes with black padding."""

    def __init__(self, size: tuple[int, int] | int, keep_aspect: bool = False) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.keep_aspect = keep_aspect

    def __call__(
        self, img: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        out_w, out_h = self.size
        boxes = _to_boxes(landmarks)
        if self.keep_aspect:
            scale = min(out_w / w, out_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            new_img = np.zeros((out_h, out_w, 3), np.uint8)
            top, left = (out_h - new_h) // 2, (out_w - new_w) // 2
            new_img[top : top + new_h, left : left + new_w, :] = cv2.resize(
                img.astype(np.uint8), (new_w, new_h)
            )
            boxes[:, :4] *= scale
            boxes[:, :4] += [(out_w - scale * w) // 2, (out_h - scale * h) // 2] * 2
        else:
            new_img = cv2.resize(img.astype(np.uint8), (out_w, out_h))
            boxes[:, (0, 2)] *= out_w / w
            boxes[:, (1, 3)] *= out_h / h
        return _finish(new_img, boxes, landmarks)


def _to_boxes(landmarks: np.ndarray) -> np.ndarray:
    """(N, 2) landmarks -> (N, 4) boxes (x, y, x + 3, y + 3)."""
    boxes = np.zeros((len(landmarks), 4))
    boxes[:, :2] = landmarks
    boxes[:, 2:] = boxes[:, :2] + _VISUAL_SIZE
    return boxes


def _clip_boxes(boxes: np.ndarray, w: int, h: int, alpha: float = 0.0025) -> np.ndarray:
    """Clip boxes to the image and drop those left with less than alpha of their area."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    clipped = np.stack(
        (
            np.maximum(boxes[:, 0], 0),
            np.maximum(boxes[:, 1], 0),
            np.minimum(boxes[:, 2], w),
            np.minimum(boxes[:, 3], h),
        ),
        axis=1,
    )
    left = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    return clipped[(area - left) / area < 1 - alpha]


def _finish(
    img: np.ndarray, boxes: np.ndarray, landmarks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes -> landmarks clamped to the image; raise if any landmark was dropped."""
    if len(boxes) != len(landmarks):
        raise LandmarksOutOfImageError(
            f"{len(landmarks) - len(boxes)} landmarks left the image"
        )
    h, w = img.shape[:2]
    new_landmarks = np.stack(
        (np.clip(boxes[:, 0], 0, w), np.clip(boxes[:, 1], 0, h)), axis=1
    )
    return img.astype(np.uint8), new_landmarks.astype(np.float32)


def _expanded_rotation(angle: float, cx: int, cy: int, h: int, w: int) -> np.ndarray:
    """Rotation about (cx, cy), shifted so the rotated image fits a larger canvas."""
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    new_w, new_h = _expanded_size(M, h, w)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return M


def _expanded_size(M: np.ndarray, h: int, w: int) -> tuple[int, int]:
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    return int(h * sin + w * cos), int(h * cos + w * sin)
