"""
DVSGesture ROI challenge - dataset loader (fwdPass only).

Standalone module (numpy + tonic, no deep-learning framework): it reproduces the
fwdPass preprocessing used by the Symbolic DNN-Tuner for the ROI version of
DVSGesture, so results are comparable with the networks found by the tuner.

For every recording the dataset gives two pieces of information:

  roi      what happens inside a 32x32 region of interest (ROI) that follows the
           gesture: event frames [T, 32, 32, 2] (2 = event polarities)
  pos      where the ROI is on the 128x128 sensor, in two forms:
             - pos_map:    the ROI position framed like the roi frames and
                           downsampled to 32x32: [T, 32, 32, 1]
             - pos_events: the raw position events of the recording
                           (structured array with fields x, y, s, p, t; x, y =
                           ROI centre in 128x128 sensor coordinates, t in microseconds)

Labels y are integers in [0, 10] (11 classes), one per recording; user is the
id of the person who performed the gesture (train users 1..23, test users 24..29).

fwdPass: every recording is a sequence of T frames, the network classifies ONE
frame at a time and the recording prediction is the average of the per-frame
probabilities (see evaluate.py). How to combine roi and pos is up to you.

Framing (the same for train and test, only the first 1.5 s of every recording):
  - "delta_t" (default, 200 ms): fixed time window of `delta_t` us,
                T = 1_500_000 // delta_t frames for every recording
  - "events":  every frame holds `n_events` ROI events (the last one the remaining
                events); pos_map is accumulated over
                the same time intervals, T changes from recording to recording
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tonic.transforms as transforms
from tonic.dataset import Dataset
from tonic.datasets import DVSGesture


N_CLASSES = 11
CLASS_NAMES = [
    "hand_clapping", "right_hand_wave", "left_hand_wave", "right_arm_clockwise",
    "right_arm_counter_clockwise", "left_arm_clockwise", "left_arm_counter_clockwise",
    "arm_roll", "air_drums", "air_guitar", "other_gestures",
]

# delta_t framing: every recording is cut to its first MAX_DURATION_US microseconds
# (every DVSGesture recording lasts at least ~1.75 s).
MAX_DURATION_US = 1_500_000

DEFAULT_DELTA_T = 200_000
DEFAULT_N_EVENTS = 2000
DEFAULT_ROOT = "./rois_and_coordinates/datasets"
DEFAULT_CACHE_DIR = "./cache/challenge"
ROI_SIZE = 32
SENSOR_SIZE = 128

# Train users held out as validation by train_val_split (the same as the Symbolic
# DNN-Tuner): a user is never in both train and validation, as in the test set.
VAL_USERS = (7, 12, 14, 18)


def center_roi_positions(position: np.ndarray) -> np.ndarray:
    """The position files store the top-left corner of the ROI: shift to the ROI
    centre (+16 per axis), clipped to the sensor pixels 0..127."""
    position = position.copy()
    for axis in ("x", "y"):
        position[axis] = np.clip(position[axis] + ROI_SIZE // 2, 0, SENSOR_SIZE - 1)
    return position


def n_window_frames(delta_t: int) -> int:
    """Number of delta_t frames kept per recording (MAX_DURATION_US // delta_t)."""
    n = MAX_DURATION_US // int(delta_t)
    if n < 1:
        raise ValueError(f"delta_t={delta_t} us is longer than the {MAX_DURATION_US} us window")
    return n


# ------------------------------- Dataset -------------------------------------

class DVSGestureROI(DVSGesture):
    """DVSGesture variant that returns the ROI events and the ROI position events.

    Expects the pre-extracted ROI dataset at <save_to>/DVSGestureROI_fs32/{train,test}/
    with, per recording, <label>.npy (ROI events) and <label>_positions.npy.
    """

    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        position_transform: Optional[Callable] = None,
    ):
        Dataset.__init__(self, save_to, transform=transform)
        self.location_on_system = self.location_on_system + f"_fs{ROI_SIZE}"
        self.position_transform = position_transform
        self.train = train
        self.folder_name = "train" if train else "test"

        self.position = []
        self.users = []
        file_path = os.path.join(self.location_on_system, self.folder_name)
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f"ROI dataset not found: {file_path}")
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in sorted(files):
                if file.endswith(".npy") and not file.endswith("_positions.npy"):
                    self.data.append(os.path.join(path, file))
                    self.targets.append(int(file[:-4]))
                    self.position.append(os.path.join(path, file[:-4] + "_positions.npy"))
                    # folder userXX_<lighting>
                    self.users.append(int(os.path.basename(path).split("_")[0][4:]))

    def __getitem__(self, index):
        events = np.load(self.data[index])
        position = center_roi_positions(np.load(self.position[index]))
        if self.transform is not None:
            events = self.transform(events)
        if self.position_transform is not None:
            position = self.position_transform(position)
        return {"data": events, "pos": position}, self.targets[index]

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return True


def _build_tonic_dataset(train: bool, frame_kw: Optional[dict], root: str) -> DVSGestureROI:
    """frame_kw=None (events framing): roi and position events are returned as events
    (ROI events denoised, position events centred), framed later by _frame_by_events."""
    roi_tfs = [
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=(ROI_SIZE, ROI_SIZE), target_size=(ROI_SIZE, ROI_SIZE)),
    ]
    if frame_kw is None:
        return DVSGestureROI(root, train=train, transform=transforms.Compose(roi_tfs))
    roi_tf = transforms.Compose(roi_tfs + [transforms.ToFrame(sensor_size=(ROI_SIZE, ROI_SIZE, 2), **frame_kw)])
    # position events (128x128 sensor coords) -> [T, 1, 32, 32] position map
    pos_tf = transforms.Compose([
        transforms.Downsample(sensor_size=(SENSOR_SIZE, SENSOR_SIZE, 1), target_size=(ROI_SIZE, ROI_SIZE)),
        transforms.ToFrame(sensor_size=(ROI_SIZE, ROI_SIZE, 1), **frame_kw),
    ])
    return DVSGestureROI(root, train=train, transform=roi_tf, position_transform=pos_tf)


def _frame_by_events(events: np.ndarray, position: np.ndarray, n_events: int):
    """First 1.5 s of a recording -> roi [T, 32, 32, 2] with n_events ROI events per
    frame (the last frame holds the remaining ones, so no event and no recording is
    lost) and pos_map [T, 32, 32, 1] with the position events falling in the same
    time intervals. None if the recording is shorter than 1.5 s."""
    t = events["t"].astype(np.int64)
    if len(t) == 0 or t[-1] - t[0] < MAX_DURATION_US:
        return None
    keep = t - t[0] < MAX_DURATION_US
    events, t = events[keep], t[keep]
    T = -(-len(events) // n_events)   # ceil
    roi = np.zeros((T, ROI_SIZE, ROI_SIZE, 2), dtype=np.float32)
    k = np.arange(len(events)) // n_events
    np.add.at(roi, (k, events["y"].astype(int), events["x"].astype(int), events["p"].astype(int)), 1)

    # frame i covers [edges[i], edges[i + 1]): from its first event to the next frame's
    # first event; the last one runs to the end of the 1.5 s window
    edges = np.append(t[::n_events], t[0] + MAX_DURATION_US)
    pos_map = np.zeros((T, ROI_SIZE, ROI_SIZE, 1), dtype=np.float32)
    pt = position["t"].astype(np.int64)
    pk = np.searchsorted(edges, pt, side="right") - 1
    m = (pk >= 0) & (pk < T)
    scale = SENSOR_SIZE // ROI_SIZE
    np.add.at(pos_map, (pk[m], position["y"][m].astype(int) // scale, position["x"][m].astype(int) // scale, 0), 1)
    return roi, pos_map


def _to_numpy(dataset: DVSGestureROI, n_keep: Optional[int], n_events: Optional[int]):
    """tonic dataset -> (list of roi [T,32,32,2], list of pos_map [T,32,32,1], y [N],
    kept indices [N]). n_keep crops every recording to its first n_keep frames
    (delta_t framing); n_events selects the events framing."""
    rois, maps, ys, kept = [], [], [], []
    n_skipped = n_short = 0
    for i in range(len(dataset)):
        try:
            x, y = dataset[i]
        except Exception as exc:  # unreadable / corrupted recording -> skip
            n_skipped += 1
            print(f"[data] skip sample {i}: {type(exc).__name__}: {exc}")
            continue
        if n_events is not None:
            framed = _frame_by_events(x["data"], x["pos"], n_events)
            if framed is None or framed[0].shape[0] == 0:
                n_short += 1
                continue
            roi, pos = framed
        else:
            roi = np.transpose(np.asarray(x["data"]), (0, 2, 3, 1))   # [T,2,H,W] -> [T,H,W,2]
            pos = np.transpose(np.asarray(x["pos"]), (0, 2, 3, 1))    # [T,1,H,W] -> [T,H,W,1]
            # roi and position map are framed independently: both must cover the window
            if roi.shape[0] < n_keep or pos.shape[0] < n_keep:
                n_short += 1
                continue
            roi, pos = roi[:n_keep], pos[:n_keep]

        rois.append(roi.astype(np.float32))
        maps.append(pos.astype(np.float32))
        ys.append(int(np.asarray(y)))
        kept.append(i)

    if n_skipped:
        print(f"[data] {n_skipped} unreadable recordings skipped")
    if n_short:
        print(f"[data] {n_short} recordings shorter than 1.5 s skipped")
    return rois, maps, np.asarray(ys, dtype=np.int64), np.asarray(kept, dtype=np.int64)


def _split_ragged(flat: np.ndarray, lengths: np.ndarray) -> list:
    return np.split(flat, np.cumsum(lengths)[:-1])


# ------------------------------- Loading -------------------------------------

def load_split(
    split: str,
    framing: str = "delta_t",
    delta_t: int = DEFAULT_DELTA_T,
    n_events: int = DEFAULT_N_EVENTS,
    root: str = DEFAULT_ROOT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """
    Load one split. Use the same framing for train and test.

    Args:
        split:     "train" or "test".
        framing:   "delta_t" (fixed time window) or "events" (fixed number of ROI events).
        delta_t:   frame length in microseconds with "delta_t" framing.
        n_events:  ROI events per frame with "events" framing.
        root:      folder containing DVSGestureROI_fs32/.
        cache_dir: the framed arrays are saved here as .npz and reused.

    Returns:
        dict with
          "roi":        ROI event frames, [N, T, 32, 32, 2] float32 ("delta_t") or a
                        list of N arrays [T_i, 32, 32, 2] ("events", T_i varies)
          "pos_map":    ROI position map aligned with "roi": [N, T, 32, 32, 1] or a
                        list of N arrays [T_i, 32, 32, 1]
          "pos_events": list of N structured arrays (x, y, s, p, t), raw position events
          "y":          [N] int64 labels
          "user":       [N] int64 user id of every recording
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if framing == "delta_t":
        frame_kw, n_keep, n_ev, tag = {"time_window": int(delta_t)}, n_window_frames(delta_t), None, f"dt{delta_t}"
    elif framing == "events":
        if int(n_events) < 1:
            raise ValueError("n_events must be >= 1")
        frame_kw, n_keep, n_ev, tag = None, None, int(n_events), f"ev{n_events}"
    else:
        raise ValueError("framing must be 'delta_t' or 'events'")

    ds = _build_tonic_dataset(split == "train", frame_kw, root)
    cache_file = Path(cache_dir) / f"roigesture_{split}_{tag}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        y, kept = d["y"], d["idx"]
        if n_ev is None:
            roi, pos_map = d["roi"], d["pos_map"]
        else:
            roi, pos_map = _split_ragged(d["roi"], d["lengths"]), _split_ragged(d["pos_map"], d["lengths"])
    else:
        print(f"[data] building roigesture/{split} ({tag}) - first time only, it takes a while")
        roi, pos_map, y, kept = _to_numpy(ds, n_keep, n_ev)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if n_ev is None:
            roi, pos_map = np.stack(roi), np.stack(pos_map)
            np.savez(cache_file, roi=roi, pos_map=pos_map, y=y, idx=kept)
        else:
            lengths = np.array([r.shape[0] for r in roi], dtype=np.int64)
            np.savez(cache_file, roi=np.concatenate(roi), pos_map=np.concatenate(pos_map),
                     lengths=lengths, y=y, idx=kept)
        print(f"[data] saved {cache_file}  {len(y)} recordings")

    pos_events = [center_roi_positions(np.load(ds.position[i])) for i in kept]
    user = np.asarray(ds.users, dtype=np.int64)[kept]
    return {"roi": roi, "pos_map": pos_map, "pos_events": pos_events, "y": y, "user": user}


def load_dataset(**kwargs) -> Tuple[dict, dict]:
    """Load train and test split with the same framing (keyword arguments as in
    load_split): returns (train, test)."""
    return load_split("train", **kwargs), load_split("test", **kwargs)


# ------------------------------- Helpers -------------------------------------

def train_val_split(split: dict, val_users=VAL_USERS) -> Tuple[dict, dict]:
    """Split the train split dict by user: the recordings of val_users go to
    validation, so that (as for the test set) the validation users are never seen
    in training. Default: the same users as the Symbolic DNN-Tuner."""
    val_mask = np.isin(split["user"], np.asarray(val_users))
    if not val_mask.any() or val_mask.all():
        raise ValueError(f"val_users {list(val_users)} leave train or validation empty")

    def sel(a, mask):
        return a[mask] if isinstance(a, np.ndarray) else [e for e, m in zip(a, mask) if m]

    def take(mask):
        return {key: sel(split[key], mask) for key in ("roi", "pos_map", "pos_events", "y", "user")}
    return take(~val_mask), take(val_mask)
