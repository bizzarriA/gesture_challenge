"""
DVSGesture ROI challenge - official metric (framework independent), parameter and
FLOPs count (Keras or PyTorch models, see count_flops / count_params).

The network classifies one frame at a time. The prediction for a recording is the
class with the highest per-frame probability averaged over its T frames (soft
majority vote), exactly as in the Symbolic DNN-Tuner.

    frame_probs: per-frame class probabilities (softmax) for the N test recordings,
                 in the order returned by data.load_split("test"): an array
                 [N, T, 11], or a list of N arrays [T_i, 11] when T varies
                 ("events" framing)
    y:           [N] integer labels
"""
from __future__ import annotations

import numpy as np


def recording_predictions(frame_probs) -> np.ndarray:
    """Per-frame probabilities ([N, T, C] or list of [T_i, C]) -> [N] predicted class per recording."""
    preds = []
    for probs in frame_probs:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[0] == 0:
            raise ValueError(f"expected [T, C] per-frame probabilities per recording, got {probs.shape}")
        preds.append(probs.mean(axis=0).argmax())
    return np.asarray(preds)


def accuracy(frame_probs, y: np.ndarray) -> float:
    """Recording-level accuracy (soft majority vote over frames)."""
    return float((recording_predictions(frame_probs) == np.asarray(y)).mean())


def confusion_matrix(frame_probs, y: np.ndarray, n_classes: int = 11) -> np.ndarray:
    preds = recording_predictions(frame_probs)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y), preds), 1)
    return cm


# ------------------------------- FLOPs ----------------------------------------
#
# FLOPs of ONE frame (one forward pass with batch 1), with the same rules as the
# TensorFlow profiler used by the Symbolic DNN-Tuner:
#   Conv1D / Conv2D, Dense / Linear   2 FLOPs per multiply-add (MAC) + 1 per output (bias)
#   MaxPooling / AveragePooling       pool_size FLOPs per output element
#   GlobalAveragePooling              1 per input element  (GlobalMaxPooling: 0)
#   Add / Subtract / Multiply         1 per output element per extra input
#   softmax                           5 per output element
#   activations (ReLU, ELU), BatchNorm, Dropout, Flatten, Concatenate, ...: 0

def _numel(shape) -> int:
    """Elements of a shape without the batch dimension."""
    return int(np.prod([int(d) for d in shape[1:]]))


def _keras_flops(model) -> int:
    flops = 0
    for layer in model.layers:
        if hasattr(layer, "layers"):            # nested Model / Sequential
            flops += _keras_flops(layer)
            continue
        name = type(layer).__name__
        out_shape = layer.output.shape
        inputs = layer.input if isinstance(layer.input, (list, tuple)) else [layer.input]
        n_out = _numel(out_shape)
        kernel = getattr(layer, "kernel", None)
        if name.startswith("Conv") or name == "Dense":
            # MACs = (outputs / output channels) * kernel size (k... * in/groups * out)
            macs = n_out // int(out_shape[-1]) * int(np.prod(kernel.shape))
            flops += 2 * macs + (n_out if getattr(layer, "use_bias", False) else 0)
        elif name.startswith(("MaxPooling", "AveragePooling")):
            flops += n_out * int(np.prod(layer.pool_size))
        elif name.startswith("GlobalAveragePooling"):
            flops += _numel(inputs[0].shape)
        elif name in ("Add", "Subtract", "Multiply"):
            flops += n_out * (len(inputs) - 1)
        if name == "Softmax" or getattr(getattr(layer, "activation", None), "__name__", "") == "softmax":
            flops += 5 * n_out
    return flops


def _torch_flops(model, input_shape) -> int:
    import torch
    import torch.nn as nn

    shapes = input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]
    inputs = [torch.zeros(1, *s) for s in shapes]
    total = [0]

    def hook(module, inp, out):
        n_out = out[0].numel() if out.dim() > 0 else 1   # batch is 1
        f = 0
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            per_out = module.in_channels // module.groups * int(np.prod(module.kernel_size))
            f = 2 * n_out * per_out + (n_out if module.bias is not None else 0)
        elif isinstance(module, nn.Linear):
            f = 2 * n_out * module.in_features + (n_out if module.bias is not None else 0)
        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
            k = module.kernel_size
            f = n_out * int(np.prod(k if isinstance(k, (tuple, list)) else [k] * (out.dim() - 2)))
        elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            f = inp[0][0].numel()
        elif isinstance(module, nn.Softmax):
            f = 5 * n_out
        total[0] += f

    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    handles = [m.register_forward_hook(hook) for m in leaves]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total[0]


def count_flops(model, input_shape=None) -> int:
    """FLOPs of one frame (see the rules above), for a Keras or a PyTorch model.

    Keras:   input_shape is not needed (taken from the model inputs).
    PyTorch: input_shape is the per-frame input shape WITHOUT batch, channels first,
             e.g. (3, 32, 32); for a model with several inputs (forward(x, pos)) a
             list of shapes, e.g. [(2, 32, 32), (2,)]. Only nn.Module layers are
             counted: use nn.Softmax / nn.MaxPool2d ... instead of the functional
             versions (an x + y in forward is not counted either).
    """
    if hasattr(model, "layers") and hasattr(model, "count_params"):
        return int(_keras_flops(model))
    if hasattr(model, "named_modules"):
        if input_shape is None:
            raise ValueError("PyTorch model: pass input_shape (per frame, without batch)")
        return int(_torch_flops(model, input_shape))
    raise TypeError("count_flops supports Keras and PyTorch models")


def count_params(model) -> int:
    """Total number of parameters (trainable and not) of a Keras or a PyTorch model."""
    if hasattr(model, "count_params"):
        return int(model.count_params())
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters()) +
                   sum(b.numel() for b in model.buffers() if b.dtype.is_floating_point))
    raise TypeError("count_params supports Keras and PyTorch models")
