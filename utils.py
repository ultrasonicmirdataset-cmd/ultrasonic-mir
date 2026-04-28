#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal utility functions required by models.py.

This file intentionally contains only the helper functions imported by the
PANN model definitions:
    - do_mixup
    - interpolate
    - pad_framewise_output
"""

import torch


def do_mixup(x, mixup_lambda):
    """
    Apply mixup augmentation to a batch tensor.

    Args:
        x: Input tensor with shape (batch_size, ...).
        mixup_lambda: Mixup coefficients with shape (batch_size,).

    Returns:
        Tensor with the same shape as x after mixup.
    """
    return (
        x.transpose(0, -1) * mixup_lambda
        + torch.flip(x, dims=[0]).transpose(0, -1) * (1.0 - mixup_lambda)
    ).transpose(0, -1)


def interpolate(x, ratio):
    """
    Interpolate framewise output along the time axis.

    Args:
        x: Tensor with shape (batch_size, time_steps, classes_num).
        ratio: Integer upsampling ratio.

    Returns:
        Tensor with shape (batch_size, time_steps * ratio, classes_num).
    """
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """
    Pad framewise output to match the target number of frames.

    The padding value is copied from the final available frame.

    Args:
        framewise_output: Tensor with shape (batch_size, frames, classes_num).
        frames_num: Target number of frames.

    Returns:
        Tensor with shape (batch_size, frames_num, classes_num).
    """
    current_frames = framewise_output.shape[1]

    if current_frames >= frames_num:
        return framewise_output[:, :frames_num, :]

    pad = framewise_output[:, -1:, :].repeat(
        1,
        frames_num - current_frames,
        1,
    )

    return torch.cat((framewise_output, pad), dim=1)
