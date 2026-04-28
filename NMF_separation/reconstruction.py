"""
reconstruction.py
Purpose: Handles the reconstruction of audio signals from NMF components by generating masks and applying them to the original complex STFT.
"""
import numpy as np
import scipy.signal
from dsp import compute_istft

def generate_masks(W, H, mask_type='soft', power=1, eps=1e-10):
    K = W.shape[1]
    V_hat_components = []

    for k in range(K):
        W_k = W[:, k:k+1]
        H_k = H[k:k+1, :]
        V_hat_k = W_k @ H_k
        V_hat_components.append(V_hat_k)

    V_hat_components = np.array(V_hat_components)

    if mask_type == 'soft':
        V_hat_components_pow = V_hat_components ** power
        V_hat_sum = np.sum(V_hat_components_pow, axis=0) + eps
        masks = V_hat_components_pow / V_hat_sum
    elif mask_type == 'hard':
        max_idx = np.argmax(V_hat_components, axis=0)
        masks = np.zeros_like(V_hat_components)
        for k in range(K):
            masks[k] = (max_idx == k).astype(float)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return masks


def apply_masks_and_reconstruct(D, masks, hop_length=512, win_length=None, length=None):
    K = masks.shape[0]
    reconstructed_signals = []

    for k in range(K):
        D_k = D * masks[k]
        y_k = compute_istft(D_k, hop_length=hop_length, win_length=win_length, length=length)
        reconstructed_signals.append(y_k)

    return reconstructed_signals


def smooth_masks(masks, kernel_size=5):
    """
    Applies median filtering along the time axis of each mask to suppress
    'musical noise' — the rapid flickering of isolated mask bins that causes
    a metallic, fluttery artifact in the output audio.

    Operates on the full (K, F, T) mask array produced by generate_masks,
    processing each frequency band independently so transient attacks are
    preserved while noisy isolated spikes are removed.

    Parameters
    ----------
    masks       : (K, F, T) float array in [0, 1] — output of generate_masks
    kernel_size : int (odd) — width of the median filter in time frames
                      3–5  subtle smoothing, good for drums/transients
                      7–11 heavier smoothing, good for pads/sustained tones

    Returns
    -------
    masks_smooth : (K, F, T) float array in [0, 1]
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # medfilt2d requires odd kernel

    K = masks.shape[0]
    masks_smooth = np.empty_like(masks)

    for k in range(K):
        # medfilt2d treats axes as (rows=freq, cols=time) — filter only along time
        masks_smooth[k] = scipy.signal.medfilt2d(
            masks[k].astype(np.float32),
            kernel_size=[1, kernel_size]   # [freq_kernel=1, time_kernel]
        )

    return np.clip(masks_smooth, 0, 1)


def floor_suppress_masks(masks, floor_threshold=0.1, transition_width=0.05):
    """
    Attenuates bins where a source's mask is very low, killing residual bleed
    from other sources that survive Wiener masking at near-zero levels.

    Uses a smooth sigmoid gate (not a hard binary cut) to avoid clicks and
    spectral discontinuities at the suppression boundary.

    The gate for each (k, f, t) bin is:
        gate(m) = sigmoid((m - threshold) / transition_width)

    so bins well below the threshold → 0, bins well above → 1,
    and the transition is a soft S-curve of width ~transition_width.

    Parameters
    ----------
    masks            : (K, F, T) float array — output of smooth_masks or generate_masks
    floor_threshold  : float in [0, 1]
                           0.05  gentle, preserves more of the source signal
                           0.10  recommended default
                           0.15  aggressive, useful for highly overlapping sources
    transition_width : float — steepness of the sigmoid transition
                           smaller → sharper cut (closer to binary)
                           larger  → softer fade, fewer edge artifacts

    Returns
    -------
    masks_gated : (K, F, T) float array in [0, 1]
    """
    gate = 1.0 / (1.0 + np.exp(
        -(masks - floor_threshold) / (transition_width + 1e-8)
    ))
    return masks * gate


def refine_masks(masks, smooth_kernel=5, floor_threshold=0.1, transition_width=0.05):
    """
    Convenience wrapper: applies temporal smoothing then spectral floor suppression
    to the (K, F, T) mask array, ready to be passed to apply_masks_and_reconstruct.

    Call this between generate_masks and apply_masks_and_reconstruct:

        masks   = generate_masks(W, H, mask_type='soft', power=2)
        masks   = refine_masks(masks)
        signals = apply_masks_and_reconstruct(D, masks, hop_length=512)

    Parameters
    ----------
    masks            : (K, F, T) float — output of generate_masks
    smooth_kernel    : int — median filter width in time frames (see smooth_masks)
    floor_threshold  : float — gate threshold (see floor_suppress_masks)
    transition_width : float — gate sigmoid width (see floor_suppress_masks)

    Returns
    -------
    masks_refined : (K, F, T) float in [0, 1]
    """
    masks = smooth_masks(masks, kernel_size=smooth_kernel)
    masks = floor_suppress_masks(masks, floor_threshold=floor_threshold,
                                 transition_width=transition_width)
    return masks
