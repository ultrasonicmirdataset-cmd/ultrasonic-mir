"""
nmf_engine.py
Purpose: Implements a hybrid NMF engine for source separation, providing both standard sklearn and custom Numba-accelerated backends.
"""
import numpy as np
import numba as nb
from sklearn.decomposition import NMF

@nb.njit(fastmath=True)
def _kl_divergence_update(V, W, H, max_iter, tol, eps=1e-10):
    """
    Custom Numba-accelerated multiplicative update for KL divergence.
    """
    for it in range(max_iter):
        W_old = W.copy()
        H_old = H.copy()
        
        # Update H
        V_hat = np.dot(W, H)
        V_hat = np.maximum(V_hat, eps)
        
        ratio = V / V_hat
        H_num = np.dot(W.T, ratio)
        H_den = np.sum(W, axis=0).reshape(-1, 1) + eps
        
        H = H * (H_num / H_den)
        H = np.maximum(H, eps)
        
        # Update W
        V_hat = np.dot(W, H)
        V_hat = np.maximum(V_hat, eps)
        
        ratio = V / V_hat
        W_num = np.dot(ratio, H.T)
        W_den = np.sum(H, axis=1).reshape(1, -1) + eps  # Correct shape for broadcasting against W (freqs, n_comp)
        
        W = W * (W_num / W_den)
        W = np.maximum(W, eps)
        
        # Normalize W columns
        for c in range(W.shape[1]):
            norm = np.sqrt(np.sum(W[:, c]**2)) + eps
            W[:, c] /= norm
            H[c, :] *= norm
            
        # Convergence check
        err_W = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + eps)
        err_H = np.linalg.norm(H - H_old) / (np.linalg.norm(H_old) + eps)
        if err_W < tol and err_H < tol:
            break
            
    return W, H

def fit_nmf(V, n_components, use_custom=False, beta_loss='kullback-leibler', max_iter=200, tol=1e-4):
    """
    Hybrid NMF engine for source separation.
    
    Parameters:
    - V: Magnitude spectrogram matrix.
    - n_components: Number of sources to separate.
    - use_custom: If True, uses the highly-optimized Numba `@njit` loop. Otherwise, routes to sklearn.
    - beta_loss: Loss function ('frobenius', 'kullback-leibler', 'Itakura–Saito', etc.)
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    """
    if not use_custom:
        # Standard Path: Route to sklearn
        # solver='mu' is required for kullback-leibler
        solver = 'mu' if beta_loss != 'frobenius' else 'cd'
        # 'nndsvda' is best suited for NMF with zeros, especially when not using sparse
        model = NMF(n_components=n_components, init='nndsvda', beta_loss=beta_loss, 
                    solver=solver, max_iter=max_iter, tol=tol, random_state=42)
        W = model.fit_transform(V)
        H = model.components_
        return W, H
    else:
        # Custom Path: Route to optimized Numba backend
        np.random.seed(42)
        # Random Initialization
        W = np.random.rand(V.shape[0], n_components)
        H = np.random.rand(n_components, V.shape[1])
        W = np.maximum(W, 1e-10)
        H = np.maximum(H, 1e-10)
        
        if beta_loss == 'kullback-leibler':
            W, H = _kl_divergence_update(V, W, H, max_iter, tol)
        else:
            raise NotImplementedError(f"Custom loop for beta_loss='{beta_loss}' is not implemented yet. Try 'kullback-leibler'.")
            
        return W, H
