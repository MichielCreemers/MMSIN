import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import gamma

def my_scale(vec):
    vec = (vec - np.mean(vec)) / np.std(vec, ddof=1)
    return vec

def get_color_nss_param(vec):
    return [estimate_basic_param(vec)]

def get_geometry_nss_param(vec):
    return [estimate_basic_param(vec), estimate_ggd_param(vec),
            estimate_aggd_param(my_scale(vec)), estimate_gamma_param(vec)]

def Entropy(labels):
    # Ensure labels is a CuPy array
    labels_np = np.asarray(labels)
    
    # Convert CuPy array to NumPy array before using in Pandas Series
    probs = pd.Series(labels_np).value_counts(bins=2000) / len(labels_np)
    en = entropy(probs)
    return en

def estimate_basic_param(vec):
    result = [np.mean(vec), np.std(vec, ddof=1), Entropy(vec)]
    return result   

def estimate_ggd_param(vec):
    gam = np.arange(0.2, 10 + 0.001, 0.001)
    r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))

    sigma_sq = np.mean(vec ** 2)
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / E ** 2

    differences = np.abs(rho - r_gam)
    array_position = np.argmin(differences)
    gamparam = gam[array_position]
    result = [gamparam, sigma]
    return result

def estimate_aggd_param(vec):
    gam = np.arange(0.2, 10 + 0.001, 0.001)
    r_gam = ((gamma(2.0 / gam)) ** 2) / (
                gamma(1.0 / gam) * gamma(3.0 / gam))

    left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
    right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))
    gamma_hat = left_std / right_std
    rhat = (np.mean(np.abs(vec))) ** 2 / np.mean((vec) ** 2)
    rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (
            (gamma_hat ** 2 + 1) ** 2)

    differences = (r_gam - rhat_norm) ** 2
    array_position = np.argmin(differences)
    alpha = gam[array_position]
    const = np.sqrt(gamma(1 / alpha)) / np.sqrt(gamma(3 / alpha))
    mean_param = (right_std - left_std) * (
                    gamma(2 / alpha) / gamma(1 / alpha)) * const
    result = [alpha, mean_param, left_std, right_std]
    return result

def estimate_gamma_param(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    shape = (mean / std) ** 2
    scale = (std ** 2) / mean
    result = [shape, scale]
    return result