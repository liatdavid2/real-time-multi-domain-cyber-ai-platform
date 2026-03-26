import numpy as np
from scipy.stats import ks_2samp


def mean_drift(train_mean, current_mean, threshold=0.2):
    """
    Method 1: Mean Drift (Simple Statistical Check)

    Idea:
    Compare the average (mean) value of a feature between training data and current data.

    Why it works:
    If the mean changes significantly, it may indicate that the data distribution shifted.

    Example:
    - Train avg bytes = 1000
    - Current avg bytes = 1400 → possible drift

    When to use:
    - Fast monitoring
    - Numeric features (bytes, packets, duration)
    - Real-time systems (low latency)

    Advantages:
    - Very fast
    - Easy to compute
    - Good first signal

    Limitations:
    - Does NOT detect shape changes (only average)
    - Can miss complex drift patterns

    Output:
    - drift_flag (True/False)
    - diff (absolute difference between means)
    """
    diff = abs(train_mean - current_mean)

    # normalize (optional but safer)
    denom = abs(train_mean) + 1e-8
    relative_diff = diff / denom

    return relative_diff > threshold, relative_diff


def ks_drift(train_values, current_values, p_threshold=0.05):
    """
    Method 2: Kolmogorov-Smirnov Test (KS Test)

    Idea:
    Compare the FULL distribution of values between training data and current data.

    How it works:
    KS test measures the maximum difference between two cumulative distributions.

    Output:
    - p-value:
        low p-value → distributions are different → drift detected

    Interpretation:
    - p >= 0.05 → no drift
    - p < 0.05  → drift detected

    When to use:
    - Detect subtle changes in distribution shape
    - More accurate than mean comparison

    Advantages:
    - Captures full distribution differences
    - Statistically grounded

    Limitations:
    - Slower than mean drift
    - Requires enough data samples

    Output:
    - drift_flag (True/False)
    - p_value (statistical significance)
    """
    if len(train_values) < 10 or len(current_values) < 10:
        return False, 1.0  # not enough data → assume no drift

    stat, p_value = ks_2samp(train_values, current_values)
    return p_value < p_threshold, p_value


def population_stability_index(train_values, current_values, bins=10, threshold=0.2):
    """
    Method 3: Population Stability Index (PSI)

    Idea:
    Compare how the distribution of values shifts across bins.

    How it works:
    1. Split data into bins
    2. Compare percentage of samples in each bin (train vs current)
    3. Calculate divergence using log ratio

    PSI Interpretation (industry standard):
    - PSI < 0.1   → no drift
    - 0.1–0.2     → moderate drift
    - PSI > 0.2   → significant drift (alert!)

    When to use:
    - Production monitoring systems
    - Financial / cybersecurity pipelines
    - Feature-level monitoring

    Advantages:
    - Industry standard
    - Easy to interpret
    - Works well in monitoring dashboards

    Limitations:
    - Sensitive to bin selection
    - Requires tuning bins

    Output:
    - drift_flag (True/False)
    - psi value (drift magnitude)
    """
    if len(train_values) == 0 or len(current_values) == 0:
        return False, 0.0

    train_hist, bin_edges = np.histogram(train_values, bins=bins)
    curr_hist, _ = np.histogram(current_values, bins=bin_edges)

    # avoid division by zero
    if np.sum(train_hist) == 0 or np.sum(curr_hist) == 0:
        return False, 0.0

    train_perc = train_hist / np.sum(train_hist)
    curr_perc = curr_hist / np.sum(curr_hist)

    psi = np.sum(
        (train_perc - curr_perc) *
        np.log((train_perc + 1e-8) / (curr_perc + 1e-8))
    )

    return psi > threshold, psi