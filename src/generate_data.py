import os
import numpy as np

def simple_bowl(length, depth=1.0):
    """ Generate a simple parabolic sinkhole profile."""
    x = np.linspace(-1, 1, length)
    return depth * (1 - x**2)

def skewed_bowl(length, depth=1.0, skew=0.5):
    """Generate a skewed parabolic sinkhole profile."""
    center = skew * (np.random.rand() * 2 - 1)
    x = np.linspace(-1, 1, length) - center
    return depth * (1 - x**2)

def multi_bowl(length, depth=1.0, count=2):
    """ Generate a profile composed of multiple overlapping bowls."""
    x = np.linspace(-1, 1, length)
    y = np.zeros_like(x)
    for _ in range(count):
        amplitude = depth * np.random.uniform(0.3, 1.0)
        center = np.random.uniform(-0.5, 0.5)
        width = np.random.uniform(0.5, 1.5)
        y += amplitude * np.clip(1 - ((x - center) / width)**2, 0, None)
    return y

def sloped_terrain(length, slope_range=0.5):
    """ Generate a linear sloped terrain profile."""
    slope = np.random.uniform(-slope_range, slope_range)
    return slope * np.linspace(0, 1, length)

def random_poly(length, degree=3, coeff_scale=1.0):
    """Generate a random polynomial terrain profile."""
    coeffs = np.random.randn(degree + 1) * coeff_scale
    x = np.linspace(-1, 1, length)
    return np.polyval(coeffs, x)

def make_profile(length=128, sinkhole=True, noise_std=0.02):
    """
    Create a synthetic elevation profile.

    Args:
        length (int): Number of data points.
        sinkhole (bool): If True, generate a sinkhole-like profile.
        noise_std (float): Standard deviation of added Gaussian noise.

    Returns:
        np.ndarray: Normalized profile array of shape (length,).
    """
    if sinkhole:
        choices = ['simple', 'skewed', 'multi']
        profile_type = np.random.choice(choices, p=[0.4, 0.3, 0.3])
        if profile_type == 'simple':
            base = simple_bowl(length, depth=np.random.uniform(0.5, 1.0))
        elif profile_type == 'skewed':
            base = skewed_bowl(
                length,
                depth=np.random.uniform(0.5, 1.0),
                skew=np.random.uniform(0.2, 0.8)
            )
        else:
            base = multi_bowl(
                length,
                depth=np.random.uniform(0.3, 1.0),
                count=np.random.randint(1, 4)
            )
    else:
        choices = ['sloped', 'poly', 'noise']
        profile_type = np.random.choice(choices, p=[0.4, 0.3, 0.3])
        if profile_type == 'sloped':
            base = sloped_terrain(length, slope_range=0.5)
        elif profile_type == 'poly':
            base = random_poly(
                length,
                degree=np.random.randint(2, 5),
                coeff_scale=0.5
            )
        else:
            base = np.random.rand(length)

    # Add noise and normalize to [0, 1]
    noise = np.random.normal(scale=noise_std, size=length)
    profile = base + noise
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    return profile.astype(np.float32)

def main():
    """Generate and save synthetic sinkhole and non-sinkhole profiles."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    n_samples = 2000
    sinkholes = [make_profile(length=128, sinkhole=True) for _ in range(n_samples)]
    non_sink = [make_profile(length=128, sinkhole=False) for _ in range(n_samples)]

    out_path = os.path.join(data_dir, "synthetic.npz")
    np.savez_compressed(out_path, sinkholes=sinkholes, non_sink=non_sink)
    print(f"Saved {n_samples} sinkhole and {n_samples} non-sink profiles to:\n  {out_path}")

if __name__ == "__main__":
    main()
