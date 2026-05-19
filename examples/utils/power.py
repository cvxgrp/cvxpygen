
import numpy as np
import matplotlib.pyplot as plt


def profiles():
    
    # grid price
    P_day = np.full(48, 0.25)
    P_day[32:46] = 0.5
    P = np.tile(P_day, 7)

    # load
    hours = np.arange(48) * 0.5
    base = 0.5
    morning = 1.0 * np.exp(-0.5 * ((hours -  7.0) / 1.2) ** 2)
    midday_dip = -0.3 * np.exp(-0.5 * ((hours - 13.0) / 2.5) ** 2)
    evening = 2.8 * np.exp(-0.5 * ((hours - 19.0) / 1.6) ** 2)
    L_day = np.clip(base + morning + midday_dip + evening, 0.1, None)
    L = np.tile(L_day, 7)

    # irradiance
    latitude_deg = 37.4  # ~Stanford / Bay Area, CA
    day_of_year  = 172  # ~June 21
    declination = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    hour_angle = np.radians(15.0 * (hours - 12.0))
    phi = np.radians(latitude_deg)
    cos_zenith = (np.sin(phi) * np.sin(declination)
                + np.cos(phi) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)   # sun below horizon
    I0, tau = 1361.0, 0.7   # solar constant, clear-sky transmittance
    with np.errstate(divide='ignore', invalid='ignore'):
        atmos = np.where(cos_zenith > 0, tau ** (1.0 / cos_zenith), 0.0)
    R_day = I0 * cos_zenith * atmos / 1e3  # kW/m^2, shape (48,)
    R = np.tile(R_day, 7)

    return P, L, R


def plot(profile, title, label):
    
    x = np.arange(len(profile)) * 0.5
    
    _, ax = plt.subplots(figsize=(12, 2))
    ax.step(x, profile, where='post', linewidth=1.2)
    ax.fill_between(x, 0, profile, step='post', alpha=0.25)
    
    # Day boundaries + centered labels
    for d in range(1, 7):
        ax.axvline(d * 24, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xticks([d * 24 + 12 for d in range(7)])
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_xlim(0, 24 * 7)
    ax.set_ylim(0, 1.1 * max(profile))
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
