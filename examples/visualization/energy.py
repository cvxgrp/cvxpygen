import numpy as np
import matplotlib.pyplot as plt


def draw(q, p):
    
    _, ax1 = plt.subplots(figsize=(9, 4))

    ax1.plot(np.arange(24 * 12 + 1) / 12, 100 * q / max(q), color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Hour of the day")
    ax1.set_ylabel("State of charge [%]", color="steelblue")
    ax1.set_ylim([0, 101])
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(24 * 12) / 12, 100 * p / max(p), color="orchid", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Relative price [%]", color="orchid")
    ax2.set_ylim([0, 101])
    ax2.tick_params(axis="y", labelcolor="orchid")

    plt.xticks(np.arange(25))
    plt.xlim([0, 24])
    plt.tight_layout()
    plt.show()
