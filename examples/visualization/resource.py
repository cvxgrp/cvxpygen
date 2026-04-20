import numpy as np
import matplotlib.pyplot as plt


def draw(W, S, r, X):
    fig, ((ax0, ax1, ax2), (ax_empty_bl, ax3, ax_empty_br)) = \
        plt.subplots(2, 3, gridspec_kw={'height_ratios': [2, 0.1], 'width_ratios': [0.1, 1, 1]})
    fig.set_size_inches(7, 5)
    kwargs = dict(cmap='cividis', vmin=0, vmax=1)

    ax0.imshow(np.diag(S).reshape(-1, 1), **kwargs)
    ax1.imshow(X, **kwargs)
    map_W = ax2.imshow(W, **kwargs)
    ax3.imshow(r.reshape(1, -1), **kwargs)

    ax0.set_title('S')
    ax1.set_title('X')
    ax2.set_title('W')
    ax3.set_title('r')

    plt.colorbar(map_W, ax=ax2)

    for ax in [ax0, ax1, ax2, ax3, ax_empty_bl, ax_empty_br]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
