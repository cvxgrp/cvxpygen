
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 0.1]})
fig.set_size_inches(4, 5)


def init():

    map_X = ax1.imshow(np.zeros((1, 1)))
    map_W = ax2.imshow(np.zeros((1, 1)))
    map_r = ax3.imshow(np.zeros((1, 1)))

    return map_X, map_W, map_r


def create_animation(prob, name):

    kwargs = dict(cmap='jet', vmin=0, vmax=1)

    def update(frame):

        for i in range(prob.param_dict['r'].shape[0]):
            prob.param_dict['r'].value[i] = np.sin((1 - 0.05 * i) * frame)

        prob.solve(method='CPG')

        map_X = ax1.imshow(prob.var_dict['X'].value, **kwargs)
        map_W = ax2.imshow(prob.param_dict['W'].value, **kwargs)
        map_r = ax3.imshow(prob.param_dict['r'].value.reshape(1, 10), **kwargs)

        ax1.set_title('X')
        ax2.set_title('W')
        ax3.set_title('r')

        if frame == 0:
            plt.colorbar(map_W, ax=ax2)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis('off')

        plt.tight_layout()

        return map_X, map_W, map_r

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 24), init_func=init, blit=True)
    ani.save('%s.gif' % name)
