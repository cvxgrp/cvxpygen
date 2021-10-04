
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

n, m = 4, 5

nodes_pos = np.array([[0, 0],
                      [1, 0],
                      [1, 1],
                      [2, 0]])

edges_ind = np.array([[0, 1],
                      [0, 2],
                      [1, 2],
                      [1, 3],
                      [2, 3]])

fig, (ax, ax_w) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})
fig.set_size_inches(10, 5)

col_pipes = '#222222'
col_flows = ['b', 'm', 'c', 'y']
width_fac = 0.06

ax.axis('equal')
ax.axis('off')


def create_animation(prob, name):

    prob.param_dict['R'].value = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 1],
                                           [1, 0, 0, 1],
                                           [0, 1, 1, 0]])
    prob.param_dict['f_min'].value = np.zeros(n)
    prob.param_dict['f_max'].value = np.ones(n)
    prob.param_dict['c'].value = np.array([0.5, 1, 1, 1, 0.7])

    def init():

        ret = []

        for i in range(nodes_pos.shape[0]):
            circle = plt.Circle(tuple(nodes_pos[i, :]), 0.15, color=col_pipes, clip_on=False)
            ax.add_patch(circle)

        for i in range(edges_ind.shape[0]):

            ret.append(ax.plot([], []))
            ret.append(ax.plot([], []))

            for j in np.argwhere(prob.param_dict['R'].value[i, :]).flatten():
                poly = plt.Polygon(np.zeros((1, 2)))
                ax.add_patch(poly)
                ret.append(poly)

        ret.append(ax_w.bar(0, 1))

        return ax, ax_w, ret

    def update(frame):

        ret = []

        prob.param_dict['w'].value = np.array([1 + 3 * (0.5 + 0.5 * np.sin(frame * 1.9)),
                                               1 + 3 * (0.5 + 0.5 * np.sin(frame * 1.4)),
                                               1 + 3 * (0.5 + 0.5 * np.sin(frame)),
                                               1 + 3 * (0.5 + 0.5 * np.sin(frame * 0.8))])
        prob.solve(method='CPG')

        ax.clear()
        ax.axis('equal')
        ax.axis('off')
        ax_w.clear()
        ax_w.axis('off')

        for i in range(nodes_pos.shape[0]):
            circle = plt.Circle(tuple(nodes_pos[i, :]), 0.15, color=col_pipes, clip_on=False)
            ax.add_patch(circle)
            ret.append(circle)

        for i in range(edges_ind.shape[0]):

            fill = 0
            start = nodes_pos[edges_ind[i, 0], :]
            end = nodes_pos[edges_ind[i, 1], :]
            vector = end - start
            offset_raw = np.array([vector[1], -vector[0]])
            offset_norm = offset_raw / np.linalg.norm(offset_raw)
            offset = width_fac * prob.param_dict['c'].value[i] * offset_norm
            x_values = np.array([start[0], end[0]])
            y_values = np.array([start[1], end[1]])

            ret.append(ax.plot(x_values + offset[0], y_values + offset[1], color=col_pipes))
            ret.append(ax.plot(x_values - offset[0], y_values - offset[1], color=col_pipes))

            for j in np.argwhere(prob.param_dict['R'].value[i, :]).flatten():

                fill_ratio = prob.var_dict['f'].value[j] / prob.param_dict['c'].value[i]

                offset1 = (1 - 2 * fill) * offset
                offset2 = offset1 - 2 * fill_ratio * offset

                poly = plt.Polygon(np.array([[x_values[0] + offset1[0], y_values[0] + offset1[1]],
                                             [x_values[1] + offset1[0], y_values[1] + offset1[1]],
                                             [x_values[1] + offset2[0], y_values[1] + offset2[1]],
                                             [x_values[0] + offset2[0], y_values[0] + offset2[1]]]),
                                   color=col_flows[j], zorder=-1)
                ax.add_patch(poly)
                ret.append(poly)

                fill += fill_ratio

        bar = ax_w.bar(np.arange(prob.param_dict['w'].size), prob.param_dict['w'].value)

        for j, b in enumerate(bar):
            b.set_color(col_flows[j])

        ret.append(bar)
        ax_w.set_ylim(0, 4)
        ax_w.set_title('w')
        ax_w.axis('off')

        return ax, ax_w, ret

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128), init_func=init, blit=False)
    ani.save('%s.gif' % name)
