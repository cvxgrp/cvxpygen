import numpy as np
import matplotlib.pyplot as plt

n, m = 4, 5


def init(prob):
    prob.param_dict['R'].value = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 1],
                                           [1, 0, 0, 1],
                                           [0, 1, 1, 0]])
    prob.param_dict['f_min'].value = 0.2 * np.ones(n)
    prob.param_dict['f_max'].value = np.ones(n)
    prob.param_dict['c'].value = np.array([0.5, 1, 1, 1, 0.7])
    prob.solve(method='CPG')
    

def draw(prob, w1, w2, w3, w4):
    prob.param_dict['w'].value = np.array([w1, w2, w3, w4])
    prob.solve(method='CPG', updated_params=['w'])
    nodes_pos = np.array([[0, 0], [1, 0], [1, 1], [2, 0]])
    edges_ind = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    col_pipes = "#333333"
    col_flows = ['#3498db', '#e91e8c', '#f1c40f', '#2ecc71']
    width_fac = 0.06
    ret = []
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
    ax.axis('equal')
    ax.axis('off')
    plt.show()
