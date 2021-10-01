import numpy as np
import matplotlib.pyplot as plt
from numpy import radians as rad
from matplotlib.patches import Arc, RegularPolygon
from matplotlib.animation import FuncAnimation


def get_circ(thet):
    if thet > 0:
        thet1 = 0
        thet2 = thet
        end = thet
        ori = thet
    else:
        thet1 = 360 + thet
        thet2 = 360
        end = 360 + thet
        ori = 180 + thet

    arc = Arc((0.0, 0.0), 1, 1, angle=0, theta1=thet1, theta2=thet2, capstyle='round', linestyle='-', lw=2, color='blue')

    endX = 0.5 * np.cos(rad(end))
    endY = 0.5 * np.sin(rad(end))

    pol = RegularPolygon((endX, endY), 3, 0.1, rad(ori), color='blue')

    return arc, pol


def init():
    arr_bl = plt.arrow([], [], [], [])
    arr_br = plt.arrow([], [], [], [])
    arr_tr = plt.arrow([], [], [], [])
    arr_tl = plt.arrow([], [], [], [])
    circ_t = Arc((0, 0), 0, 0)
    head_t = RegularPolygon((0, 0), 3)
    return arr_bl, arr_br, arr_tr, arr_tl, circ_t, head_t


def create_animation(prob, name):
    
    fig, ax = plt.subplots()
    arr_width = .02
    kw = dict(width=arr_width, head_width=3 * arr_width, head_length=9 * arr_width, color='r')

    def update(frame):
        w_value = np.array([0.8 * np.cos(frame), 0.8 * np.sin(frame), np.sin(1.4 * frame)])
        prob.param_dict['w'].value = w_value
        prob.param_dict['u_prev'].value = prob.var_dict['u'].value
        prob.solve(method='CPG')
        u_value = prob.var_dict['u'].value

        ax.clear()
        ax.axis('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, facecolor='w', edgecolor='k'))

        arr_bl = plt.arrow(-1, -1, u_value[0], u_value[1], **kw)
        arr_br = plt.arrow(1, -1, u_value[2], u_value[3], **kw)
        arr_tr = plt.arrow(1, 1, u_value[4], u_value[5], **kw)
        arr_tl = plt.arrow(-1, 1, u_value[6], u_value[7], **kw)
        arr_w = plt.arrow(0, 0, w_value[0], w_value[1], **kw)

        circ_t, head_t = get_circ(340 * w_value[2])
        ax.add_patch(circ_t)
        ax.add_patch(head_t)

        return arr_bl, arr_br, arr_tr, arr_tl, arr_w, circ_t, head_t

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128), init_func=init, blit=True)

    ani.save('%s.gif' % name)
