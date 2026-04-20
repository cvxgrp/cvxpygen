import numpy as np
from numpy import radians as rad
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon


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

    pol = RegularPolygon((endX, endY), 3, radius=0.1, orientation=rad(ori), color='blue')

    return arc, pol


def draw(prob, force_mag, force_angle, torque):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    arr_width = .02
    kw = dict(width=arr_width, head_width=3 * arr_width, head_length=9 * arr_width, color='r')
    
    w_value = np.array([force_mag * np.cos(rad(force_angle)), force_mag * np.sin(rad(force_angle)), torque])
    prob.param_dict['w'].value = w_value
    prob.param_dict['u_prev'].value = prob.var_dict['u'].value
    prob.solve(method='CPG')
    u_value = prob.var_dict['u'].value
    
    ax.clear()
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')  # 'box' shrinks the box, never overrides limits
    
    ax.axis('off')
    
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, facecolor='w', edgecolor='k'))
    ax.arrow(-1, -1, u_value[0], u_value[1], **kw)
    ax.arrow(1, -1, u_value[2], u_value[3], **kw)
    ax.arrow(1,  1, u_value[4], u_value[5], **kw)
    ax.arrow(-1,  1, u_value[6], u_value[7], **kw)
    kw['color'] = 'm'
    ax.arrow(0,  0, w_value[0], w_value[1], **kw)
    
    circ_t, head_t = get_circ(340 * w_value[2])
    ax.add_patch(circ_t)
    ax.add_patch(head_t)
    
    plt.show()
