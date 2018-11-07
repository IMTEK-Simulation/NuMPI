
from matplotlib.patches import FancyArrowPatch, Arc,Rectangle
import numpy as np

rec = 0
def asscalar(a):
#    global rec
#    print(rec)
#    rec=rec+1;
#    print(a)
    if isinstance(a,np.ndarray):
        return asscalar(a.tolist()[0])
    elif isinstance(a,(list,tuple)):
        return asscalar(a[0])
    else :
        return a

def draw_npArrow2D(ax, start, end=None, delta=None, arrowprops={}, label=None, textprops={}):
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    for key in arrowprops:
        arrow_prop_dict[key] = arrowprops[key]

    xs = asscalar(start[0])
    ys = asscalar(start[1])

    if end is not None:
        xe = asscalar(end[0])
        ye = asscalar(end[1])
    elif delta is not None:
        xe = xs + asscalar(delta[0])
        ye = ys + asscalar(delta[1])
    else:
        raise AssertionError('Missing Argument end or delta')

    ax.add_artist(FancyArrowPatch((xs, ys), (xe, ye), **arrow_prop_dict))
    if label is not None:
        ax.text(xe, ye, label, **textprops)