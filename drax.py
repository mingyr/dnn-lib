import math
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import imsave

class Drax():
    def __init__(self, save_path):
        self._save_path = save_path

    def __call__(self, images):
        num_figs = len(images)
        height = math.ceil(num_figs/2)

        # draw figures in two column
        self._fig = Figure(figsize=(12.8 if num_figs>1 else 6.4, 4.8*height))
        canvas = FigureCanvas(self._fig) # Attach canvas

        for i in range(height):
            for j in range(2):
                seq = i * 2 + j + 1
                if seq > num_figs:
                    break

                if num_figs == 1:
                    ax = self._fig.add_subplot(1, 1, 1)
                else:
                    ax = self._fig.add_subplot(height, 2, seq)

                ax.axis('off')
                ax.imshow(images[seq-1]['data'])
                ax.set_title(images[seq-1]['title'], fontsize=24)

        self._fig.tight_layout()
        canvas.draw() # renders the figure onto the canvas
        w, h = self._fig.canvas.get_width_height()
        imsave(self._save_path, 
               np.fromstring(self._fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4))


