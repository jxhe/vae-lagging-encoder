import visdom
import numpy as np


class VisPlotter(object):
    """A plotter class based on visdom"""
    def __init__(self, env, contour_layout=None):
        super(VisPlotter, self).__init__()
        self.vis = visdom.Visdom()
        self.env = env

        self.contour_confg = dict(colorscale='Jet', 
                                  type=u'contour',
                                  contours=dict(coloring='lines'))

        if contour_layout:
            self.contour_layout = dict(xaxis={'title': 'z1'},
                                       yaxis={'title': 'z2'},
                                       **contour_layout)
        else:
            self.contour_layout = dict(xaxis={'title': 'z1'},
                                       yaxis={'title': 'z2'},
                                       dx=0.1, 
                                       dy=0.1, 
                                       x0=-5, 
                                       y0=-5)

    def plot_contour(self, data, win, name):
        """
        Args:
            data: list of tensors
        """

        traces = []

        for dt in data:
            dt = dt.tolist()
            dt_dict = dict(z=dt, **self.contour_confg)
            traces.append(dt_dict)

        layout = dict(title=name, **self.contour_layout)

        self.vis._send({'data': traces, 'layout': layout, 
                        'win': win, 'env': self.env})

    def save(self, name):
        self.vis.save(name)


if __name__ == '__main__':
    plotter = VisPlotter()
    # plotter.plot_text('hello', 'test')
    # plotter.plot_text('fefe', 'test')
    plotter.plot_contour()

    # plotter.save(['main'])


        