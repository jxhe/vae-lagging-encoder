import visdom
import numpy as np


class VisPlotter(object):
    """A plotter class based on visdom"""
    def __init__(self, server='http://localhost', env='main', contour_layout=None):
        super(VisPlotter, self).__init__()
        self.vis = visdom.Visdom(server=server)
        self.env = env

        self.color_list = ['Jet', 'RdBu']

        if contour_layout:
            self.contour_confg = dict(colorscale='Jet',
                                      type=u'contour',
                                      contours=dict(coloring='lines'),
                                      **contour_layout)
        else:
             self.contour_confg = dict(colorscale='Jet',
                                       type=u'contour',
                                       contours=dict(coloring='lines'))

        self.contour_layout = dict(xaxis={'title': 'z1'},
                                   yaxis={'title': 'z2'})

    def plot_contour(self, data, win, name):
        """
        Args:
            data: list of tensors
        """

        traces = []

        for dt, color in zip(data, self.color_list):
            dt = dt.tolist()
            self.contour_confg['colorscale'] = color
            dt_dict = dict(z=dt, title=win, **self.contour_confg)
            traces.append(dt_dict)

        layout = dict(title=name, **self.contour_layout)
        opts = dict(title=win)

        self.vis._send({'data': traces, 'layout': layout,
                        'win': win, 'opts': opts})

    def plot_text(self):
        self.vis.text('Hello, world!')


    def save(self, name):
        self.vis.save(name)


if __name__ == '__main__':
    plotter = VisPlotter()
    # plotter.plot_text('hello', 'test')
    # plotter.plot_text('fefe', 'test')
    plotter.plot_text()

    # plotter.save(['main'])



