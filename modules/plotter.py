import visdom
import numpy as np

class VisPlotter(object):
    """A plotter class based on visdom"""
    def __init__(self, server='http://localhost', env='main', contour_layout=None):
        super(VisPlotter, self).__init__()
        self.vis = visdom.Visdom(server=server, env=env)

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

    def plot_scatter(self, data, labels, legend, zmin, zmax, dz, win, name):
        """
        """
        self.vis.scatter(X=np.array(data),
                         Y=np.array(labels).astype(int),
                         win=win,
                         opts=dict(
                            title=name,
                            legend=legend,
                            xtickmin=zmin,
                            xtickmax=zmax,
                            xtickstep=0.5,
                            ytickmin=zmin,
                            ytickmax=zmax,
                            ytickstep=0.5,
                            markersize=3))

    def plot_line(self, batch_x, batch_y, zmin, zmax, dz):
        """
        Args:
            batch_x: [batch, time_s]
            batch_y: [batch, time_s]
        """
        for id_, (x, y) in enumerate(zip(batch_x, batch_y)):
            win_name = "sample %d" % id_
            self.vis.line(X=np.array(x),
                          Y=np.array(y),
                          win=win_name,
                          opts=dict(
                            title=win_name,
                            markers=True,
                            xtickmin=zmin,
                            xtickmax=zmax,
                            xtickstep=0.5,
                            ytickmin=zmin,
                            ytickmax=zmax,
                            ytickstep=0.5,
                            markersize=3))


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



