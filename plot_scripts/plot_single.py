import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 20})
plt.tight_layout()

def plot_line(x, y, fname='', scale=None, xmin=-1.5, xmax=1.5, ymin=-0.002, ymax=0.002):
    line_opt=dict(marker='o', markersize=15, mec='red', mfc='red', linewidth=4)
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(plot_x, 
              plot_y, 
              **line_opt)
    if scale:
        xmin = ymin = -scale
        xmax = ymax = scale
    dec_factor = 1.00
    inc_factor = 1.00
    
    axes.set_xlim(xmin * inc_factor, xmax * inc_factor)
    axes.set_ylim(ymin * inc_factor, ymax * inc_factor)
    linear_range = np.arange(xmin, xmax, 0.1)
    diag_opt=dict(linewidth=4, linestyle='--', color='tab:purple')
    axes.plot(linear_range,
              linear_range,
              **diag_opt)

    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    
    axes.spines['right'].set_color('black')
    axes.spines['top'].set_color('black')
    axes.spines['left'].set_color('black')
    axes.spines['bottom'].set_color('black')
    
    axes.spines['right'].set_linewidth(4)
    axes.spines['top'].set_linewidth(4)
    axes.spines['left'].set_linewidth(4)
    axes.spines['bottom'].set_linewidth(4)
    
    axis_opt = dict(linewidth=1.5, linestyle='--', color='black')
    xaxis = np.arange(xmin, xmax + 0.1, 0.1)
    yaxis = np.arange(ymin, ymax + 0.1, 0.1)
    axes.plot(xaxis, np.zeros(len(xaxis)), **axis_opt)
    axes.plot(np.zeros(len(yaxis)), yaxis, **axis_opt)
    axes.tick_params(labelsize=30)
    axes.set_xlabel('mean of true model posterior', fontsize=30)
    axes.set_ylabel('mean of approximate posterior', fontsize=30)
    
    axes.set_xticks([-scale, 0, scale])
    axes.set_yticks([-scale, 0, scale])

    # plot arrow
    # print(len(plot_x) - 1)
    for i in range(len(plot_x) - 1):
        axes.annotate('',
                      xytext=(plot_x[i], plot_y[i]),
                      xy=(plot_x[i+1], plot_y[i+1]), 
                      arrowprops=dict(headlength=15, headwidth=14, width=5, edgecolor='#289323', facecolor='#289323'))
    if fname != '':
        fig.savefig(fname, bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot single point trajectory on posterior mean space')
    parser.add_argument('--aggressive', type=int, default=0,
        help='the aggressive mode of plot data')
    parser.add_argument('--id', type=int, default=0,
        help='the point id to be plotted, we saved trajectory of 50 points by default, \
        so the id should be in [0, 49].')

    args = parser.parse_args()

    save_dir = "figures/single"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = "plot_data/single/aggr%d_single.pickle" % args.aggressive
    save_path = os.path.join(save_dir, "aggr%d_id%d_single.pdf" % (args.aggressive, args.id))
    data = pickle.load(open(data_path, 'rb'))

    plot_x = data["posterior"][args.id][11:20]
    plot_y = data['inference'][args.id][11:20]

    plot_line(plot_x, plot_y, scale=3.0, fname=save_path)
