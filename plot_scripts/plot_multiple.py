import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 20})
plt.tight_layout()

def load_data(fname):
    data = pickle.load(open(fname, 'rb'))
    return data['posterior'], data['inference']

def plot_multiple(x, y, scale=None, fname='', xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, 
    dx=0.5, xlabel="mean of true model posterior", ylabel="mean of approximate posterior"):
    line_opt=dict(linewidth=4, linestyle='--', color='tab:purple')
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(1, 1, 1)
    if scale:
        xmin = ymin = -scale
        xmax = ymax = scale
    dec_factor = 1.00
    inc_factor = 1.00
    axes.set_xlim(xmin * inc_factor, xmax * inc_factor)
    axes.set_ylim(ymin * inc_factor, ymax * inc_factor)
    linear_range = np.arange(xmin * dec_factor, xmax * dec_factor, 0.1)
    axis_opt = dict(linewidth=1.5, linestyle='--', color='black')
    xaxis = np.arange(xmin, xmax + 0.1, 0.1)
    yaxis = np.arange(ymin, ymax + 0.1, 0.1)
#     # Move left y-axis and bottim x-axis to centre, passing through (0,0)
#     axes.spines['left'].set_position('center')
#     axes.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    axes.spines['right'].set_color('black')
    axes.spines['top'].set_color('black')
    axes.spines['left'].set_color('black')
    axes.spines['bottom'].set_color('black')
    
    axes.spines['right'].set_linewidth(4)
    axes.spines['top'].set_linewidth(4)
    axes.spines['left'].set_linewidth(4)
    axes.spines['bottom'].set_linewidth(4)
    # Show ticks in the left and lower axes only
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    
#     axes.set_yticks(list(np.arange(ymin, 0, 1)) + list(np.arange(1.0, ymax+1, 1)))
    axes.set_yticks(np.arange(ymin, ymax+dx, dx))
    axes.set_xticks(np.arange(xmin, xmax+dx, dx))
    axes.set_xticklabels(['%.1f' % i for i in np.arange(xmin, xmax+dx, dx)])
    axes.set_yticklabels(['%.1f' % i for i in np.arange(ymin, ymax+dx, dx)])
    axes.plot(linear_range,
              linear_range,
              **line_opt)
    axes.plot(xaxis, np.zeros(len(xaxis)), **axis_opt)
    axes.plot(np.zeros(len(yaxis)), yaxis, **axis_opt)
    axes.tick_params(labelsize=35)
    scatter_opt=dict(s=100, c='r', marker='x')
    axes.scatter(x=x, y=y,**scatter_opt)
    
    if xlabel != '':
        axes.set_xlabel(xlabel, fontsize=20)
        
    if ylabel != '':
        axes.set_ylabel(ylabel, fontsize=20)
    if fname != '':
        fig.savefig(fname, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot multiple points on posterior mean space')
    parser.add_argument('--aggressive', type=int, default=0,
        help='the aggressive mode of plot data')
    parser.add_argument('--iter', type=int, default=0,
        help='the iteration of plot data')

    args = parser.parse_args()

    save_dir = "figures/multiple"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = "plot_data/multiple/aggr%d_iter%d_multiple.pickle" % (args.aggressive, args.iter)
    save_path = os.path.join(save_dir, "aggr%d_iter%d_multiple.pdf" % (args.aggressive, args.iter))
    data_model_p, data_infer_p = load_data(data_path)
    plot_multiple(x=data_model_p, y=data_infer_p, scale=3.0, dx=1.0, fname=save_path)

