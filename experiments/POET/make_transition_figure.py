import os
import sys
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import patches

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))

LIB_DIR = os.path.join(ROOT_DIR, 'libs')
sys.path.append(LIB_DIR)
from experiment_utils import load_experiment


def main(expt_name):

    expt_path = os.path.join(CURR_DIR, 'out', 'evogym_poet', expt_name)
    expt_args = load_experiment(expt_path)

    niches_file = os.path.join(expt_path, 'niches.csv')
    niches = pd.read_csv(niches_file, dtype='Int64')
    niches = niches.fillna(-1)

    colors = plt.get_cmap("tab10")

    niches_dict = {}
    max_iter = 0
    for idx, row in niches.iterrows():
        iteration = int(row['iteration'])
        max_iter = max(max_iter, iteration)
        key = int(row['key'])
        parent = int(row['parent'])
        niches_dict[key] = {'iteration': iteration, 'parent': parent, 'color': colors(idx%10)}

    order = []
    stack = [0]
    while len(stack)>0:
        key = stack.pop(0)
        for key_,data in niches_dict.items():
            if data['parent']==key:
                stack.insert(0, key_)
        order.append(key)

    arrows = []
    style = 'Simple, tail_width=0.25, head_width=3, head_length=3'
    arrow_kwargs = {'arrowstyle': style, 'alpha': 0.4, 'lw': 0.1}

    interval = expt_args['transfer_interval']
    fig,ax = plt.subplots(dpi=500)
    max_iter = 0
    for niche_align,niche_key in enumerate(order):
        iteration = niches_dict[niche_key]['iteration']
        parent_key = niches_dict[niche_key]['parent']

        niche_path = os.path.join(expt_path, 'niche', f'{niche_key}')
        history_file = os.path.join(niche_path, 'history.csv')
        history = pd.read_csv(history_file, dtype={'step': 'Int64', 'score': 'Float64', 'transferred_from': 'Int64'})

        convolved_length = math.ceil(len(history)/interval)
        
        scores = history['score'].fillna(0)
        iterations = history['step'].values[::interval] + iteration
        scores = [scores[i*interval:(i+1)*interval].max() for i in range(convolved_length)]
        colorbar = ax.scatter(iterations, [niche_align]*len(iterations), vmin=0, vmax=expt_args['width']/10, c=scores, cmap='plasma', s=5)

        ax.text(iterations[0]-interval, niche_align, str(niche_key), ha='right', va='center', fontsize=7)

        if parent_key!=-1:
            parent_align = order.index(parent_key)
            ax.plot([iterations[0]]*2, [parent_align, niche_align], color='k', ls=':', lw=0.7)

        transfer_froms = history['transferred_from'].fillna(-1)
        transfer_froms = [transfer_froms[k*interval:(k+1)*interval].max() for k in range(convolved_length)]
        scores_cummax = np.maximum.accumulate(scores)

        iter_i = 0
        score_past_max = float('-inf')
        while iter_i < convolved_length:
            from_key = transfer_froms[iter_i]
            valid = 0
            while iter_i+valid < convolved_length and transfer_froms[iter_i+valid] in [-1, from_key]:
                valid += 1

            if from_key != -1 and scores_cummax[iter_i+valid-1] > score_past_max:
                from_align = order.index(from_key)
                transfer_iter = iterations[iter_i]
                rad = 0.4/(np.log(abs(from_align - niche_align)) + 1)
                if from_align > niche_align:
                    connectionstyle=f'arc3,rad={rad}'
                else:
                    connectionstyle=f'arc3,rad=-{rad}'
                arrows.append(
                    patches.FancyArrowPatch(
                        (transfer_iter-interval, from_align),
                        (transfer_iter, niche_align),
                        connectionstyle=connectionstyle,
                        color=niches_dict[from_key]['color'], **arrow_kwargs))
            
            iter_i += valid
            score_past_max = scores_cummax[iter_i-1]

        max_iter = max(max_iter, iterations.max())

    for a in arrows:
        ax.add_patch(a)
    
    fig.colorbar(colorbar, ax=ax, aspect=80, pad=0.01, label='reward')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([-max_iter/100, max_iter*101/100])

    ax.set_xlabel('steps')
    ax.set_ylabel('niche')
    ax.set_title('transition')

    fig.set_figwidth(max_iter/100)
    fig.set_figheight(len(niches_dict)/5)

    plt.savefig(expt_path + '/transition.jpg', bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    assert len(sys.argv)==2, 'input "{experiment name}"'
    expt_name = sys.argv[1]
    main(expt_name)