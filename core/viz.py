import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='darkgrid')
# sns.set_palette('flare', as_cmap=True)
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def plot_class_dist(train, test, results_dir, dataset_name=None):

    df = pd.concat([train, test])
    fig, ax = plt.subplots(1, 1)
    chart = sns.countplot(hue='Label', orient='v', dodge=False,
                          x='Label', linewidth=3, palette='flare',
                          data=df, ax=ax)

    box = chart.get_position()
    chart.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontsize=16)
    chart.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1, fontsize=16)
    # ax.get_legend().remove()  # As xticks laels are sufficient
    chart.set(xlabel=None)
    chart.set(ylabel='Total number of samples')
    # chart.set_title(f'{dataset_name}: Class Distribution', fontsize=20)

    # fig.savefig(os.path.join(results_dir, 'plots', 'class_distribution.eps'), format='eps', transparent=True)
    fig.savefig(os.path.join(results_dir, 'plots', dataset_name + '_cd.png'))
    plt.show()
    plt.close(fig)
    logging.info(f'** Class distribution plots saved **')
    print(f'** Class distribution plots saved **')