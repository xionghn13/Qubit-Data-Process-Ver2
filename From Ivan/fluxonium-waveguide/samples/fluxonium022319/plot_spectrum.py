import os
import sys
import numpy as np
from matplotlib import pyplot as plt

import Labber

if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    SCRIPT_PATH = os.path.dirname(os.getcwd())
else:
    SCRIPT_PATH = os.path.dirname(__file__)
LOCAL_PATH = os.path.join(SCRIPT_PATH.rsplit('fluxonium-waveguide', 1)[0],
                          'fluxonium-waveguide')
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

from plotting import correct_grid
from local_settings import samples_path

sample = 'fluxonium022319'


def plot_spectrum():
    I0 = 0.9  # mA
    hI = 1.52  # mA

    path_filenames = [
        # 'one tone_57.hdf5',
        # 'one tone_58.hdf5',
        'two tone_28.hdf5',
        'two tone_29.hdf5',
        'two tone_30.hdf5',
        'two tone_31.hdf5',
        'two tone_32.hdf5',
        'two tone_33.hdf5',
        'two tone_34.hdf5',
        'two tone_35.hdf5',
        'two tone_36.hdf5',
        'two tone_37.hdf5',
        'two tone_40.hdf5',
        'two tone_64.hdf5',
        'two tone_66.hdf5',
        'two tone_69.hdf5',
        'two tone_70.hdf5',
        'two tone_71.hdf5',
        'two tone_91.hdf5',
        'two tone_92.hdf5',
    ]

    path_filenames = [os.path.join(samples_path, sample, pf)
                      for pf in path_filenames]
    for path_filename in path_filenames:
        f = Labber.LogFile(path_filename)

        data_2d = data = f.getData(f.getLogChannels()[0]['name'])
        freq_2d = f.getData('Pump - Frequency') / 1.e9
        bias_2d = f.getData('Yoko - Current') / 1.e-3

        n_entries = f.getNumberOfEntries()

        bias_size = bias_2d.shape[0]
        freq_size = bias_2d.shape[1]

        if n_entries != bias_size:
            bias_size = n_entries
            for k in range(bias_size):
                d = f.getEntry(k)

                freq_1d = d['Pump - Frequency'] / 1.e9
                bias_1d = d['Yoko - Current'] / 1.e-3

                freq_size = freq_1d.size

                if k == 0:
                    data_2d = np.empty((bias_size, freq_size),
                                       dtype=complex)
                    freq_2d = np.empty_like(data_2d, dtype=float)
                    bias_2d = np.empty_like(data_2d, dtype=float)
                    plot_2d = np.empty_like(data_2d, dtype=float)

                data_2d[k] = d[data_name]
                freq_2d[k] = freq_1d
                bias_2d[k] = bias_1d
        else:
            bias_size = bias_2d.shape[0]
            freq_size = bias_2d.shape[1]

            freq_2d.shape = (bias_size, freq_size)
            bias_2d.shape = (bias_size, freq_size)
            data_2d.shape = (bias_size, freq_size)
            plot_2d = np.zeros_like(data_2d, dtype=np.float)

        bias_2d -= I0  # mA
        bias_2d /= (2. * hI)  # mA

        plot_2d = np.abs(data_2d - np.mean(data_2d))
        for k in range(data_2d.shape[1]):
            plot_2d[:, k] /= np.max(plot_2d[:, k])
        plot_2d = 1. - np.abs(plot_2d)

        bias_2d, freq_2d = correct_grid(bias_2d, freq_2d)
        plt.pcolormesh(bias_2d, freq_2d, plot_2d, cmap='PuBu')


def label_axes(title='', xlim=[-.7, 1.], ylim=[3, 10],
               title_color='k'):
    labelsize = 18
    plt.xlabel('$\Phi_\mathrm{ext}/\Phi_0$', fontsize=labelsize)
    plt.ylabel('Frequency (GHz)', fontsize=labelsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title, fontsize=labelsize - 3, color=title_color)
    plt.gcf().set_size_inches(7.5, 10., forward=True)
    plt.gca().tick_params(which='major', direction='in', right=True,
                          top=True, labelsize=labelsize, length=6, width=1)
    plt.gca().tick_params(which='minor', direction='in', right=True,
                          top=True, length=3, width=1)
    for axis in ['left', 'right', 'top', 'bottom']:
        plt.gca().spines[axis].set_linewidth(1)
    plt.tight_layout()


if __name__ == '__main__':
    filename = os.path.join(samples_path, sample, 'Plots', 'spectrum.png')
    plot_spectrum()
    label_axes(ylim=[0, 10])
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()
