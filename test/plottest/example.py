import matplotlib
matplotlib.use('Qt5Agg')   # generate postscript output by default
import pyqtgraph as pg
from matplotlib.pyplot import imread, ion, plot, close
from plot_pg import imagescpg, plotpg
from numpy import arange, mean, linspace
from numpy.random import randn

ion()
plot([1, 2, 3])
close()


def plot_image():
    data = imread(r'images/example_imagesc_color.jpg')

    x = linspace(-54.7321, 5812.512, num=data.shape[1])
    y = arange(data.shape[0]) * 1
    imv_color = imagescpg(
        x, y, data, colormap='jet', title='Color Image', xlabel='Column',
        ylabel='Row', colorbar=True)

    data = mean(imread(r'images/example_imagesc_grey.jpg'), axis=2)

    x = arange(data.shape[1]) * 1
    y = arange(data.shape[0]) * 1
    imv_gray = imagescpg(
        x, y, data, colormap='viridis',
        title='Gray Image with viridis Colormap', xlabel='Column',
        ylabel='Row', colorbar=True)

    data = randn(2048, 2048)
    x = arange(data.shape[1]) * 5
    y = arange(data.shape[0]) * 1

    imv_gray0 = imagescpg(
        x, y, data,
        colormap='gray', title='Gray Image with gray Colormap',
        xlabel='Column', ylabel='Row', colorbar=True)

    imv_gray1 = imagescpg(
        data,
        'parula', 'Gray Image with parula Colormap', 'Column', 'Row', False)

    imv_gray2 = imagescpg(
        x, y, data,
        'jet', 'Gray Image with jet Colormap', 'Column', 'Row', True,
        {'font_family': 'Malgun Gothic', 'title_font_size': '17pt',
         'title_bold': True, 'label_font_size': 15,
         'tick_font_size': 10, 'tick_thickness': 2, 'tickTextOffset': 5})

    print(3)

    return imv_gray2


def plot_1d():
    n_data = 32768
    x = linspace(-5812.7321, 5812.512, num=n_data)
    y = linspace(-5812.7321, 5812.512, num=n_data)
    fig, ax = plotpg(x, y, title='title', xlabel='xlabel', ylabel='ylabel',
                     name='0')

    for ii in range(1, 7):
        plotpg(x - 500 * ii, y + 500 + ii,
               name=f'{ii}', ax=ax)

    return fig, ax


if __name__ == '__main__':
    QAPP = pg.mkQApp()
    # fig = plot_image()
    fig, ax = plot_1d()
    QAPP.exec()