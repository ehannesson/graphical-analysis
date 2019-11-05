# mandelbrot.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import progressbar
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def mandel_set(iters=50, N=5, k=2, res=(4096, 2844), window=(-2.4, 1, -1.18, 1.18),
                save=False, downsample=False, verbose=True):
#    if verbose:
#        iters = progressbar.progressbar(range(iters))
#    else:
    iters = range(iters)

    # create meshgrid of our *c* values
    x = np.linspace(window[0], window[1], res[0])
    y = np.linspace(window[2], window[3], res[1])
    X, Y = np.meshgrid(x, y)
    C = X + Y*1j
    # del x, y, X, Y
    # savespots = [1000, 5000, 10000, 20000, 50000, 100000, 250000, 500000]
    if window[1] - window[0] < 1e-12:
        Z = np.zeros_like(C, dtype=np.longcomplex)
        # C = C.astype(longcomplex)
    else:
        Z = np.zeros_like(C)

    escape = np.zeros((res[1], res[0]))

    try:
        for i in iters:
            # apply 'f' to our values
            Z = Z**k + C
            escaped = np.abs(Z)>N
            escape[escaped] = i+2 - np.log(np.log(np.abs(Z[escaped])))/np.log(np.abs(k))
            Z[escaped] = np.nan

            # if i in savespots:
            #     sv = save + str(i)
            #     with open('mandelbrot/escape{}'.format(i), 'wb') as f:
            #         pickle.dump(escape, f)

    except KeyboardInterrupt:
        # prompt user to choose if current progress should be saved
        print('Keyboard interrupt detected. Halting iteration.')
        user_in = input('Would you like to save the current array?[y/n]')

        if user_in.lower() in ['y', 'yes']:
            print('Saving current progress...')
            # check if current file already exists
            if os.path.exists(save + 'dump-{}'.format(i)):
                # chose new file name to save as
                print('Default file already exists.')
                file_name = input('Please enter a new filename: ')
            else:
                file_name = save + 'dump-{}'.format(i)
            # save data to file_name
            with open(file_name, 'wb') as f:
                pickle.dump(escape, f)

        raise KeyboardInterrupt

    if save:
        with open(save, 'wb') as f:
            pickle.dump(escape, f)

    return X, Y, escape

def plot_mandel(cmap='jet', shading='gouraud', iters=50, N=5, k=2, escape=None,
                dpi=400, res=(4096, 2844), window=(-2.4, 1, -1.18, 1.18), lognorm=True,
                plot=True, savefig=False, **kwargs):
    if not plot:
        matplotlib.use('Agg')
        plt.switch_backend('Agg')
    if escape is None:
        X, Y, escape = mandel_set(iters=iters, N=N, k=k, res=res, window=window, **kwargs)
    else:
        x, y = escape.shape
        x = np.linspace(-1, 1, x)
        y = np.linspace(-1, 1, y)
        X, Y = np.meshgrid(y, x)

    vmax = np.max(escape)

    escape = np.ma.masked_where(escape == 0, escape)
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='black')

    if lognorm:
        norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=1, vmax=vmax)

    # if escape is None:
    plt.pcolormesh(X, Y, escape,
                    shading=shading,
                    norm=norm,
                    cmap=cmap)
    # else:
    #     plt.pcolormesh(escape,
    #                     shading=shading,
    #                     norm=norm,
    #                     cmap=cmap)

    plt.axis('off')
    plt.gca().axis('image')
    # xmid = (window[1] + window[0])/2
    # plt.plot([xmid, xmid], [window[2], window[3]], c='black')

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0, dpi=dpi)

    if plot:
        plt.show()
    return

def gen_zoom_data(center, save_folder, window_len=(3.6, 2.025), rate=0.95,
                        end_width=1e-8, iters=2000, res=(1920, 1080),
                        downsample=False, verbose=False, **kwargs):
    """
    Parameters:

        res (tuple(int, int) or str): if tuple of ints, sets the pixel resolution
            width x height. The following strings set various ~16:9 resolutions.
            Accepted strings (can be upper- or lower-case):
                'ultrafast':    (596, 334)
                'veryfast':     (800, 450)
                'fast':         (1024, 578)
                'hd':           (1920, 1080)
                'uhd' or '4k':  (3840, 2160)
                '8k':           (7680, 4320) (don't use this, it will take forever)
    """
    # dictionary mapping res str arguments to numerical resolutions
    res_map = {'ultrafast': (596, 334),
               'veryfast':  (800, 450),
               'fast':      (1024, 578),
               'hd':        (1920, 1080),
               'uhd':       (3840, 2160),
               '4k':        (3840, 2160),
               '8k':        (7680, 4320)
               }
    try:
        # set resolution if input was string
        res = res_map[res]
    except KeyError:
        if type(res) is not tuple:
            raise ValueError('res must be type str or tuple(int, int)')
        else:
            pass


    # window width/height (half, that is)
    xd, yd = window_len
    # x/y centers
    x, y = center
    # starting window
    window = (x-xd, x+xd, y-yd, y+yd)

    # find required iterations
    needed_iters = int(np.ceil(np.log(end_width/xd)/np.log(rate))) + 1

#    if verbose is True:
#        needed_iters = progressbar.progressbar(range(1, needed_iters))
#        verbose = False
#    elif verbose is False:
    needed_iters = range(1, needed_iters)
#     verbose = True
#    else:
#        needed_iters = range(1, needed_iters)

    # iterate until xd < end_width
    for image in needed_iters:
        # file extension w/ image number
        img_ext = '/' + ('000' + str(image))[-4:]

        try:
            # compute the mandelbrot set, save the escape data
            mandel_set(iters=iters,
                       res=res,
                       window=window,
                       save = save_folder + '/' + img_ext,
                       downsample=downsample,
                       verbose=verbose,
                       **kwargs
                       )
        except KeyboardInterrupt as e:
            raise e
        except:
            continue
        finally:
            # update xd, yd and the new window
            xd *= rate
            yd *= rate
            window = (x-xd, x+xd, y-yd, y+yd)

    return None

def gen_zoom_images(data_folder, save_folder, cmap='inferno', dpi=600,
                        verbose=False, **kwargs):
    """
    """
    images = os.listdir(data_folder)
    images.sort()
#    if verbose:
#        images = progressbar.progressbar(images)

    for image in images:
        try:
            with open(data_folder + '/' + image, 'rb') as f:
                escape = pickle.load(f)

            # generate and save image
            plot_mandel(cmap=cmap,
                        escape=escape,
                        plot=False,
                        savefig=save_folder + '/' + image,
                        **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except:
            continue

    return None

# gen_zoom_images('mandelbrot/zoom1/data',
#                 'mandelbrot/zoom1/images',
#                 dpi=300,
#                 verbose=True)
# cv2.resize
# img = cv2.imread(image)
# res = cv2.resize(img, dsize=(width, height), interpolation=cv2.ARGUMENT)
# INTER_NEAREST
# INTER_LINEAR
# INTER_AREA
# INTER_CUBIC
# INTER_LANCZOS4



























# good zoom = {'xc': -1.7685306251475799, 'yc': 0.0008426158256747134, 'xd': 6.400000000000001e-15, 'yd': 4.266666666666667e-15}
# cmaps = ('inferno', 'gist_ncar', 'twilight', 'magma', 'viridis', 'gnuplot2','terrain', 'rainbow', 'gist_rainbow', 'jet', 'hsv', 'plasma', 'spring', 'cool', 'hot', 'nipy_spectral', 'brg')
#  xc
# Out[241]: -1.768530625146282
#
# In [242]: yc
# Out[242]: 0.0008426158356280194
#
# In [243]: xd
# Out[243]: 1.0000000000000003e-10
#
# In [244]: yd
# Out[244]: 6.666666666666668e-11



# import pickle
# for cmap in cmaps:
#     for iters_ in iters:
#         for N in Ns:
#             savefig1 = cmap + '-' + str(iters_) + '-' + str(N) + '-flat.png'
#             savefig2 = cmap + '-' + str(iters_) + '-' + str(N) + '-logn.png'
#             save = str(iters_) + '-' + str(N)
#             try:
#                 plot_mandel(cmap=cmap, iters=iters_,N=N, dpi=1200,plot=False,savefig=savefig1, save=save)
#             except MemoryError:
#                 print('mem error bitch')
#                 pass
#             except KeyboardInterrupt as e:
#                 raise e
#             finally:
#                 pass
#             try:
#                 with open(save, 'rb') as f:
#                     Z = pickle.load(f)
#                 plot_mandel(cmap=cmap,escape=Z,dpi=1200,plot=False,savefig=savefig2)
#             except MemoryError:
#                 print('mem error bitch')
#                 continue
#             except KeyboardInterrupt as e:
#                 raise e
#             finally:
#                 pass

# def GETSHITDONE():
# for iters_ in iters:
#     for N in Ns:
#         mandel_set(iters=iters_,
#                     N=N,
#                     save='res-4096-iters-' + str(iters_) + '-N-' + str(N))
#         mandel_set(iters=iters_,
#                     res=(5120, 3414),
#                     N=N,
#                     save='res-5120-iters-' + str(iters_) + '-N-' + str(N))

# def GENPLOTSBITCHES():
# for cmap in cmaps:
#     for iters_ in iters:
#         for N in Ns:
#             fn = 'mandelbrot/escape' + 'res-4096-iters-' + str(iters_) + '-N-' + str(N)
#             with open(fn, 'rb') as f:
#                 Z = pickle.load(f)
#             savefig1 = cmap + '-' + str(iters_) + '-' + str(N) + '-flat.png'
#             savefig2 = cmap + '-' + str(iters_) + '-' + str(N) + '-logn.png'
#             kwargs_ = {'cmap': cmap, 'escape': Z, 'dpi': 1200, 'plot': False, 'savefig': savefig1}
#             try:
#                 p = mp.Process(target=plot_mandel, kwargs=kwargs_)
#                 p.daemon=True
#                 p.start()
#                 p.join()
#             except KeyboardInterrupt as e:
#                 raise e
#             except:
#                 pass
#
#             kwargs_ = {'cmap': cmap, 'escape': Z, 'dpi': 1200, 'plot': False, 'lognorm': True, 'savefig': savefig2}
#             try:
#                 p = mp.Process(target=plot_mandel, kwargs=kwargs_)
#                 p.daemon=True
#                 p.start()
#                 p.join()
#             except KeyboardInterrupt as e:
#                 raise e
#             except:
#                 pass
