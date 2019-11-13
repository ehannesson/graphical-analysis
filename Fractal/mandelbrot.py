# mandelbrot.py
import numpy as np
from decimal import Decimal
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc
import progressbar
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from opencl_mandel import mandel_cl

# # pycuda imports
# import pycuda.driver as drv
# import pycuda.tools
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# import pycuda.gpuarray as gpuarray
# from pycuda.elementwise import ElementwiseKernel

# pyopencl import
import pyopencl as cl

def mandel_set_naive(iters=50, N=5, k=2, res=(3840, 2160), window=(-2.4, 1, -1.18, 1.18),
                save=False, downsample=False, verbose=True, threshold=0.05, scale=1,
                min_esc=0):
    if verbose:
        iters = progressbar.progressbar(range(iters))
    else:
        iters = range(iters)

    # create meshgrid of our *c* values
    if window[1] - window[0] < 1e-12:
        # if our window is small, increase precision
        x = np.linspace(window[0], window[1], res[0], dtype=np.float128)
        y = np.linspace(window[2], window[3], res[1], dtype=np.float128)
    else:
        x = np.linspace(window[0], window[1], res[0])
        y = np.linspace(window[2], window[3], res[1])

    X, Y = np.meshgrid(x, y)
    C = X + Y*1j
    Z = C.copy()

    # to track escaped orbits (for coloring)
    escape = np.zeros((res[1], res[0]))
    # total pixels we are computing for
    tot_pix = Z.size
    # percentage escaped
    esc_perc = 0
    esc_win = [0]*50
    esc_wn = len(esc_win)
    try:
        for i in iters:
            # apply 'f' to our values
            Z = Z**k + C
            escaped = np.abs(Z)>N
            escape[escaped] = i+1 - np.log(np.log(np.abs(Z[escaped])))/np.log(np.abs(k))
            Z[escaped] = np.nan

            # check if we should break
            # esc_win[i%esc_wn] = Z[escaped].size/tot_pix
            # esc_perc += esc_win[i%esc_wn]
            # if sum(esc_win) < threshold and i >= 50*scale:
            #     break

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

# # C++ code for gpu_mandelbrot_set
# complex_gpu = ElementwiseKernel(
#     "pycuda::complex<float> *q, int *output, float *zoutput, int maxiter",
#     """
#     {
#         float nreal, real = 0;
#         float imag = 0;
#         output[i] = 0;
#         zoutput[i] = 0;
#         for(int curiter = 0; curiter < maxiter; curiter++) {
#             float real2 = real*real;
#             float imag2 = imag*imag;
#             nreal = real2 - imag2 + q[i].real();
#             imag = 2* real*imag + q[i].imag();
#             real = nreal;
#             if (real2 + imag2 > 4.0f){
#                 output[i] = curiter;
#                 zoutput[i] = real*real + imag*imag;
#                 break;
#             };
#         };
#     }
#     """,
#     "complex5",
#     preamble="#include <pycuda-complex.hpp>",)
#
# def cuda_mandelbrot_set(c, maxiter):
#     q_gpu = gpuarray.to_gpu(c.astype(np.complex64))
#     output_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
#     zoutput_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.float64))
#     complex_gpu(q_gpu, output_gpu, zoutput_gpu, maxiter)
#
#     return output_gpu.get(), zoutput_gpu.get()

# # pyopencl gpu computations
# ctx = cl.create_some_context(interactive=True)
#
# def _opencl_mandelbrot_set(c, maxiter):
#     global ctx
#
#     queue = cl.CommandQueue(ctx)
#     output = np.empty(c.shape, dtype=np.uint16)
#     z_output = np.empty(c.shape, dtype=np.float64)
#
#     prg = cl.Program(ctx, """
#     #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#     __kernel void mandelbrot(__global float2 *c, __global float *z_output,
#                     __global ushort *output, ushort const maxiter)
#     {
#         int gid = get_global_id(0);
#         float real = c[gid].x;
#         float imag = c[gid].y;
#         output[gid] = 0;
#         z_output[gid] = 0.0f;
#         for(int curiter = 0; curiter < maxiter; curiter++) {
#             float real2 = real*real, imag2 = imag*imag;
#             if (real2 + imag2 > 4.0f){
#                 z_output[gid] = real*real + imag*imag;
#                 output[gid] = curiter + 1;
#                 return;
#             }
#             imag = 2* real*imag + c[gid].y;
#             real = real2 - imag2 + c[gid].x;
#         }
#     }
#     """).build()
#
#     mf = cl.mem_flags
#     c_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
#     z_output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, z_output.nbytes)
#     output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
#
#     prg.mandelbrot(queue, output.shape, None, c_opencl,
#                     z_output_opencl, output_opencl, np.uint16(maxiter))
#
#     cl.enqueue_copy(queue, output, output_opencl).wait()
#     cl.enqueue_copy(queue, z_output, z_output_opencl).wait()
#
#     return output, z_output


def calc_fractal_opencl(q, maxiter):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    output = np.empty(q.shape, dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, real = 0;
        float imag = 0;
        output[gid] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;
            if (real*real + imag*imag > 4.0f) {
                 output[gid] = cur
#     with open('iters-test', 'wb') as f:
#         pickle.dump(iters, f)
#     with open('zvals-test', 'wb') as f:
#         pickle.dump(zvals, f)iter + 1;
                 return;
            }
        }
    }
    """).build()

    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

def _mandlebrot_set(res=(3840, 2160), window=(-2.4, 1, -1.18, 1.18),
                    iters=2000):
    x = np.linspace(window[0], window[1], res[0], dtype=np.float64)
    y = np.linspace(window[2], window[3], res[1], dtype=np.float64)
    c = np.ravel(x + y[:, np.newaxis]).astype(np.complex64)
    # c = np.ravel(c)
    output = calc_fractal_opencl(c, iters)
    # print(output)
    # output = (output[0].reshape((res[1], res[0])),
    #           output[1].reshape((res[1], res[0])))
    output = output.reshape((res[1], res[0]))

    return output




# def opencl_mandlebrot_set(res=(4096, 2844), window=(-2.4, 1, -1.18, 1.18),
#                             iters=2000):
#     x = np.linspace(window[0], window[1], res[0], dtype=np.float64)
#     y = np.linspace(window[2], window[3], res[1], dtype=np.float64)
#     c = x + y[:, None]*1j
#     c = np.ravel(c)
#     outputs = _opencl_mandelbrot_set(c, iters)
#     outputs = (outputs[0].reshape((res[1], res[0])),
#                outputs[1].reshape((res[1], res[0])))
#
#     return outputs

def plot_mandel(cmap='jet', shading='gouraud', iters=50, N=5, k=2, escape=None,
                dpi=400, res=(3840, 2160), window=(-2.4, 1, -1.18, 1.18), lognorm=True,
                plot=True, savefig=False, **kwargs):
    if not plot:
        matplotlib.use('Agg')
        plt.switch_backend('Agg')
    if escape is None:
        X, Y, escape = mandel_set_naive(iters=iters, N=N, k=k, res=res, window=window, **kwargs)
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
                        end_width=1e-8, iters=2000, res=(1920, 1080), save=True,
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
    if verbose is True:
        needed_iters = progressbar.progressbar(range(1, needed_iters))
        verbose = False
    elif verbose is False:
        needed_iters = range(1, needed_iters)
        verbose = True
    else:
        needed_iters = range(1, needed_iters)

    # iterate until xd < end_width
    for image in needed_iters:
        # file extension w/ image number
        img_ext = '/' + ('000' + str(image))[-4:]

        try:
            # compute the mandelbrot set, save the escape data
            # mandel_set_naive(iters=iters,
            #            res=res,
            #            window=window,
            #            save = save_folder + '/' + img_ext,
            #            downsample=downsample,
            #            verbose=verbose,
            #            **kwargs
            #            )
            z_vals = mandel_cl(iters=iters,
                                res=res,
                                window=window)#,
                                # save=save)

            if save:
                with open(save_folder + '/' + img_ext, 'wb') as f:
                    pickle.dump(scaled_escape, f)

        except KeyboardInterrupt as e:
            raise e
        except Error as e:
            print(e)
            user_in = input('continue?[y/n]')
            if user_in == 'y':
                pass
            else:
                raise e
        finally:
            # update xd, yd and the new window
            xd *= rate
            yd *= rate
            window = (x-xd, x+xd, y-yd, y+yd)

    return None

def gen_image_from_data(escape_data, savepath, cmap='jet', shading='gouraud',
                        lognorm=True):
    """
    Useless docstring
    """
    # make sure to use gui-less backend
    matplotlib.use('agg')

    # load escape data (if its a path to pickled data)
    if type(escape_data) is str:
        with open(escape_data, 'rb') as f:
            escape_data = pickle.load(f)

    # set norm
    if lognorm == 'double':
        escape_data = np.log(escape_data)
        escape_data[escape_data<0] = 0
        norm = matplotlib.colors.LogNorm()
    elif lognorm:
        norm = matplotlib.colors.LogNorm()
    else:
        norm = matplotlib.colors.Normalize()

    # save dimensions for later reshaping
    w, h = escape_data.shape
    # mask data for cmap
    escape_data = np.ma.masked_where(escape_data==0, escape_data)


    # get cmap object
    try:
        cmap = plt.get_cmap(cmap)
    except:
        cmap = cc.cm[cmap]

    cmap.set_bad(color='black')

    pcolor = plt.pcolormesh(escape_data,
                            cmap=cmap,
                            shading=shading,
                            norm=norm)

    # retrieve color data
    colored = pcolor.cmap(pcolor.norm(pcolor.get_array()))
    # reshape data
    colored = colored.reshape(w, h, 4)
    # scale data to 255 and convert to uint16 (and drop alpha row (all 1s))
    colored = colored[:, :, :3]*255
    colored = colored.astype(np.uint8)

    # reverse RGB to BGR for cv2 compatibility
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    # save image with cv2
    cv2.imwrite(savepath, colored)

    # clear matplotlib because its a memory leaking monster
    plt.clf()
    plt.close()

    return None

def gen_video_from_img(img_folder, save_path, res, fps=30, encoding='mp4',
                        color=1, verbose=False):
    """
    Note that save_path should specify the filename WITHOUT the extension
    """
    # set encoding
    fourcc = VideoWriter_fourcc(*'XVID')
    # initialize video
    video = VideoWriter(save_path + '.avi', fourcc, fps, res)

    # get files in img_folder
    images = os.listdir(img_folder)
    images.sort()

    if verbose:
        images = progressbar.progressbar(images)

    for image in images:
        frame = cv2.imread(img_folder + '/' + image, color)
        video.write(frame)

    video.release()

    # convert to mp4 (if desired)
    if encoding.lower() == 'mp4':
        os.system('ffmpeg -i {}.avi {}.mp4'.format(save_path, save_path))
        os.system('rm {}.avi'.format(save_path))

    return None

def gen_zoom_img_cl(img_folder, center, window_len=(3.6, 2.025), rate=0.975,
                    end_len=1e-20, cmap='hot', res='hd', max_iter=5000,
                    iter_steps=3, shading='gouraud', lognorm=True, ctx=None,
                    downsample=False, verbose=True):
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
    # create opencl context if we did not pass one in
    if ctx is None:
        ctx = cl.create_some_context(interactive=True)

    # unpack variables
    w, h = res

    dx, dy = window_len
    x, y = center
    dx, dy = Decimal(dx), Decimal(dy)
    x, y = Decimal(x), Decimal(y)

    # cast rate as Decimal
    rate = Decimal(rate)
    approx_iters = int(np.ceil(np.log(float(end_len)/float(dx))/np.log(float(rate)))) + 1

    sts = '******************************************************'
    for image_num in range(approx_iters):
        # if image_num == 800:
        #     iter_steps = 3
        if dx < 1e-14:
            c_dtype = 'longdouble'
        else:
            c_dtype = 'double'

        if verbose:
            if dx < 1e-22:
                print("\n{}\n{}\n1E-22 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))
            elif dx < 1e-20:
                print("\n{}\n{}\n1E-20 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))
            elif dx < 1e-16:
                print("\n{}\n{}\n1E-16 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))
            elif dx < 1e-14:
                print("\n{}\n{}\n1E-14 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))
            elif dx < 1e-12:
                print("\n{}\n{}\n1E-12 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))
            elif dx < 1e-9:
                print("\n{}\n{}\n1E-09 AT ITERATION {}\n{}\n{}\n".format(sts,sts,image_num,sts,sts))


        # update image number
        image = '0000{}'.format(image_num+1)[-4:]

        # get the escape data
        zvals = mandel_cl(ctx, x, y, dx, dy, max_iter=max_iter, w=w, h=h,
                          iter_steps=iter_steps, use_dtype=c_dtype)

        # generate the images
        gen_image_from_data(zvals, img_folder + '/{}.png'.format(image),
                            cmap=cmap, shading=shading, lognorm=lognorm)

        # create new window
        dx = dx*rate
        dy = dy*rate

    return None


def gen_video_from_data(data_folder, img_folder, save_path, cmap='jet',
                        shading='gouraud', lognorm=True, fps=30, encoding='mp4',
                        color=1, verbose=False, clean_images=True):
    """
    """
    data_files = os.listdir(data_folder)
    data_files.sort()
    # grab the shape of the images to set resolution
    with open(data_folder + '/{}'.format(data_files[0]), 'rb') as f:
        im = pickle.load(f)
    if verbose:
        data_files = progressbar.progressbar(data_files)

    for file in data_files:
        # generate the images
        gen_image_from_data(data_folder + '/{}'.format(file),
                            img_folder + '/{}.png'.format(file),
                            cmap=cmap,
                            shading=shading,
                            lognorm=lognorm)

    res = im.shape[::-1]
    # generate the video
    gen_video_from_img(img_folder,
                        save_path,
                        res,
                        fps=fps,
                        encoding=encoding,
                        color=color)

    if clean_images:
        os.system('rm -r {}/*.png'.format(img_folder))
    return None





# cv2.resize
# img = cv2.imread(image)
# res = cv2.resize(img, dsize=(width, height), interpolation=cv2.ARGUMENaT)
# INTER_NEAREST
# INTER_LINEAR
# INTER_AREA
# INTER_CUBIC
# INTER_LANCZOS4

# x, y = 0.460461402798336928490700548099, 0.378282833392635582819400474591
