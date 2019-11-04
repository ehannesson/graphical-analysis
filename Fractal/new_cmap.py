# new_cmap.py
import os
import pickle
import numpy as np
import progressbar
import scipy.interpolate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def vmax_hash(sub_path):
    dirs = os.listdir(sub_path)
    dirs.sort()
    hash_dict = dict()
    max_iter = 0
    # for _dir in dirs:
    #     with open(sub_path + '/' + _dir + '/max_iter', 'rb') as f:
    #         temp_max = pickle.load(f)
    #     if temp_max > max_iter:
    #         max_iter = temp_max
    for _dir in dirs:
        # load subplot data
        with open(sub_path + '/' + _dir + '/escape', 'rb') as f:
            esc = pickle.load(f)

        # update max escape iterations
        temp_max = np.max(esc)
        if temp_max > max_iter:
            max_iter = temp_max

        # add esc hash to hash_dict if it isn't already there
        if hash(esc.tobytes()) not in hash_dict.keys():
            hash_dict[hash(esc.tobytes())] = 'temp{}.png'.format(_dir)

    return max_iter, hash_dict

def julia_from_data(Z, max_iter, save, dpi=1200, cmap='gist_rainbow'):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='black')

    plt.pcolormesh(Z, cmap=cmap, vmin=0, vmax=max_iter)
    plt.axis('off')
    plt.gca().axis('image')
    plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=dpi)

    return None

def combine(splits, outfile):
    print('Combining images into rows...')
    for row in range(splits):
        # string to match images in this row
        row_match = 'temp{}*.png'.format(row)
        # stitch images into a row
        os.system('convert {} +append temp{}.png'.format(row_match, row))

    print('Combining rows into final image...')

    # stitch first two rows together
    retry = 0
    while True:
        os.system('convert temp1.png temp0.png -append temp_10.png')
        time.sleep(2)
        if not os.path.exists('temp_10.png'):
            time.sleep(5)
            retry += 1
        if retry == 3:
            print('Failed to convert images')
            return

    # now stitch the rest of the rows to the temp_.png image
    for row in range(2, splits):
        retry = 0
        while not os.path.exists('temp_{}{}.png'.format(row, row-1)):
            os.system('convert temp{}.png temp_{}{} -append temp_{}{}.png'.format(row, row-1, row-2, row, row-1))
            time.sleep(5)
            retry += 1
            if retry == 4:
                print('Failed to convert images')
                return

    # rename the temp file as outfile, clean all temp files
    os.system('mv temp_{}{}.png {}'.format(splits-1, splits-2, outfile))

    return None

def new_cmap(sub_path, outfile, dpi=1200, cmap='gist_rainbow', clean=False):
    max_iter, hash_dict = vmax_hash(sub_path)
    print(max_iter)

    if os.path.exists('temp'):
        del_temp = input('temp folder already exists. delete it? [y/n]')
        if del_temp == 'y':
            os.system('rm -r temp')
        else:
            return None

    os.mkdir('temp')

    dirs = os.listdir(sub_path)
    dirs.sort()
    bar = progressbar.progressbar(dirs)
    for _dir in bar:
        with open(sub_path + '/' + _dir + '/escape', 'rb') as f:
            Z = pickle.load(f)

        # check if we have already produced an equivalent subimage
        Z_img = hash_dict[hash(Z.tobytes())]
        if os.path.exists('temp/' + Z_img):
            os.system('cp temp/{} temp/temp{}.png'.format(Z_img, _dir))
        else:
            # if we haven't already produced an equivalent image, make it now
            julia_from_data(Z, max_iter, 'temp/temp{}.png'.format(_dir), dpi=dpi, cmap=cmap)

    splits = int(np.sqrt(len(dirs)))

    combine(splits, outfile)

    if clean:
        clean = input('Clean temp files? [y/n]')
        if clean == 'y':
            os.system('rm -r temp')

    return
