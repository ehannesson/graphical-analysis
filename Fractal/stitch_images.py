# stitch_images.py
import os
import time

def stitch_images(splits, outfile):
    print('Combining images into rows...')
    for row in range(splits):
        # string to match images in this row
        row_match = 'temp{}*.png'.format(row)
        # stitch images into a row
        os.system('convert {} +append temp{}.png'.format(row_match, row))
        os.system('convert temp{}.png temp{}.jpg'.format(row, row))

    print('Combining rows into final image...')
    # stitch first two rows together
    os.system('convert temp1.jpg temp0.jpg -append temp_.jpg')
    # now stitch the rest of the rows to the temp_.png image
    for row in range(2, splits):
        time.sleep(1)
        os.system('convert temp{}.jpg temp_.jpg -append temp_.jpg'.format(row))

    # rename the temp file as outfile, clean all temp files
    os.system('mv temp_.jpg ../{}'.format(outfile))
    # os.system('rm temp*.png')
    os.chdir('../../..')

    return
