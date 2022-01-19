import torch
import tensorflow as tf
import numpy as np
import torch.nn.functional as F


def grid_sample_tf(image, grid, align_corners=False, padding='border'):
    def grid_sampler_unnormalize(coord, side, align_corners, padding='border'):
        if align_corners:
            x = ((coord + 1) / 2) * (side - 1)
        else:
            x = ((coord + 1) * side - 1) / 2

        if padding == 'border':
            eps = 0.0
            x = tf.clip_by_value(tf.cast(x, tf.float32), eps, tf.cast(side, tf.float32) - 1.0)

        return x

    def grid_sampler_compute_source_index(coord, size, align_corners, padding='border'):
        coord = grid_sampler_unnormalize(coord, size, align_corners, padding)
        return coord

    def safe_get(image, n, c, x, y, H, W):
        value = tf.zeros([], dtype='int32')
        if x >= 0 and x < W and y >= 0 and y < H:
            n = tf.cast(n, tf.int32)
            c = tf.cast(c, tf.int32)
            y = tf.cast(y, tf.int32)
            x = tf.cast(x, tf.int32)
            value = image[n, c, y, x]
        return value

    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]

         output shape = [N, C, H, W]
    '''
    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]

    output_tensor = np.zeros(shape=np.shape(image))
    for n in range(N):
        for w in range(grid_W):
            for h in range(grid_H):
                # get corresponding grid x and y
                x = grid[n, h, w, 1]
                y = grid[n, h, w, 0]

                # Unnormalize with align_corners condition
                ix = grid_sampler_compute_source_index(x, W, align_corners, padding)
                iy = grid_sampler_compute_source_index(y, H, align_corners, padding)

                # x0 = torch.floor(ix).type(torch.LongTensor)
                x0 = tf.floor(ix)
                x1 = x0 + 1

                # y0 = torch.floor(iy).type(torch.LongTensor)
                y0 = tf.floor(iy)
                y1 = y0 + 1

                # Get W matrix before I matrix, as I matrix requires Channel information
                wa = tf.cast((x1 - ix) * (y1 - iy), tf.float32)
                wb = tf.cast((x1 - ix) * (iy - y0), tf.float32)
                wc = tf.cast((ix - x0) * (y1 - iy), tf.float32)
                wd = tf.cast((ix - x0) * (iy - y0), tf.float32)

                # Get values of the image by provided x0,y0,x1,y1 by channel
                for c in range(C):
                    # image, n, c, x, y, H, W
                    Ia = safe_get(image, n, c, y0, x0, H, W)
                    Ib = safe_get(image, n, c, y1, x0, H, W)
                    Ic = safe_get(image, n, c, y0, x1, H, W)
                    Id = safe_get(image, n, c, y1, x1, H, W)

                    Ia = tf.cast(tf.transpose(Ia), tf.float32)
                    Ib = tf.cast(tf.transpose(Ib), tf.float32)
                    Ic = tf.cast(tf.transpose(Ic), tf.float32)
                    Id = tf.cast(tf.transpose(Id), tf.float32)
                    out_ch_val = tf.transpose(Ia * wa) + tf.transpose(Ib * wb) + \
                                 tf.transpose(Ic * wc) + tf.transpose(Id * wd)

                    output_tensor[n, c, h, w] = out_ch_val
    return output_tensor

# h = 4
# w = 4
# b = 1
# c = 10
#
# tf_emb = np.tile(np.random.normal(size=[1, h, w, c]), [b, 1, 1, 1]) # B, H, W, C
# torch_emb = torch.Tensor(np.transpose(tf_emb, axes=[0, 3, 1, 2]))
# tf_emb = np.transpose(tf_emb, axes=[0, 3, 1, 2])
# tf_emb = tf.cast(tf_emb, tf.float32)
#
# x = tf.linspace(-1, 1, w)
# x = tf.reshape(x, shape=[1, 1, w, 1])
# x = tf.tile(x, multiples=[b, w, 1, 1])
# y = tf.linspace(-1, 1, h)
# y = tf.reshape(y, shape=[1, h, 1, 1])
# y = tf.tile(y, multiples=[b, 1, h, 1])
#
# tf_coords = tf.concat([x, y], axis=-1)
# torch_coords = torch.Tensor(tf_coords.numpy())
#
# tf_x2 = grid_sample_tf(tf_emb, tf_coords)
# torch_x = F.grid_sample(torch_emb, torch_coords, padding_mode='border', mode='bilinear')
# print(tf_x2)
# print(torch_x)
