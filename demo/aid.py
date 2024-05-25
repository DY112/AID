import pickle
import cv2
import numpy as np
import os

class AIDResult:
    """AID result class.
    There should be 4 files in the root_path:
        name + '_input_srgb.png': input image
        name + '_output_srgb.png': output image
        name + '.tiff': raw image
        name + '.pkl': pickle file

    Args:
        root_path (str): root path of the result
        name (str): name of the result

    Example:
        >>> result = AIDResult("result", "Place684_12")
    """

    def __init__(self, root_path, name):
        rgb_in_path = os.path.join(root_path, name + '_input_srgb.png')
        grid_out_path = os.path.join(root_path, name + '_result.png')
        raw_path = os.path.join(root_path, name + '.tiff')
        pkl_path = os.path.join(root_path, name + '.pkl')

        self.rgb_in = cv2.imread(rgb_in_path) # [b, g, r]
        grid_out = cv2.imread(grid_out_path) # [b, g, r]
        h = self.rgb_in.shape[0]
        self.grid_out = cv2.resize(grid_out, (int(grid_out.shape[1] / grid_out.shape[0] * h), h))

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.chromas = data['chroma'] # (n, 2=[r, b]) 

        # chroma1, chroma2 = self.chromas
        # self.chroma1 = chroma1
        # self.chroma2 = chroma2

        self.mixmaps = data['mixmap'] # (n, h, w)

        # mixmap1, mixmap2 = self.mixmaps
        # self.mixmap1 = mixmap1
        # self.mixmap2 = mixmap2
        # import pdb; pdb.set_trace()
        illum_maps = np.einsum('nhw,nc->nhwc', self.mixmaps, self.chromas) # (n, h, w, 2)
        mixed_illum_map = np.sum(illum_maps, axis=0) # (h, w, 2)

        # illum_map1 = np.stack([mixmap1 * chroma1[0], mixmap1 * chroma1[1]], axis=0)
        # illum_map2 = np.stack([mixmap2 * chroma2[0], mixmap2 * chroma2[1]], axis=0)
        # mixed_illum_map = illum_map1 + illum_map2 # [r, b]

        ones = np.ones((*mixed_illum_map.shape[:2], 1))
        mixed_illum_map = np.concatenate([mixed_illum_map[..., :1], ones, mixed_illum_map[..., 1:]], axis=2) # (n, h, w, 3=[r, g, b]) 

        # ones = np.ones_like(mixed_illum_map[0])
        # mixed_illum_map = np.stack([mixed_illum_map[0], ones, mixed_illum_map[1]], axis=0) # [r, g, b]
        # mixed_illum_map = np.transpose(mixed_illum_map, (1, 2, 0))

        raw = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED) # [b, g, r]
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 1023.0 # [r, g, b]

        raw_wb = np.clip(raw / mixed_illum_map, 0, 1)
        self._raw_wb = raw_wb  # [r, g, b]

        raw_wb_gamma = raw_wb ** (1 / 2.2)
        self.raw_wb = (raw_wb_gamma * 255).astype(np.uint8) # [r, g, b]

        cam2srgb = [[1.26749312212703, 0.538554442735016, -0.806047564862046],
        [-0.310501790822702, 1.72121703773475, -0.410715246912047],
        [-0.0835698217586037, -0.611945832583168, 1.69551565434177]]
        cam2srgb = np.array(cam2srgb)
        self.cam2rgb = cam2srgb

        img_wb_srgb = np.clip(np.matmul(raw_wb, cam2srgb.T), 0, 1)
        img_wb_srgb_gamma = img_wb_srgb ** (1 / 2.2)
        self.rgb_wb = (img_wb_srgb_gamma * 255).astype(np.uint8) # [r, g, b]

    def _tinted(self, tint1, tint2):
        # tint: [r, b]

        mixmap1 = self.mixmap1
        mixmap2 = self.mixmap2
        raw_wb = self._raw_wb
        cam2srgb = self.cam2rgb

        tint_chroma1 = np.array(tint1)
        tint_chroma2 = np.array(tint2)

        tint_illum_map1 = np.stack([mixmap1 * tint_chroma1[0], mixmap1 * tint_chroma1[1]], axis=0)
        tint_illum_map2 = np.stack([mixmap2 * tint_chroma2[0], mixmap2 * tint_chroma2[1]], axis=0)

        tint_mixed_illum_map = tint_illum_map1 + tint_illum_map2
        ones = np.ones_like(tint_mixed_illum_map[0])
        tint_mixed_illum_map = np.stack([tint_mixed_illum_map[0], ones, tint_mixed_illum_map[1]], axis=0)
        tint_mixed_illum_map = np.transpose(tint_mixed_illum_map, (1, 2, 0))

        tint_raw = np.clip(raw_wb * tint_mixed_illum_map, 0, 1)

        tint_raw_gamma = tint_raw ** (1 / 2.2)

        tint_img = (tint_raw_gamma * 255).astype(np.uint8)

        tint_img_srgb = np.clip(np.matmul(tint_raw, cam2srgb.T), 0, 1)
        tint_img_srgb_gamma = (tint_img_srgb ** (1 / 2.2) * 255).astype(np.uint8)
        
        return tint_img, tint_img_srgb_gamma # [r, g, b]

    def tinted(self, *tints):
        # tints: [n, r, b]
        assert len(tints) == len(self.chromas)

        mixmaps = self.mixmaps # (n, h, w)
        # mixmap1 = self.mixmap1
        # mixmap2 = self.mixmap2
        raw_wb = self._raw_wb
        cam2srgb = self.cam2rgb

        tint_chromas = np.array(tints) # (n, 2)
        # tint_chroma1 = np.array(tint1)
        # tint_chroma2 = np.array(tint2)

        tint_illum_maps = np.einsum('nhw,nc->nhwc', mixmaps, tint_chromas)
        # tint_illum_map1 = np.stack([mixmap1 * tint_chroma1[0], mixmap1 * tint_chroma1[1]], axis=0)
        # tint_illum_map2 = np.stack([mixmap2 * tint_chroma2[0], mixmap2 * tint_chroma2[1]], axis=0)

        tint_mixed_illum_map = np.sum(tint_illum_maps, axis=0)
        # tint_mixed_illum_map = tint_illum_map1 + tint_illum_map2

        ones = np.ones((*tint_mixed_illum_map.shape[:2], 1))
        # tint_mixed_illum_map = np.concatenate([tint_mixed_illum_map, ones], axis=2) # (n, h, w, 3=[r, g, b]) 
        tint_mixed_illum_map = np.concatenate([tint_mixed_illum_map[..., :1], ones, tint_mixed_illum_map[..., 1:]], axis=2)
        # ones = np.ones_like(tint_mixed_illum_map[0])
        # tint_mixed_illum_map = np.stack([tint_mixed_illum_map[0], ones, tint_mixed_illum_map[1]], axis=0)
        # tint_mixed_illum_map = np.transpose(tint_mixed_illum_map, (1, 2, 0))

        tint_raw = np.clip(raw_wb * tint_mixed_illum_map, 0, 1)

        tint_raw_gamma = tint_raw ** (1 / 2.2)

        tint_img = (tint_raw_gamma * 255).astype(np.uint8)

        tint_img_srgb = np.clip(np.matmul(tint_raw, cam2srgb.T), 0, 1)
        tint_img_srgb_gamma = (tint_img_srgb ** (1 / 2.2) * 255).astype(np.uint8)
        
        return tint_img, tint_img_srgb_gamma # [r, g, b]