import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import]

from core.parameters import Parameters


class Signal:
    def __init__(self, parameters: Parameters):
        image_path = parameters.paths["img_pixels"]
        self.im = np.load(image_path)

        scale_z = parameters.z_res * parameters.signal_downsample / parameters.scale
        scale_xy = parameters.xy_res * parameters.signal_downsample / parameters.scale

        # offset = np.array([-0.925719, -1.735312, -2.288967])
        # offset = np.array([-22.89, -17.35, -9.26])
        offset = np.array([-2.289, -1.735, -0.926])
        # offset = np.array([0, 0, 0])

        self.points = None
        if "img_points" in parameters.paths:
            points_path = parameters.paths["img_points"]
            self.points = np.load(points_path)
            # FIXME
            # self.points = np.vstack([self.points[:, 2], self.points[:, 1], self.points[:, 0]]).T
            # self.points = self.points * np.array([scale_xy, scale_xy, scale_z / 1.41]) + offset
            self.points = self.points * np.array([scale_z / 1.41, scale_xy, scale_xy]) + offset

        print("========")
        print(np.min(self.points, axis=0))
        print(np.max(self.points, axis=0))
        print(self.im.shape)

        z = np.arange(self.im.shape[0]) * scale_z * 2 + offset[0]
        y = np.arange(self.im.shape[1]) * scale_xy * 2 + offset[1]
        x = np.arange(self.im.shape[2]) * scale_xy * 2 + offset[2]

        print(np.min(z), np.min(y), np.min(x))
        print(np.max(z), np.max(y), np.max(x))

        # Does not give any bounds error!
        # We should probably add some sort of constraint that it can't go outside
        self.interpolator = RegularGridInterpolator(
            (z, y, x),
            # (x, y, z),
            self.im,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
