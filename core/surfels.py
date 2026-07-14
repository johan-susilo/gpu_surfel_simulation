import pickle
from typing import Tuple, List, Any
import numpy as np
from scipy.spatial.kdtree import KDTree  # type: ignore[import]

from core.vtk_utils import read_vtp_file, get_data_array, get_points
from core.types import Array
from core.parameters import Parameters

np.set_printoptions(legacy="1.25")


class Surfels:
    def __init__(self, surfel_elements: List[List[Any]], parameters: Parameters):
        surfel_p, surfel_n, surfel_i = Surfels.build_surfels(surfel_elements, parameters)

        assert surfel_p.shape == surfel_n.shape
        assert surfel_p.shape[0] == surfel_i.shape[0]

        self.x = surfel_p
        self.n = surfel_n
        self.cell_index = surfel_i.astype(np.int64)

        self.d0: float = self.estimate_d0() if parameters.d0 == "auto" else parameters.d0

        print(
            f"Surfels loaded from {surfel_elements[0]} : | number of surfels: {self.x.shape[0]} | Coord. range: {np.min(self.x), np.max(self.x)} | d0: {self.d0}"
        )

        self.phase = np.zeros_like(self.cell_index)

        self.makes_contact_with_different_cell = None
        self.neighbor_count = np.zeros((self.x.shape[0],)).astype(int)

        self.log_color = np.zeros((self.x.shape[0],))

    @staticmethod
    def make_sphere(d0: float, center: np.ndarray, radius: float) -> Tuple[Array, Array]:
        # Build sphere based on fibonacci spiral
        # see: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

        area_sphere = 4 * np.pi * (radius**2)
        # area_point = 1.0

        # surfels have unit area
        area_point = 1.0
        n_points = int(area_sphere / area_point)

        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

        y = np.linspace(-1, 1, num=n_points)
        r = np.sqrt(1 - y * y)
        theta = phi * np.arange(n_points)

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        normals = np.stack((x, y, z), axis=-1)
        positions = center + radius * normals

        return (positions, normals)

    @staticmethod
    def vtp_point_cloud_to_surfel(
        vtp_in, cell_index_col="lbl_cell", flip_x_z=True, scale=1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The XYZ coordinates, and the cell_index can be read directly from the vtp file
        The normals are constructed as a unit vector pointing outwards from the centroid
        """

        poly = read_vtp_file(vtp_in)
        cell_index = get_data_array(poly, field_type="POINT", attribute=cell_index_col)

        surfel_p = get_points(poly) / scale
        if flip_x_z:
            surfel_p = surfel_p[:, [2, 1, 0]]

        surfel_n = np.zeros_like(surfel_p)

        for cell_indexi in np.unique(cell_index):
            point_idx = np.where(cell_index == cell_indexi)[0]
            normal = surfel_p[point_idx]
            centroid = np.mean(normal, axis=0)
            normal -= centroid
            normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
            surfel_n[point_idx, :] = normal

            # print(centroid)
            # print(np.average(np.sqrt(np.sum((surfel_p[point_idx] - centroid) ** 2, axis=1))))

            # scale all points toward centroid to avoid overlaps
            surfel_p[point_idx] = surfel_p[point_idx] * 0.95 + centroid * 0.05

        return surfel_p, surfel_n, cell_index

    @staticmethod
    def build_surfels(
        surfel_elements: List[List[Any]], parameters: Parameters
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constructing surfels using surfel_elements

        options
            [point, [0, 0, 0], [0, 1, 0], 0]  # xyz, normal, cell_index
            [sphere, [30, 41, 50], 20] # centroid, radius
            [pickle, "data/Base_sim_log_359.pkl"] # a pickled Surfel object
            [vtpPointCloud, "data/sdt_pcd_ixt00.vtp", 'lbl_cell']  # name, vtp input, column where cell_index can be extracted
        """

        surfel_p = []
        surfel_n = []
        cell_index = []
        fixed = []
        for ix, surfel_el in enumerate(surfel_elements):
            if surfel_el[0] == "sphere":
                center = surfel_el[1]
                radius = surfel_el[2]
                index = surfel_el[3] if 3 < len(surfel_el) else ix

                assert isinstance(parameters.d0, float), "You must provide d0 when adding spheres"

                sphere_p, sphere_n = Surfels.make_sphere(parameters.d0, np.array(center), radius)
                cell_index.append(np.repeat(index, sphere_p.shape[0]))
                surfel_p.append(sphere_p)
                surfel_n.append(sphere_n)
            elif surfel_el[0] == "point":
                _, center, normal, cell_index_pt = surfel_el
                cell_index.append(cell_index_pt)
                surfel_p.append(np.array(center).reshape(1, 3))
                surfel_n.append(np.array(normal).reshape(1, 3))
            elif surfel_el[0] == "pickle":
                _, pickle_file = surfel_el
                with open(pickle_file, "rb") as file:
                    data = pickle.load(file)
                    cell_index.append(data["cell_index"])
                    if "y" in data:
                        print("WARNING: y datastructure is deprecated")
                        surfel_p.append(data["y"][:, :, 0])
                        surfel_n.append(data["y"][:, :, 1])
                    else:
                        surfel_p.append(data["x"])
                        surfel_n.append(data["n"])
                    print(f"Surfels loaded from {pickle_file}")
            elif surfel_el[0] == "vtpPointCloud":
                _, vtp_file, col_cell_index = surfel_el

                # instead of scaling input, pick appropriate d0 (d0=auto)
                vtp_p, vtp_n, vtp_i = Surfels.vtp_point_cloud_to_surfel(
                    vtp_file, col_cell_index, scale=parameters.scale
                )
                cell_index.append(vtp_i)
                surfel_p.append(vtp_p)
                surfel_n.append(vtp_n)
            elif surfel_el[0] == "cylinder":
                center = np.array(surfel_el[1])
                radius = surfel_el[2]
                height = surfel_el[3] * 0.5
                assert isinstance(parameters.d0, float)
                d0 = parameters.d0 * 1.05

                n_circle = int((2 * np.pi * radius) / d0)

                p_list: List[Array] = []
                n_list: List[Array] = []
                i_list = []
                i = 0

                # height of equilateral triangle tiling
                h = np.sqrt(3.0) / 2.0

                h_step = d0 * h
                n_h = int(height / h_step)
                h_step = height / n_h
                n_h = 2 * n_h + 1
                for i in range(n_h):
                    z = -height + i * h_step
                    u = np.linspace(0, 2 * np.pi, num=n_circle, endpoint=False)
                    u += i * (np.pi / n_circle)
                    x = np.cos(u)
                    y = np.sin(u)
                    p_list.append(np.stack([radius * x, radius * y, z * np.ones_like(x)], axis=-1))
                    n_list.append(np.stack([x, y, np.zeros_like(x)], axis=-1))

                    fix = ix
                    if i == 0 or i == n_h - 1:
                        fix = -1
                    i_list.append(fix * np.ones_like(x))
                p = np.vstack(p_list)
                n = np.vstack(n_list)

                cell_index.append(np.hstack(i_list))
                surfel_p.append(p)
                surfel_n.append(n)
            elif surfel_el[0] == "mesh":
                import igl
                from core.poisson_sample import sample_points_poisson_disk

                path = surfel_el[1]
                v, _, _, f, _, _ = igl.read_obj(path)

                # swap XYZ -> ZYX
                v = np.vstack([v[:, 2], v[:, 1], v[:, 0]]).T
                # flip triangles
                f = f[:, ::-1]

                ff, c = igl.bfs_orient(f)

                MIN_TRIS = 100

                a_sum = 0

                # TODO: estimate initial points density? or pass as input parameter
                n_points_total = 8000

                areas = {}
                n_points = {}
                for i in np.unique(c):
                    if np.sum(c == i) > MIN_TRIS:
                        f_select = f[c == i]
                        print(f_select.shape)

                        # We can use the mesh to estimate initial normals properly
                        a = np.sum(igl.doublearea(v, f_select))
                        a_sum += a
                        print(a)
                        areas[i] = a
                print("total area: ", a_sum)

                for k in areas.keys():
                    n_points[k] = int(n_points_total * areas[k] / a_sum)

                print(n_points)

                # Loop over connected components
                for i in np.unique(c):
                    if np.sum(c == i) > MIN_TRIS:
                        f_select = f[c == i]
                        # We can use the mesh to estimate initial normals properly
                        n = igl.per_vertex_normals(v, f_select)

                        n_pts = n_points[i]

                        points, normals, r = sample_points_poisson_disk(v, f_select, n, n_pts)

                        # "Deflate" initial points slightly
                        points -= 1.0 * r * normals

                        surfel_p.append(points)
                        surfel_n.append(normals)
                        cell_index.append(i * np.ones((n_pts,)))

            else:
                assert False, f'Unknown surfel type "{surfel_el[0]}"'

        return np.vstack(surfel_p), np.vstack(surfel_n), np.hstack(cell_index)

    def estimate_d0(self):
        """Estimate smoothing length such that on average each points has ~20 neighbors"""
        tree = KDTree(self.x)
        dist, _ = tree.query(self.x, k=20)
        d0 = np.mean(dist[:, -1])

        print(f"Estimated d0: {d0}")
        return d0

    def estimate_volume(self, c_index, density):
        # assert False, "TODO"

        # Calculate sum of signed cones to estimate the volume
        # This only gives sensible results assuming the surfels have a constant radius d0, and describe a closed surface
        # Should act correctly with non-convex shapes

        subset = self.cell_index == c_index
        centroid = np.average(self.x[subset], axis=0)
        d = density[subset]

        # TODO: maybe get a more accurate estimate based on local density?
        # surfel_area = np.pi * (self.d0**2)
        # surfel_area = np.pi * (d**2)
        surfel_area = 1 / d
        # surfel_area = 1

        dist = self.x[subset] - centroid
        r = np.sqrt(np.sum(dist**2, axis=1))

        # base area is projection of a disk onto the normal plane
        areas = surfel_area * np.einsum("ij, ij->i", dist / r[:, np.newaxis], self.n[subset])

        # volume of a cone = area * h / 3
        return np.sum(areas * r / 3)
