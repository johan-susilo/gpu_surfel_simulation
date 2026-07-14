"""
@author: wth
"""

import numpy as np
import pyvista as pv  # type: ignore[import]
import vtk  # type: ignore[import]
import vtk.numpy_interface.dataset_adapter as dsa  # type: ignore[import]


def read_vtp_file(vtp_file):
    vtp_reader = vtk.vtkXMLPolyDataReader()
    vtp_reader.SetFileName(str(vtp_file))
    vtp_reader.Update()

    return vtp_reader.GetOutput()  # vtkPolyData


def write_vtp_file(polyout, path_file):
    vtk_writer = vtk.vtkXMLPolyDataWriter()
    vtk_writer.SetDataMode(vtk.vtkXMLWriter.Ascii)
    vtk_writer.SetByteOrderToBigEndian()
    vtk_writer.SetFileName(str(path_file))
    vtk_writer.SetInputData(polyout)

    return vtk_writer.Write()


def get_points(poly):
    return pv.wrap(poly).points


def get_data_array(poly, field_type="CELL", attribute="parentIndex", verbose=True):
    """get back a point or cell data array"""
    if field_type == "CELL":
        data_array = dsa.WrapDataObject(poly).CellData[attribute]
    elif field_type == "POINT":
        data_array = dsa.WrapDataObject(poly).PointData[attribute]

    if isinstance(data_array, vtk.numpy_interface.dataset_adapter.VTKNoneArray):
        if verbose:
            print(
                "{} is no attribute in this vtk object (field_type={})".format(
                    attribute, field_type
                )
            )
        return None

    if str(data_array).startswith("vtkString"):
        data_array = pv.convert_string_array(data_array)

    return data_array


def add_array(poly, a_added, name_array, field_type="CELL"):
    """use this to append a data array. This also works for replacing an array"""

    if isinstance(a_added[0], bytes) or (a_added.dtype.type is np.str_):
        a_added = pv.convert_string_array(a_added)
        a_added.SetName(name_array)
        if field_type == "CELL":
            poly.GetCellData().AddArray(a_added)
        elif field_type == "POINT":
            poly.GetPointData().AddArray(a_added)

    else:
        wdo = dsa.WrapDataObject(poly)
        if field_type == "CELL":
            wdo.GetAttributes(vtk.vtkDataObject.CELL).append(a_added, name_array)
        elif field_type == "POINT":
            wdo.GetAttributes(vtk.vtkDataObject.POINT).append(a_added, name_array)

    return poly


def add_vector_to_poly(poly, vector, vectorName):
    # no need to add the norm, paraview apparently adds this automaticaly
    # add_array(
    #     poly,
    #     np.linalg.norm(vector, axis=1),
    #     f"{vectorName}_norm",
    #     field_type="POINT",
    # )
    add_array(poly, vector, vectorName, field_type="POINT")


def simlog_to_vtp(x, n, cell_index, phase, force_obj, neighbor_count, output_vtp):
    poly = pv.PolyData(x)

    n_pts = x.shape[0]

    assert n.shape[0] == n_pts
    assert cell_index.shape[0] == n_pts
    assert phase.shape[0] == n_pts
    assert neighbor_count.shape[0] == n_pts

    add_array(poly, cell_index, "cell_index", field_type="POINT")
    add_array(poly, phase, "phase", field_type="POINT")
    add_array(poly, neighbor_count, "neighbor_count", field_type="POINT")
    add_vector_to_poly(poly, n, "n")
    if force_obj:
        for forceType, force in force_obj.__dict__.items():
            if not forceType.startswith("force_obj"):
                continue
            if forceType == "f_result":
                # only store the resulting force, not the normal delta
                force = force[..., 0]
            add_vector_to_poly(poly, force, forceType)

    write_vtp_file(poly, output_vtp)


def signal_to_vtp(points, output_vtp):
    write_vtp_file(pv.PolyData(points), output_vtp)
