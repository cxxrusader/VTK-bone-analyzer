import pyvista as pv
import numpy as np

path_to_file = "example_mesh1.vtk"
slice_to_bone_ratio = 1/3


def min_coord(arr, coord_idx):
    return min((elem[coord_idx] for elem in arr))


def max_coord(arr, coord_idx):
    return max((elem[coord_idx] for elem in arr))


def linear_size(mesh):
    bounds = mesh.bounds
    return [bounds[i*2+1] - bounds[i*2] for i in range(3)]


def point_furthest_from_center(mesh):
    center = mesh.center
    return max(mesh.points, key=lambda p: np.linalg.norm(center-p))


def bone_ends(mesh):
    first_end = point_furthest_from_center(mesh)
    bone_middle = mesh.center
    diff_vec = bone_middle-first_end
    second_end = bone_middle+diff_vec
    return first_end, second_end


def align_to_z_axis(mesh):
    ends = bone_ends(mesh)
    ends_dir = ends[0]-ends[1]
    unit_z = (0, 0, 1)
    angle = np.degrees(np.arccos(np.dot(ends_dir, unit_z) / np.linalg.norm(ends_dir)))
    rotation_axis = np.cross(ends_dir, unit_z)
    mesh.rotate_vector(rotation_axis, angle, inplace=True)


def bone_slice(bone_mesh, bone_end_idx):
    assert bone_end_idx == 0 or bone_end_idx == 1, "bone_end_idx must be equal to 0 or 1"

    bone_length = abs(point_furthest_from_center(bone_mesh)[2] * 2)

    clipping_plane_normal = (0, 0, 1) if bone_end_idx == 0 else (0, 0, -1)

    clipping_plane_z_offset = bone_length * (1/2 - slice_to_bone_ratio)
    if bone_end_idx == 0:
        clipping_plane_z_offset = -clipping_plane_z_offset

    res = bone_mesh.clip(clipping_plane_normal, (0, 0, clipping_plane_z_offset))
    return res


def print_bone_part_info(mesh, bone_name):
    print(bone_name, ":")
    print("Linear size: ", linear_size(mesh))
    print("Volume: ", mesh.volume)


def main():
    mesh = pv.PolyData(path_to_file).triangulate()
    mesh.translate(np.negative(mesh.center), inplace=True)
    align_to_z_axis(mesh)
    print_bone_part_info(mesh, "Whole bone")

    slice1 = bone_slice(mesh, 0)
    slice2 = bone_slice(mesh, 1)

    print_bone_part_info(slice1, "Slice 1")
    print_bone_part_info(slice2, "Slice 2")

    slices = slice1.merge(slice2)
    slices.plot()


main()
