import pyvista as pv
import numpy as np

import config
from config import path_to_file, slice_to_bone_ratio, points_of_interest_names


def dist(p1, p2):
    return np.linalg.norm(np.asarray(p1)-np.asarray(p2))


class PointsSelectionHandler:
    def __init__(self, names, plotter, selection_completion_callback, visualize_clicks=True):
        assert names
        self.i = iter(names)
        self.cur_label = next(self.i)
        self.label_actor_name = "point name label"
        self.points_dict = dict()
        self.attach_callback(plotter, selection_completion_callback)
        self.plotter = plotter
        self.visualize_clicks = visualize_clicks

    def attach_callback(self, plotter, selection_completion_callback):
        plotter.add_text(self.cur_label, name=self.label_actor_name)

        def callback(click_pos):
            try:
                next_label = next(self.i)
            except StopIteration:
                self.process_current_point(click_pos)
                self.clean_up_plotter(plotter)
                selection_completion_callback(self.points_dict)
            else:
                self.process_current_point(click_pos)
                self.cur_label = next_label
                plotter.add_text(self.cur_label, name=self.label_actor_name)

        plotter.track_click_position(callback)

    def process_current_point(self, point):
        self.points_dict[self.cur_label] = point
        if self.visualize_clicks:
            visualize_points(self.plotter, (point,))

    def clean_up_plotter(self, plotter):
        plotter.untrack_click_position()
        plotter.remove_actor(self.label_actor_name)


def print_morf_info(param_name, val):
    print("{}: {:.1f} мм".format(param_name, val))


def get_styloid_and_head_points(bone_ends, head_pit):
    end1, end2 = bone_ends[0], bone_ends[1]
    return (end1, end2) if dist(end1, head_pit) > dist(end2, head_pit) else (end2, end1)


def get_slice_max_d(slice):
    points = slice.points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    points_len = len(points)
    center = (sum(x) / points_len, sum(y) / points_len, sum(z) / points_len)
    dists_to_center = (dist(p, center) for p in points)
    width = max(dists_to_center) * 2
    return width


def get_bone_ends(mesh):
    first_end = point_furthest_from_center(mesh)
    second_end = find_second_end(mesh, first_end)
    return first_end, second_end


def get_width_along_axis(slice, axis_id):
    coord = slice.points[:, axis_id]
    min_c = min(coord)
    max_c = max(coord)
    return max_c - min_c

def get_slice_width(slice):
    return get_width_along_axis(slice, 1)


def get_slice_sag_width(slice):
    return get_width_along_axis(slice, 0)


def show_slice_and_print_width(plotter, slice, displayed_info):
    visualize_slice(plotter, slice)
    slice_width = get_slice_width(slice)
    print_morf_info(displayed_info, slice_width)


def show_slice_and_print_d(plotter, slice, displayed_info):
    visualize_slice(plotter, slice)
    slice_d = get_slice_max_d(slice)
    print_morf_info(displayed_info, slice_d)


def print_slice_sag_width(slice, displayed_info):
    slice_sag_width = get_slice_sag_width(slice)
    print_morf_info(displayed_info, slice_sag_width)


def print_slice_d(slice, displayed_info):
    slice_d = get_slice_max_d(slice)
    print_morf_info(displayed_info, slice_d)


def dist_to_line(point, line):
    p3 = point
    p1, p2 = line
    dist = np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
    return dist


class MeshProcessor:
    def __init__(self, mesh, plotter):
        self.mesh = mesh
        self.plotter = plotter

    def align_to_y_and_continue(self, selected_points):
        """After call medial side of the bone will be placed towards y axis"""
        points = [np.asarray(v) for v in selected_points.values()]
        align_to_yz_plane(self.mesh, points)
        visualize_points(self.plotter, get_bone_ends(self.mesh), color=config.ends_colors)
        add_yz_plane(self.plotter, self.mesh)

        processor = MeshProcessor(self.mesh, self.plotter)
        PointsSelectionHandler(config.points_of_interest_names, self.plotter, processor.process)

    def process(self, selected_points):
        print("Морфометрические параметры:")

        first_end, second_end = get_bone_ends(self.mesh)
        print_morf_info("Наибольшая длина", dist(first_end, second_end))

        upper_pit = selected_points[config.upper_pit_point_name]
        lower_pit = selected_points[config.lower_pit_point_name]
        print_morf_info("Физиологическая длина", dist(upper_pit, lower_pit))

        styloid_process, head = get_styloid_and_head_points((first_end, second_end), upper_pit)
        lateral_head_point = selected_points[config.lateral_head_point_name]
        print_morf_info("Параллельная длина", dist(lateral_head_point, styloid_process))

        center_of_tuberosity = selected_points[config.center_of_tuberosity_name]
        print_morf_info("Расстояние от головки до бугристости", dist(center_of_tuberosity, head))

        point_on_diaphysis = selected_points[config.diaphysis_point_name]
        diaphysis_slice = self.mesh.slice(origin=point_on_diaphysis, normal="z")
        show_slice_and_print_width(self.plotter, diaphysis_slice, "Ширина диафиза")

        middle_slice = self.mesh.slice(normal="z")
        show_slice_and_print_width(self.plotter, middle_slice, "Ширина середины диафиза")

        opposite_head_point = selected_points[config.opposite_point_on_head]
        print_morf_info("Ширина головки", dist(opposite_head_point, head))

        neck_point = selected_points[config.neck_point_name]
        neck_slice = self.mesh.slice(origin=neck_point, normal="z")
        show_slice_and_print_width(self.plotter, neck_slice, "Ширина шейки")

        distal_point = selected_points[config.distal_point_name]
        distal_slice = self.mesh.slice(origin=distal_point, normal="z")
        show_slice_and_print_width(self.plotter, distal_slice, "Ширина дистального эпифиза")

        print_slice_sag_width(diaphysis_slice, "Сагиттальный диаметр диафиза")
        print_slice_sag_width(neck_slice, "Сагиттальный диаметр шейки")

        thinnest_point = selected_points[config.thinnest_point_name]
        thinnest_slice = self.mesh.slice(origin=thinnest_point, normal="z")
        show_slice_and_print_d(self.plotter, thinnest_slice, "Наименьшая окружность диафиза")

        print_slice_d(middle_slice, "Окружность середины диафиза")
        print_slice_d(neck_slice, "Окружность шейки")

        tub_highest_point = selected_points[config.tuberosity_highest_point_name]
        tub_lowest_point = selected_points[config.tuberosity_lowest_point_name]
        print_morf_info("Длина бугристости", dist(tub_lowest_point, tub_highest_point))

        tub_lat_point = selected_points[config.tuberosity_lateral_point_name]
        tub_med_point = selected_points[config.tuberosity_medial_point_name]
        print_morf_info("Ширина бугристости", dist(tub_lat_point, tub_med_point))

        tub_slice = self.mesh.slice(origin=center_of_tuberosity, normal="z")
        print_slice_sag_width(tub_slice, "Сагиттальный диаметр проксимального отдела в области бугристости")

        pit_depth = dist_to_line(upper_pit, (head, opposite_head_point))
        print_morf_info("Глубина суставной ямки", pit_depth)


def find_second_end(aligned_mesh, first_end):
    second_end_z_coord_sign = -np.sign(first_end[2])
    points_of_second_half = filter(lambda point: np.sign(point[2]) == second_end_z_coord_sign, aligned_mesh.points)
    return max(points_of_second_half, key=lambda point: abs(point[2]))


def min_coord(arr, coord_idx):
    return min((elem[coord_idx] for elem in arr))


def max_coord(arr, coord_idx):
    return max((elem[coord_idx] for elem in arr))


def linear_size(mesh):
    bounds = mesh.bounds
    return [bounds[i*2+1] - bounds[i*2] for i in range(3)]


def point_furthest_from_center(mesh):
    center = mesh.center
    return max(mesh.points, key=lambda p: dist(center, p))


def bone_axis_points(mesh):
    first_end = point_furthest_from_center(mesh)
    bone_middle = mesh.center
    diff_vec = bone_middle-first_end
    second_end = bone_middle+diff_vec
    return first_end, second_end


def angle_between_vecs(v1, v2):
    return np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))


def align_to_yz_plane(mesh, points_to_align):
    p1 = points_to_align[0]
    p2 = np.copy(points_to_align[1])
    p2[2] = p1[2]
    align_mesh(mesh, (p1, p2), (0, 1, 0))


def align_mesh(mesh, points_to_align, unit_vec):
    dir = points_to_align[0]-points_to_align[1]
    angle = angle_between_vecs(dir, unit_vec)
    rotation_axis = np.cross(dir, unit_vec)
    mesh.rotate_vector(rotation_axis, angle, inplace=True)


def align_to_z_axis(mesh):
    ends = bone_axis_points(mesh)
    unit_z = (0, 0, 1)
    align_mesh(mesh, ends, unit_z)


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
    print(f"{bone_name}:")
    print("Линейный размер: ", linear_size(mesh))
    print("Объем: ", mesh.volume)


def visualize_points(plotter, points, color="blue"):
    for point in points:
        plotter.add_mesh(pv.Sphere(3, point), color=color)


def visualize_slice(plotter, slice):
    plotter.add_mesh(slice, color="red")


def add_yz_plane(plotter, mesh):
    plane = pv.Plane(direction=(1, 0, 0), i_size=linear_size(mesh)[2], j_size=linear_size(mesh)[1])
    plotter.add_mesh(plane, color="green", opacity=0.2)


def set_up_and_run_plotter(mesh):
    plotter = pv.Plotter()
    align_medial_side_to_y(plotter, mesh)
    plotter.add_axes()
    plotter.add_mesh(mesh)
    plotter.show()


def align_medial_side_to_y(plotter, mesh):
    processor = MeshProcessor(mesh, plotter)
    points_names = ("medial point", "lateral point")
    PointsSelectionHandler(points_names, plotter, processor.align_to_y_and_continue, visualize_clicks=False)


def import_mesh():
    mesh = pv.PolyData(path_to_file).triangulate()
    mesh.translate(np.negative(mesh.center), inplace=True)
    align_to_z_axis(mesh)
    return mesh


def main():
    mesh = import_mesh()
    print_bone_part_info(mesh, "Целая кость")

    slice1 = bone_slice(mesh, 0)
    slice2 = bone_slice(mesh, 1)

    print_bone_part_info(slice1, "Конец 1")
    print_bone_part_info(slice2, "Конец 2")

    # slices = slice1.merge(slice2)
    set_up_and_run_plotter(mesh)


main()
