import pyvista as pv
import numpy as np
import pickle

import params_names
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


def rounded_string(number):
    return "{:.1f}".format(number)


def print_morf_info(param_name, val):
    print("{}: ".format(param_name) + rounded_string(val) + " мм")


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


def circumference(r):
    return 2 * np.pi * r


def get_slice_circumference(slice):
    return circumference(get_slice_max_d(slice)/2)


def dist_to_line(point, line):
    p3 = point
    p1, p2 = line
    dist = np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
    return dist


def slice_through_points(mesh, p1, p2, p3):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    n = np.cross(p1-p2, p1-p3)
    slice = mesh.slice(origin=p1, normal=n)
    return slice


def find_tub_diam_points_of_slice(slice, head_center, tub_center, height_function):
    potential_diam_points = list(filter(lambda p: height_function(p) > height_function(tub_center), slice.points))
    medial_points = filter(lambda p: p[1] > head_center[1], potential_diam_points)
    highest_med = max(medial_points, key=height_function)
    lateral_points = filter(lambda p: p[1] < head_center[1], potential_diam_points)
    most_distant_lat = max(lateral_points, key=lambda p: dist(p, tub_center))
    return highest_med, most_distant_lat


def get_head_slice(mesh, head_center):
    head_normal = np.asarray(head_center) - np.asarray(mesh.center)
    head_normal /= np.linalg.norm(head_normal)
    slice_offset = 1.5
    head_slice = mesh.slice(origin=head_center - head_normal * slice_offset, normal=head_normal)
    return head_slice


def find_nontub_diam_points_of_slice(head_slice, head_center, tub_diams):
    td1 = np.asarray(tub_diams[0])
    td2 = np.asarray(tub_diams[1])
    nontub_slice = head_slice.slice(origin=head_center, normal=td1-td2)
    p1 = np.copy(max(nontub_slice.points, key=lambda p: p[0] > head_center[0]))
    p1[2] = head_center[2]
    p2 = np.copy(max(nontub_slice.points, key=lambda p: p[0] < head_center[0]))
    p2[2] = head_center[2]
    return p1, p2


def save_params(params_dict):
    with open(config.path_to_programmatically_measured_params, "wb") as f:
        pickle.dump(params_dict, f)


def request_params_saving(params_dict):
    print("Желаете сохранить измеренные параметры? (ДА/нет)")
    print(">", end="")
    answer = input().lower()
    while True:
        if answer in ["н", "нет"]:
            return
        if answer in ["д", "да", ""]:
            save_params(params_dict)
            print("Параметры будут сохранены после закрытия программы")
            return
        print("Недопустимый ввод. Попробуйте еще раз:")
        answer = input().lower()


class MeshProcessor:
    def __init__(self, mesh, plotter, head_upper_points=None):
        self.head_upper_points = head_upper_points
        self.mesh = mesh
        self.plotter = plotter
        self.params_dict = dict()

    def align_to_y_and_continue(self, selected_points):
        """After call medial side of the bone will be placed towards y axis"""
        points = [np.asarray(v) for v in selected_points.values()]
        align_to_yz_plane(self.mesh, points)
        visualize_points(self.plotter, get_bone_ends(self.mesh), color=config.ends_colors)
        add_yz_plane(self.plotter, self.mesh)

        processor = MeshProcessor(self.mesh, self.plotter)
        PointsSelectionHandler(config.points_of_interest_names, self.plotter, processor.process)

    def show_slice_and_print_width(self, plotter, slice, displayed_info, dict_slice_name):
        visualize_slice(plotter, slice)
        slice_width = get_slice_width(slice)
        self.params_dict[dict_slice_name] = slice_width
        print_morf_info(displayed_info, slice_width)

    def print_and_save_slice_sag_width(self, slice, displayed_info, dict_slice_name):
        slice_sag_width = get_slice_sag_width(slice)
        self.params_dict[dict_slice_name] = slice_sag_width
        print_morf_info(displayed_info, slice_sag_width)

    def show_slice_and_print_circumference(self, plotter, slice, displayed_info, dict_slice_name):
        visualize_slice(plotter, slice)
        slice_circumference = get_slice_circumference(slice)
        self.params_dict[dict_slice_name] = slice_circumference
        print_morf_info(displayed_info, slice_circumference)

    def print_and_save_slice_circumference(self, slice, displayed_info, dict_slice_name):
        slice_circumference = get_slice_circumference(slice)
        self.params_dict[dict_slice_name] = slice_circumference
        print_morf_info(displayed_info, slice_circumference)

    def process(self, selected_points):
        print("Морфометрические параметры:")

        first_end, second_end = get_bone_ends(self.mesh)
        max_length = dist(first_end, second_end)
        print_morf_info("Наибольшая длина", max_length)
        self.params_dict[params_names.biggest_length_of_bone] = max_length

        upper_pit = selected_points[config.upper_pit_point_name]
        lower_pit = selected_points[config.lower_pit_point_name]
        phys_length = dist(upper_pit, lower_pit)
        print_morf_info("Физиологическая длина", phys_length)
        self.params_dict[params_names.phys_length_of_bone] = phys_length

        styloid_process, head = get_styloid_and_head_points((first_end, second_end), upper_pit)
        lateral_head_point = selected_points[config.lateral_head_point_name]
        par_length = dist(lateral_head_point, styloid_process)
        print_morf_info("Параллельная длина", par_length)
        self.params_dict[params_names.paral_length_of_bone] = par_length

        center_of_tuberosity = selected_points[config.center_of_tuberosity_name]
        head_to_tub_dist = dist(center_of_tuberosity, head)
        print_morf_info("Расстояние от головки до бугристости", head_to_tub_dist)
        self.params_dict[params_names.dist_from_head_to_tub] = head_to_tub_dist

        point_on_diaphysis = selected_points[config.diaphysis_point_name]
        diaphysis_slice = self.mesh.slice(origin=point_on_diaphysis, normal="z")
        self.show_slice_and_print_width(self.plotter, diaphysis_slice, "Ширина диафиза", params_names.diaphisys_width)

        middle_slice = self.mesh.slice(normal="z")
        self.show_slice_and_print_width(self.plotter, middle_slice, "Ширина середины диафиза",
                                        params_names.diaphisys_middle_width)

        opposite_head_point = selected_points[config.opposite_point_on_head]
        head_width = dist(opposite_head_point, head)
        print_morf_info("Ширина головки", head_width)
        self.params_dict[params_names.head_width] = head_width

        neck_point = selected_points[config.neck_point_name]
        neck_slice = self.mesh.slice(origin=neck_point, normal="z")
        self.show_slice_and_print_width(self.plotter, neck_slice, "Ширина шейки", params_names.neck_width)

        distal_point = selected_points[config.distal_point_name]
        distal_slice = self.mesh.slice(origin=distal_point, normal="z")
        self.show_slice_and_print_width(self.plotter, distal_slice, "Ширина дистального эпифиза",
                                        params_names.lower_epiphysis_width)

        self.print_and_save_slice_sag_width(diaphysis_slice, "Сагиттальный диаметр диафиза",
                                            params_names.diaphysis_sag_diam)
        self.print_and_save_slice_sag_width(neck_slice, "Сагиттальный диаметр шейки", params_names.neck_sag_diam)

        thinnest_point = selected_points[config.thinnest_point_name]
        thinnest_slice = self.mesh.slice(origin=thinnest_point, normal="z")
        self.show_slice_and_print_circumference(self.plotter, thinnest_slice, "Наименьшая окружность диафиза",
                                                params_names.diaphysis_smallest_circ)

        self.print_and_save_slice_circumference(middle_slice, "Окружность середины диафиза",
                                                params_names.diaphysis_mid_circ)
        self.print_and_save_slice_circumference(neck_slice, "Окружность шейки", params_names.neck_circ)

        tub_highest_point = selected_points[config.tuberosity_highest_point_name]
        tub_lowest_point = selected_points[config.tuberosity_lowest_point_name]
        tub_length = dist(tub_lowest_point, tub_highest_point)
        print_morf_info("Длина бугристости", tub_length)
        self.params_dict[params_names.tub_length] = tub_length

        tub_lat_point = selected_points[config.tuberosity_lateral_point_name]
        tub_med_point = selected_points[config.tuberosity_medial_point_name]
        tub_width = dist(tub_lat_point, tub_med_point)
        print_morf_info("Ширина бугристости", tub_width)
        self.params_dict[params_names.tub_width] = tub_width

        tub_slice = self.mesh.slice(origin=center_of_tuberosity, normal="z")
        self.print_and_save_slice_sag_width(tub_slice,
                                            "Сагиттальный диаметр проксимального отдела в области бугристости",
                                            params_names.sag_prox_diam_near_tub)

        pit_depth = dist_to_line(upper_pit, (head, opposite_head_point))
        print_morf_info("Глубина суставной ямки", pit_depth)
        self.params_dict[params_names.head_depth] = pit_depth

        head_circumference = circumference(head_width/2)
        print_morf_info("Окружность головки", head_circumference)
        self.params_dict[params_names.head_circ] = head_circumference

        head_tub_slice = slice_through_points(self.mesh, center_of_tuberosity, upper_pit, tub_lowest_point)

        def get_point_height(p):
            return p[2] if head[2] > 0 else -p[2]

        tub_diam_points = find_tub_diam_points_of_slice(head_tub_slice, upper_pit, center_of_tuberosity,
                                                        get_point_height)
        visualize_points(self.plotter, tub_diam_points, color="green")
        visualize_slice(self.plotter, head_tub_slice, color="green")
        head_diam_no_1 = dist(tub_diam_points[0], tub_diam_points[1])
        print_morf_info("Диаметр головки, направленный к бугристости", head_diam_no_1)

        head_slice = get_head_slice(self.mesh, upper_pit)
        visualize_slice(self.plotter, head_slice, color="green")
        nontub_diam_points = find_nontub_diam_points_of_slice(head_slice, upper_pit, tub_diam_points)
        visualize_points(self.plotter, nontub_diam_points, color="green")
        head_diam_no_2 = dist(nontub_diam_points[0], nontub_diam_points[1])
        print_morf_info("Диаметр головки, не напрвленный к бугристости",
                        head_diam_no_2)
        self.params_dict[params_names.head_size_in_two_dirs] = [head_diam_no_1, head_diam_no_2]

        processor = MeshProcessor(self.mesh, self.plotter, tub_diam_points + nontub_diam_points)
        processor.params_dict = self.params_dict
        head_lower_points_names = [
            config.lower_head_point_1,
            config.lower_head_point_2,
            config.lower_head_point_3,
            config.lower_head_point_4
        ]
        PointsSelectionHandler(head_lower_points_names, self.plotter, processor.select_head_points)

    def select_head_points(self, selected_points):
        head_lower_edge_points = [
            selected_points[config.lower_head_point_1],
            selected_points[config.lower_head_point_2],
            selected_points[config.lower_head_point_3],
            selected_points[config.lower_head_point_4]
        ]
        head_heights = find_head_heights(self.head_upper_points, head_lower_edge_points)
        print("Высота головки (в четырех точках):", join_with_slash(head_heights))
        self.params_dict[params_names.head_heights] = head_heights

        request_params_saving(self.params_dict)


def join_with_slash(values):
    return "/".join(map(rounded_string, values)) + " мм"


def find_corresponding_height(upper_point, lower_points):
    distances = map(lambda p: dist(p, upper_point), lower_points)
    return min(distances)


def find_head_heights(upper_points, lower_points):
    heights = []
    for lower_point in lower_points:
        height = find_corresponding_height(lower_point, upper_points)
        heights.append(height)
    return heights


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


def visualize_slice(plotter, slice, color="red"):
    plotter.add_mesh(slice, color=color)


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
    set_up_and_run_plotter(mesh)


main()
