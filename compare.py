import config
import pickle
from params_names import head_size_in_two_dirs, head_heights


def read_saved_dict(file_name):
    with open(file_name, "rb") as f:
        d = pickle.load(f)
        return d


def abs_error(abs_val, measured_val):
    return measured_val - abs_val


def rel_error(abs_val, measured_val):
    return abs(abs_error(abs_val, measured_val) / abs_val)


def float_val_to_percent_string(val):
    return "{:.2f}%".format(val*100)


def main():
    true_params = read_saved_dict(config.path_to_true_params)
    prog_params = read_saved_dict(config.path_to_programmatically_measured_params)

    print("Относительные погрешности:")
    for param_name in true_params.keys():
        abs_val = true_params[param_name]
        measured_val = prog_params[param_name]
        error_str = ""
        if param_name in [head_size_in_two_dirs, head_heights]:
            errors = [rel_error(av, mv) for av, mv in zip(abs_val, measured_val)]
            error_str = "/".join(map(float_val_to_percent_string, errors))
        else:
            error_str = float_val_to_percent_string(rel_error(abs_val, measured_val))
        print(param_name, ": ", error_str, sep="")


main()
