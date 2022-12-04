import pickle
import params_names
from config import path_to_true_params


def to_float_list(str):
    return list(map(float, str.split("/")))


def is_list_param(param_name):
    return param_name == params_names.head_size_in_two_dirs or param_name == params_names.head_heights


def main():
    params_dict = dict()

    for param_name in params_names.params_names:
        print(param_name, "=", end="")
        param_str = input().replace(",", ".")
        param_value = to_float_list(param_str) if is_list_param(param_name) else float(param_str)
        params_dict[param_name] = param_value

    with open(path_to_true_params, "wb") as f:
        pickle.dump(params_dict, f)

    print("Saved dict:")
    print(params_dict)


main()
