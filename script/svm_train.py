import argparse
from os.path import isfile

import numpy as np
from joblib import dump
from sklearn.svm import SVC


def get_basenames(id_list):
    basenames = []
    with open(id_list) as handle:
        for line in handle:
            element = line.rstrip()

            if len(element) > 0:
                basenames.append(element)

    return tuple(basenames)


def dssp_get_assignments(dssp_file):
    num_seqs = 0
    dssp_assignments = ""
    with open(dssp_file) as handle:
        for line in handle:
            if line[0] == ">":
                num_seqs += 1

                continue
            dssp_assignments = line.rstrip()
    assert num_seqs == 1

    return dssp_assignments


def get_y_array(dssp_assignments, mapped_chars=("H", "E", "-")):
    for element in dssp_assignments:
        assert element in mapped_chars
    mapping = {element: index for index, element in enumerate(mapped_chars)}
    dssp_array = np.array([mapping[element] for element in dssp_assignments])

    return dssp_array


def preprocess_profile(profile, window_size, num_residues=20):
    out_list = []
    central_index = int((window_size - 1) / 2)
    feature_matrix = np.zeros((window_size, num_residues))

    for i in range(central_index):
        if i > len(profile) - 1:
            feature_matrix = np.pad(feature_matrix, ((0, 1), (0, 0)))[1:]

            continue
        feature_matrix[central_index + i + 1] = profile[i]



    for i, _ in enumerate(profile):
        feature_matrix = np.pad(feature_matrix, ((0, 1), (0, 0)))[1:]

        if (i + central_index) < len(profile):
            feature_matrix[-1] = profile[i + central_index]

        out_list.append(feature_matrix.reshape(1, window_size * num_residues))
    assert len(out_list) == len(profile)

    return out_list


def get_training_vectors(basenames, window_size, num_residues=20):
    x_list, y_str = [], ""

    for i, basename in enumerate(basenames):
        print("Loading:", i, "\tname:", basename)
        dssp_file = basename + ".dssp"
        profile_file = basename + ".profile.npy"

        if not isfile(dssp_file) or not isfile(profile_file):
            print("Missing input\t", basename)

            continue
        dssp_assignments = dssp_get_assignments(dssp_file)
        profile = np.load(profile_file)
        assert len(profile) == len(dssp_assignments)
        profile_list = preprocess_profile(profile, window_size)
        assert len(profile_list) == len(profile)
        y_str += dssp_assignments
        x_list += profile_list
    y_array = get_y_array(y_str)
    x_array = np.squeeze(np.stack(x_list))
    assert x_array.shape == (y_array.shape[0], num_residues * window_size)

    return x_array, y_array


def train(x_train, y_train, arguments, kernel="rbf"):
    classifier = SVC(
        kernel=kernel,
        gamma=arguments.gamma,
        C=arguments.c_param,
        verbose=True,
        cache_size=arguments.cache_size,
    )
    classifier.fit(x_train, y_train)

    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            inputs in profile.npy format and dssp
        """
    )
    parser.add_argument("id_list", type=str)
    parser.add_argument(
        "--window_size",
        help="""
            sliding window size
        """,
        default=17,
        type=int,
    )
    parser.add_argument(
        "--gamma",
        help="""
            gamma hyperparameter value
        """,
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--c_param",
        help="""
            C hyperparameter value
        """,
        default=2,
        type=float,
    )

    arguments = parser.parse_args()

    return arguments


def main(arguments):

    assert arguments.window_size % 2 == 1
    assert isfile(arguments.id_list)
    basenames = get_basenames(arguments.id_list)
    x_train, y_train = get_training_vectors(basenames, arguments.window_size)
    trained_svm = train(x_train, y_train, arguments)
    np.save("svm_support_vectors.npy", trained_svm.support_vectors_)
    dump(trained_svm, "svm_model.joblib.xz")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


    
