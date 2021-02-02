import argparse
from os.path import isfile

import numpy as np
from joblib import load


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
    out_array = np.squeeze(np.stack(out_list))

    return out_array

def predict(x_test, svm_model, class_labels=("H", "E", "-")):
    assert x_test.shape[1] == svm_model.shape_fit_[1]
    prediction_array = svm_model.predict(x_test)
    prediction_list = [
        class_labels[class_int] for class_int in prediction_array
    ]
    prediction_str = "".join(prediction_list)

    return prediction_str


def write_fasta(filename, header, sequence):
    with open(filename, "w") as handle:
        handle.write(">" + header + "\n")
        handle.write(sequence + "\n")


def process_single_input(profile_file, svm_model, num_residues=20):
    basename = ".".join(profile_file.split(".")[:-2])
    feature_dim = svm_model.shape_fit_[1]
    window_size = int(feature_dim / num_residues)
    assert feature_dim % num_residues == 0 and window_size % 2 == 1

    if not isfile(profile_file):
        print("File not found\t", profile_file)

        return

    assert profile_file.split(".")[-1] == "npy"
    profile = np.load(profile_file)
    x_test = preprocess_profile(profile, window_size)
    prediction = predict(x_test, svm_model)
    fileout_name = basename + ".pred.fa"
    fileout_header = basename
    write_fasta(fileout_name, fileout_header, prediction)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            sequence profiles in .profile.npy format, svm model as joblib dump,
            output is in fasta-like format pred.fa
        """
    )
    parser.add_argument(
        "svm_model",
        type=str,
        help="svm model as joblib dump",
    )
    parser.add_argument(
        "--id_list",
        help="""
            list of input fasta files 
        """,
        action="store_true",
    )

    arguments = parser.parse_args()

    return arguments


def main(input_file, svm_model_file, is_list):

    assert isfile(svm_model_file) and isfile(input_file)
    assert "joblib" in svm_model_file
    svm_model = load(svm_model_file)
    if is_list:
        with open(input_file) as id_list_handle:
            for line in id_list_handle:
                profile_in = line.rstrip() + ".profile.npy"
                process_single_input(profile_in, svm_model)
    else:
        process_single_input(input_file, svm_model)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.svm_model, args.id_list)


