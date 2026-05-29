#!/usr/bin/env python3

# Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import contextlib
from pathlib import Path

import h5py
import numpy as np

NUM_INPUTS = 3
NUM_INOUTS = 3
NUM_OUTPUTS = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Validate no-Torch AMS HDF5 end-to-end output.")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--num-elements", type=int, required=True)
    args = parser.parse_args()
    if args.num_elements <= 0:
        parser.error("--num-elements must be positive")
    return args


def expected_data(workload, dtype, num_elements):
    rows = np.arange(num_elements, dtype=dtype).reshape(-1, 1)

    if workload == "ordinary":
        inputs = np.repeat(rows, NUM_INPUTS, axis=1)
        outputs = np.repeat(rows * NUM_INPUTS, NUM_OUTPUTS, axis=1)
    elif workload == "inout":
        input_width = NUM_INPUTS - 1 + NUM_INOUTS
        output_width = NUM_OUTPUTS + NUM_INOUTS
        inputs = np.repeat(rows, input_width, axis=1)
        outputs = np.repeat(rows * input_width, output_width, axis=1)
    elif workload == "broadcast":
        broadcast = np.full((num_elements, 1), NUM_INPUTS - 1, dtype=dtype)
        inputs = np.concatenate([np.repeat(rows, NUM_INPUTS - 1, axis=1), broadcast], axis=1)
        outputs = np.repeat(rows * (NUM_INPUTS - 1) + broadcast, NUM_OUTPUTS, axis=1)
    else:
        raise ValueError(f"Unknown workload: {workload}")

    return inputs, outputs


def validate_dataset(dataset, expected, name, file_path):
    if dataset.dtype != expected.dtype:
        raise AssertionError(f"{file_path}: {name} dtype is {dataset.dtype}, expected {expected.dtype}")
    if dataset.shape != expected.shape:
        raise AssertionError(f"{file_path}: {name} shape is {dataset.shape}, expected {expected.shape}")
    np.testing.assert_array_equal(dataset[...], expected, err_msg=f"{file_path}: incorrect {name} values")


def validate_file(file_path, domain, workload, dtype, num_elements):
    if not file_path.is_file():
        raise AssertionError(f"Missing expected HDF5 file: {file_path}")

    expected_inputs, expected_outputs = expected_data(workload, dtype, num_elements)
    with h5py.File(file_path, "r") as database:
        domain_dataset = database["domain_name"]
        expected_domain_shape = (len(domain),)
        if domain_dataset.shape != expected_domain_shape:
            raise AssertionError(
                f"{file_path}: domain_name shape is {domain_dataset.shape}, expected {expected_domain_shape}"
            )
        if domain_dataset.dtype.itemsize != 1:
            raise AssertionError(
                f"{file_path}: domain_name dtype is {domain_dataset.dtype}, expected one-byte characters"
            )
        actual_domain = np.asarray(domain_dataset[...]).tobytes().decode()
        if actual_domain != domain:
            raise AssertionError(f"{file_path}: domain_name is {actual_domain!r}, expected {domain!r}")

        validate_dataset(database["input_data"], expected_inputs, "input_data", file_path)
        validate_dataset(database["output_data"], expected_outputs, "output_data", file_path)

        if workload == "inout":
            np.testing.assert_array_equal(
                database["input_data"][:, NUM_INPUTS - 1 :],
                expected_inputs[:, NUM_INPUTS - 1 :],
                err_msg=f"{file_path}: original inout values are misplaced",
            )
            np.testing.assert_array_equal(
                database["output_data"][:, NUM_OUTPUTS:],
                expected_outputs[:, NUM_OUTPUTS:],
                err_msg=f"{file_path}: updated inout values are misplaced",
            )
        elif workload == "broadcast":
            np.testing.assert_array_equal(
                database["input_data"][:, -1],
                expected_inputs[:, -1],
                err_msg=f"{file_path}: broadcast input was not materialized",
            )


def main():
    args = parse_args()
    generated_files = []
    for type_name, dtype in (("float", np.float32), ("double", np.float64)):
        for workload in ("ordinary", "inout", "broadcast"):
            domain = f"no_torch_{type_name}_{workload}"
            file_path = args.db_path / f"{domain}_0.h5"
            validate_file(file_path, domain, workload, dtype, args.num_elements)
            generated_files.append(file_path)

    for file_path in generated_files:
        file_path.unlink()
    with contextlib.suppress(OSError):
        args.db_path.rmdir()


if __name__ == "__main__":
    main()
