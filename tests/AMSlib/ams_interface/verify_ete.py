import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


def get_suffix(db_type):
    if db_type == "none":
        return "none"
    if db_type == "hdf5":
        return "h5"
    return "unknown"


def verify_fs_db(db_type: str, fs_path: str, name: str) -> Tuple[bool, bool, str]:
    """
    @brief verifies that the db file exists and is correct in format

    @param db_type The type of the database ('can be  only hdf5')
    @param fs_path: The path of the database
    @param name: The filename prefix of the db

    @return [continue, has_error] Whether we should continue processing the test or stop and whether the formats etc are correct
    db_type:
    """

    if not Path(fs_path).exists():
        print("Expecting output directory to exist")
        return True, 1, None

    suffix = get_suffix(db_type)
    if suffix == "none":
        return False, 0, None
    fn = f"{name}_0.{suffix}"
    fp = Path(f"{fs_path}/{fn}")

    if name == "":
        # We don't have a file and we should return to stop analysis
        return False, 0, None

    if name == "" and fp.exists():
        print(f"I was expecting file({fp}) to not exist")
        fp.unlink()
        return False, 1, None

    return True, 0, str(fp)


def read_file(fp: str, threshold: float) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]:
    """
    @brief verifies that the db file has the appropriate format

    @param fp the path to the file
    @param threshold The threshold of UQ

    @return [input, output], has_error] The data in the file and whether there is some error
    """

    with h5py.File(fp, "r") as fd:
        dsets = fd.keys()
        if threshold == 1.0:
            dsets = fd.keys()
            print(dsets)
            assert len(dsets) == 1, "Expected input, output and domain_name dset"
            return (None, None), 0
        print(dsets)
        if len(dsets) != 3:
            print(f"Expected input, output and domain_name dset")
            return (None, None), 1

        data = {}
        for d in {"input_data", "output_data"}:
            if d not in dsets:
                print(f"Expected {d} to be in dataset names {dests}")
                return (None, None), 1
            data[d] = fd[d]
            print(d, fd[d].shape)
        input_data = np.array(data["input_data"])
        output_data = np.array(data["output_data"])
        print(input_data.shape)
        print(output_data.shape)
        return (input_data, output_data), 0


def verify_data_collection(fs_path, db_type, num_inputs, num_outputs, threshold, name="test", debug_db=False):
    # Returns a tuple of the input/ouput data and 0/1 for correct incorrect file.
    # Checks whether the files also have the right number of columns
    if not Path(fs_path).exists():
        print("Expecting output directory to exist")
        return None, 1

    suffix = get_suffix(db_type)
    if suffix == "none":
        return None, 0

    fn = f"{name}_0.{suffix}"
    if debug_db and db_type != "hdf5":
        print("Debug DB is only supported on hdf5")
        return None, 1
    elif debug_db:
        fn = f"{name}_0.debug.{suffix}"

    fp = Path(f"{fs_path}/{fn}")

    if name == "" and fp.exists():
        print(f"I was expecting file({fp}) to not exist")
        fp.unlink()
        return None, 1
    elif name == "":
        return (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)), 0
    elif not fp.exists():
        print(f"File path {fn} does not exist")
        return None, 1

    if db_type == "hdf5":
        with h5py.File(fp, "r") as fd:
            dsets = fd.keys()
            if threshold == 1.0:
                dsets = fd.keys()
                print(dsets)
                assert len(dsets) == 1, "Expected input, output and domain_name dset"

                return None, 0

            print(dsets)
            assert len(dsets) == 3, "Expected input, output and domain_name dset"
            data = {}
            for d in {"input_data", "output_data"}:
                assert d in dsets, f"Expected {d} to be in dataset names {dests}"
                data[d] = fd[d]
                print(d, fd[d].shape)
            input_data = np.array(data["input_data"])
            output_data = np.array(data["output_data"])
            print(input_data.shape)
            print(output_data.shape)
            fp.unlink()
            if debug_db:
                return (input_data, output_data, predicate), 0
            return (input_data, output_data), 0

    else:
        return None, 1


def verify_data(
    threshold: float, uq_name, num_elements, inputs: np.ndarray, outputs: np.ndarray, num_inputs: int, num_outputs: int
) -> bool:
    """
    @brief verifies that the collected data have the expected values

    @param threshold The threshold of UQ
    @param uq_name The uq name/technology being used
    @param num_elements The number of elements we executed our simulation with
    @param inputs: The data collected as inputs
    @param outputs: The data collected as outputs
    @param num_inputs: The number of inputs on the outer dimension
    @param num_outputs: The number of outputs on the outer dimension

    @return has_error Whether we are correct or not
    """
    if threshold == 0.0:
        # Threshold 0 means collect all data. Verify the sizes.
        assert (
            inputs.shape[0] == num_elements and outputs.shape[0] == num_elements
        ), f"Did not collect all expected data, Input Size: {inputs.shape} : Output Size: {outputs.shape}, expected size: {num_elements}"

    elif threshold == 1.0:
        # Threshold 1.0 means to not collect any data. Verify the sizes.
        assert inputs.shape[0] == 0 and outputs.shape[0] == 0, "Num elements should be zero"
        # There is nothing else we can check here
        return 0
    else:
        # Compute a theoritical range of possible values in the db.
        # The duq/faiss tests have specific settings. The random one can have a
        # bound. This checks for all these cases
        lb = num_elements * (1 - threshold) - num_elements * 0.1
        ub = num_elements * (1 - threshold) + num_elements * 0.1
        assert (
            inputs.shape[0] > lb and inputs.shape[0] < ub
        ), f"Not in the bounds of correct items {lb} {ub} {inputs.shape[0]}"
        assert (
            outputs.shape[0] > lb and outputs.shape[0] < ub
        ), f"Not in the bounds of correct items {lb} {ub} {outputs.shape[0]}"

    if "delta" in uq_name:
        assert "mean" in uq_name or "max" in uq_name, "unknown Delta UQ mechanism"
        d_type = np.float32
        # Our DUQ-mean model skips odd evaluations.
        # Here we set on verify_inputs the inputs of those evaluations
        verify_inputs = np.zeros(inputs.shape, dtype=d_type)
        if threshold == 0.0:
            step = 1
        elif threshold == 0.5:
            verify_inputs[0] = np.ones(num_inputs, dtype=d_type)
            step = 2
        for i in range(1, len(inputs)):
            verify_inputs[i] = verify_inputs[i - 1] + step
        # Compare whether the results match our base function.
        diff_sum = np.sum(np.abs(verify_inputs - inputs))
        assert np.isclose(diff_sum, 0.0), "Mean Input data do not match"
        verify_output = np.sum(inputs, axis=1).T * num_outputs
        outputs = np.sum(outputs, axis=1)
        diff_sum = np.sum(np.abs(outputs - verify_output))
        assert np.isclose(diff_sum, 0.0), "Mean Output data do not match"
    else:
        return 0


def get_fs_data(db_type, fs_path, model_path, threshold, name="test"):
    if db_type != "hdf5" and db_type != "none":
        print(f"Wrong db_type, we support only hdf5 instead got {db_type}")
        return 1

    if model_path == None or model_path == "":
        threshold = 0.0

    cont, has_error, fp = verify_fs_db(db_type, fs_path, name)

    if has_error:
        return (None, None), threshold, 1

    if not cont:
        return (None, None), threshold, 0

    (_in, _out), has_error = read_file(fp, threshold)

    Path(fp).unlink()
    return (_in, _out), threshold, has_error


def from_cli(argv):
    use_device = int(argv[0])
    num_inputs = int(argv[1])
    num_outputs = int(argv[2])
    model_path = argv[3]
    data_type = argv[4]
    uq_name = argv[5]
    threshold = float(argv[6])
    num_iterations = int(argv[7])
    num_elements = int(argv[8])
    db_type = argv[9]
    fs_path = argv[10]

    (_in, _out), thresh, has_error = get_fs_data(db_type, fs_path, model_path, threshold)

    if has_error:
        return 1

    if (_in is None) and (_out is None):
        return 0

    if _in is None:
        print("In is None, Out is not")
        return 1

    if _out is None:
        print("Out is None, In is not")
        return 1

    error = verify_data(thresh, uq_name, num_elements, _in, _out, num_inputs, num_outputs)
    return error


def get_rmq_data(ams_config, domain_names, num_iterations, timeout=1):
    from ams.rmq import BlockingClient, default_ams_callback

    rmq_json = ams_config["db"]["rmq_config"]
    print(rmq_json)
    host = rmq_json["service-host"]
    vhost = rmq_json["rabbitmq-vhost"]
    port = rmq_json["service-port"]
    user = rmq_json["rabbitmq-user"]
    password = rmq_json["rabbitmq-password"]
    queue = rmq_json["rabbitmq-queue-physics"]
    cert = None
    if "rabbitmq-cert" in rmq_json:
        cert = rmq_json["rabbitmq-cert"]
        cert = None if cert == "" else cert
    with BlockingClient(host, port, vhost, user, password, cert, default_ams_callback) as client:
        with client.connect(queue) as channel:
            msgs = channel.receive(n_msg=num_iterations * len(domain_names), timeout=timeout)

    dns = set(domain_names)

    _data = {k: ([], []) for k in dns}

    # sim_data[] = (_in, _out, thresh, uq_type)
    for msg in msgs:
        domain, input_data, output_data = msg.decode()
        print(domain)
        print(msg)
        if domain not in dns:
            raise RuntimeError(f"Received unknown domain name {domain}")
        _data[domain][0].append(input_data)
        _data[domain][1].append(output_data)

    sim_data = {}
    for d in _data.keys():
        ml_id = ams_config["domain_models"][d]
        model = ams_config["ml_models"][ml_id]
        threshold = model["threshold"]
        model_path = model.get("model_path", None)

        if model_path == None or model_path == "":
            threshold = 0.0

        uq_type = "unknown"
        if "random" in ml_id:
            uq_type = "random"
        elif "uq" in ml_id:
            if "mean" in ml_id:
                uq_type = "deltaUQ(mean)"
            elif "max" in ml_id:
                uq_type = "deltaUQ(max)"

        print("Type for domain", d, type(_data[d]), len(_data[d]))
        store_data = model.get("store", True)
        if store_data == False and len(_data[d][0]) != 0:
            raise RuntimeError("Expected data to not exist")
        elif store_data:
            inputs = np.vstack(_data[d][0])
            outputs = np.vstack(_data[d][1])
            sim_data[d] = (inputs, outputs, threshold, uq_type)

    return sim_data


def from_json(argv):
    print(argv)
    use_device = int(argv[0])
    num_inputs = int(argv[1])
    num_outputs = int(argv[2])
    data_type = argv[3]
    num_elements = int(argv[4])
    model_1 = argv[5]
    model_2 = argv[6]

    env_file = Path(os.environ["AMS_OBJECTS"])
    if not env_file.exists():
        print("Environment file does not exist")
        return -1

    with open(env_file, "r") as fd:
        data = json.load(fd)

    db_type = data["db"]["dbType"]

    sim_data = {}
    if db_type == "hdf5" or db_type == "none":
        db_path = data["db"]["fs_path"]
        for m in [model_1, model_2]:
            print("Reading Model", m)
            ml_id = data["domain_models"][m]
            model = data["ml_models"][ml_id]
            store_data = model.get("store", True)

            threshold = model["threshold"]
            model_path = model.get("model_path", None)

            (_in, _out), thresh, has_error = get_fs_data(
                db_type, db_path, model_path, threshold, m if store_data else ""
            )

            if has_error:
                return 1

            if (_in is None) and (_out is None):
                # This means we returned 0 as an error and we don't have any data
                # to analyze, so we skip
                continue

            if _in is None:
                print("In is None, Out is not")
                return 1

            if _out is None:
                print("Out is None, In is not")
                return 1

            uq_type = "unknown"
            if "random" in m:
                uq_type = "random"
            elif "uq" in m:
                if "mean" in m:
                    uq_type = "deltaUQ(mean)"
                elif "max" in m:
                    uq_type = "deltaUQ(max)"

            print("Uq type is ", uq_type)
            sim_data[m] = (_in, _out, thresh, uq_type)
    elif db_type == "rmq":
        print("RMQ")
        sim_data = get_rmq_data(data, [model_1, model_2], 1)

    for m, (_in, _out, thresh, uq_type) in sim_data.items():
        print("Verify data of Model", m)
        error = verify_data(thresh, uq_type, num_elements, _in, _out, num_inputs, num_outputs)
        if error:
            print("Error when verify_data")
            return error
    return 0


if __name__ == "__main__":
    if "AMS_OBJECTS" in os.environ:
        sys.exit(from_json(sys.argv[1:]))
    sys.exit(from_cli(sys.argv[1:]))
