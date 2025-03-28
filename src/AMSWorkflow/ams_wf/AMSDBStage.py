# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import time
import sys

from pathlib import Path

from ams.loader import load_class
from ams.stage import get_pipeline


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="AMS Stage mechanism. The mechanism moves data to the file-system and optionally registers them in a Kosh store",
    )

    parser.add_argument(
        "--load",
        "-l",
        dest="user_module",
        help="Path implementing a custom pipeline stage module",
        default=None,
    )
    parser.add_argument(
        "--class",
        dest="user_class",
        help="Class implementing the 'Action' performed on data",
        default=None,
    )
    parser.add_argument(
        "--policy",
        "-p",
        help="The execution vehicle of every stage in the pipeline",
        choices=["process", "thread", "sequential"],
        default="process",
    )

    parser.add_argument(
        "--json-monitoring",
        "-jm",
        dest="output_json",
        help="Prefix for file to output the monitoring data from the stage (JSON), the prefix will be extended with -<PID>-<hostname>.json",
        default=None,
    )

    parser.add_argument(
        "--mechanism", "-m", dest="mechanism", choices=["fs", "network"], default="fs"
    )

    args, extras = parser.parse_known_args()

    if (args.user_module is not None) and args.user_class is None:
        raise argparse.ArgumentTypeError(
            "User custom module was specified but the 'class' was not defined"
        )

    user_class = None
    user_args = None
    user_prog = ""
    print(f"User class is {args.user_module} {args.user_class}")

    if args.user_module is not None:
        user_class = load_class(args.user_module, args.user_class)
        user_prog = user_class.__name__
        user_parser = argparse.ArgumentParser(
            prog=user_prog,
            description="User provided class to prune data before storing into candidate database",
            epilog=f"Executed in conjustion with {parser.prog}",
        )
        user_class.add_cli_args(user_parser)
        user_args, extras = user_parser.parse_known_args(extras)

    pipeline_cls = get_pipeline(args.mechanism)
    print(pipeline_cls)
    pipeline_parser = argparse.ArgumentParser(
        prog=pipeline_cls.__name__,
        description="Pipeline mechanism to load data from specified end-point",
        epilog=f"Executed in conjustion with {parser.prog} and {user_prog}",
    )

    pipeline_cls.add_cli_args(pipeline_parser)
    pipeline_args, extras = pipeline_parser.parse_known_args(extras)

    if len(extras) != 0:
        options = " ".join(extras)
        raise argparse.ArgumentTypeError(f"Could not parse the {options}")

    pipeline = pipeline_cls.from_cli(pipeline_args)

    if user_class is not None:
        obj = user_class.from_cli(user_args)
        pipeline.add_user_action(obj)

    output_json = args.output_json
    if output_json is not None:
        import socket
        import os
        from ams.monitor import AMSMonitor

        hostname = socket.gethostname()
        pid = os.getpid()
        output_json = Path(output_json).stem
        output_json = f"{output_json}-{hostname}-{pid}.json"
        print(f"Monitoring output file: {output_json}")

    start = time.time()
    pipeline.execute(args.policy)
    end = time.time()
    print(f"End to End time spend : {end - start}")

    if output_json is not None:
        # Output profiling output to JSON
        AMSMonitor.json(output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
