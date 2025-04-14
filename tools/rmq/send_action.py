#!/usr/bin/env python3
# Copyright 2021-2025 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pika
import sys
import ssl
import json
import argparse
import time

from pika.exchange_type import ExchangeType
import logging

logging.basicConfig(level=logging.WARN)  # or INFO


def get_rmq_connection(json_file):
    data = {}
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def main(args):
    conn = get_rmq_connection(args.creds)
    cacert = conn.get("rabbitmq-cert", None)
    if cacert is None:
        ssl_options = None
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False
        context.load_verify_locations(cacert)
        ssl_options = pika.SSLOptions(context)

    print(f"Connecting to {conn['service-host']} ...")

    credentials = pika.PlainCredentials(conn["rabbitmq-user"], conn["rabbitmq-password"])
    cp = pika.ConnectionParameters(
        host=conn["service-host"],
        port=conn["service-port"],
        virtual_host=conn["rabbitmq-vhost"],
        credentials=credentials,
        ssl_options=ssl_options,
    )

    routing_key = ""
    retries = 0
    max_retries = 5
    while True:
        try:
            connection = pika.BlockingConnection(cp)
            channel = connection.channel()
            print("Connection successful!")
            break
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            retries += 1
            if retries >= max_retries:
                print("Max retries reached. Exiting.")
                raise
        time.sleep(1)

    # Turn on delivery confirmations
    channel.confirm_delivery()

    channel.exchange_declare(exchange=args.exchange, exchange_type=ExchangeType.fanout, auto_delete=False)

    result = channel.queue_declare(queue="", exclusive=False)

    msg = {"request_type": str(args.action)}

    queue_name = result.method.queue
    try:
        channel.basic_publish(exchange=args.exchange, routing_key=routing_key, body=json.dumps(msg))
        print(f"Sent {msg} on exchange='{args.exchange}'/routing_key='{routing_key}'")
    except pika.exceptions.UnroutableError:
        print(f"Message could not be confirmed")
    connection.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool that sends actions to the AMS Stager (terminate, done-training etc)"
    )
    parser.add_argument("-c", "--creds", help="Credentials file (JSON)", required=True)
    parser.add_argument(
        "-e",
        "--exchange",
        help="On which exchange to send actions (default = control-panel)",
        default="control-panel",
        required=False,
    )
    parser.add_argument("-a", "--action", help="action to send to the stager", required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
