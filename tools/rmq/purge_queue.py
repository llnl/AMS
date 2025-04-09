#!/usr/bin/env python3
# Copyright 2021-2025 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import pika
import ssl
import sys

# CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
#   openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null
#   2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > rabbitmq-credentials.cacert

def get_rmq_connection(json_file):
    data = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def main(credentials: str, cacert: str, queue: str):
    try:
        conn = get_rmq_connection(credentials)
        if cacert is None:
            ssl_options = None
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = False
            context.load_verify_locations(cacert)
            ssl_options = pika.SSLOptions(context)

        credentials = pika.PlainCredentials(conn["rabbitmq-user"], conn["rabbitmq-password"])
        cp = pika.ConnectionParameters(
            host=conn["service-host"],
            port=conn["service-port"],
            virtual_host=conn["rabbitmq-vhost"],
            credentials=credentials,
            ssl_options=ssl_options
        )

        connection = pika.BlockingConnection(cp)
        channel = connection.channel()

        print(f"Connecting to {conn['service-host']} ...")

        queues = channel.queue_declare(queue=queue)
        method_frame = channel.queue_purge(queue=queue)
        print(f"Purged {queue} => {method_frame}")
        connection.close()
    except KeyboardInterrupt:
        print("")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Tools that consumes AMS-encoded messages from RabbitMQ queue")
    parser.add_argument('-c', '--creds', help="Credentials file (JSON)", required=True)
    parser.add_argument('-t', '--tls-cert', help="TLS certificate file", required=False)
    parser.add_argument('-q', '--queue', help="Queue to purge and delete", required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(
        credentials = args.creds,
        cacert = args.tls_cert,
        queue = args.queue
    )
