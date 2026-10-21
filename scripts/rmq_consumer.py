#!/usr/bin/env python3
"""
Simple RabbitMQ consumer for testing AMS message queue.

Usage:
  python rmq_consumer.py --config rmq-conf/pds.json --queue ams-physics
  python rmq_consumer.py  # Uses AMS_OBJECTS env var and default queue
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
from pathlib import Path

try:
    import pika
except ImportError:
    print("ERROR: pika library not found. Install with: pip install pika", file=sys.stderr)
    sys.exit(1)


def load_config(config_path):
    """Load RabbitMQ configuration from JSON file.

    Supports two formats:
    1. Legacy format: config fields at top level
    2. AMS format: config nested under db.rmq_config
    """
    with open(config_path) as f:
        data = json.load(f)

    # Check if this is the new AMS format with db.rmq_config
    if "db" in data and "rmq_config" in data["db"]:
        return data["db"]["rmq_config"]

    # Legacy format - config at top level
    return data


def generate_tls_cert(config, cert_path):
    """Generate TLS certificate from RabbitMQ server for PDS connection."""
    remote_host = config["service-host"]
    remote_port = config["service-port"]

    print(f"Generating TLS certificate from {remote_host}:{remote_port}...")

    # Use openssl to fetch the certificate
    cmd = (
        f"openssl s_client -connect {remote_host}:{remote_port} -showcerts < /dev/null 2>/dev/null | "
        f"sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p'"
    )

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10, check=False)

        if result.returncode != 0 or not result.stdout.strip():
            print("ERROR: Failed to fetch certificate from server", file=sys.stderr)
            print(f"Command: {cmd}", file=sys.stderr)
            return False

        # Save certificate to file
        with open(cert_path, "w") as f:
            f.write(result.stdout)

        print(f"✓ Certificate saved to: {cert_path}")
        return True

    except subprocess.TimeoutExpired:
        print("ERROR: Timeout while fetching certificate", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Failed to generate certificate: {e}", file=sys.stderr)
        return False


def connect_rabbitmq(config, cert_path=None, use_ssl=True):
    """Create RabbitMQ connection from config."""
    credentials = pika.PlainCredentials(config["rabbitmq-user"], config["rabbitmq-password"])

    # SSL context for secure connections
    ssl_options = None
    if use_ssl:
        ssl_context = ssl.create_default_context()

        # Load the PDS certificate if provided
        if cert_path and Path(cert_path).exists():
            print(f"Using TLS certificate: {cert_path}")
            ssl_context.load_verify_locations(cafile=cert_path)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        else:
            # Fallback to no verification
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        ssl_options = pika.SSLOptions(ssl_context)

    parameters = pika.ConnectionParameters(
        host=config["service-host"],
        port=config["service-port"],
        virtual_host=config["rabbitmq-vhost"],
        credentials=credentials,
        ssl_options=ssl_options,
        # Add timeouts for better error handling
        connection_attempts=3,
        retry_delay=2,
        socket_timeout=10,
        heartbeat=60,
        blocked_connection_timeout=300,
    )

    return pika.BlockingConnection(parameters)


def decode_ams_message_header(data):
    """Decode AMS message header (first 12 bytes)."""
    if len(data) < 12:
        return None

    # Header structure from basedb.hpp:
    # uint8_t header_size (1 byte)
    # uint8_t domain_name_length (1 byte)
    # uint16_t num_inputs (2 bytes)
    # uint16_t num_outputs (2 bytes)
    # uint32_t total_size (4 bytes)
    # uint16_t metadata_size (2 bytes)

    header = {
        "header_size": data[0],
        "domain_name_length": data[1],
        "num_inputs": int.from_bytes(data[2:4], byteorder="little"),
        "num_outputs": int.from_bytes(data[4:6], byteorder="little"),
        "total_size": int.from_bytes(data[6:10], byteorder="little"),
        "metadata_size": int.from_bytes(data[10:12], byteorder="little"),
    }

    return header


def list_queues(channel):
    """List available queues (requires management permissions)."""
    print("\nNote: Queue listing requires RabbitMQ management API permissions.")
    print("If you don't have permissions, you'll need to know the queue name in advance.")


def open_consumer_queue(connection, queue_name):
    """Open a non-durable queue, replacing a conflicting durable queue."""
    channel = connection.channel()

    try:
        queue_state = channel.queue_declare(queue=queue_name, durable=False)
    except pika.exceptions.ChannelClosedByBroker as error:
        durable_mismatch = error.reply_code == 406 and "inequivalent arg 'durable'" in error.reply_text
        if not durable_mismatch:
            raise

        print(f"WARNING: Replacing existing durable queue '{queue_name}' with a non-durable queue.", file=sys.stderr)

        # RabbitMQ closes the channel after a declaration precondition failure.
        # Use a fresh channel to remove the incompatible queue and declare it
        # with the same durability setting used by AMS.
        channel = connection.channel()
        channel.queue_delete(queue=queue_name)
        queue_state = channel.queue_declare(queue=queue_name, durable=False)

    return channel, queue_state


def consume_messages(connection, queue_name, max_messages=None, wait=False):
    """Consume messages from queue."""
    print(f"\nConsuming from queue: {queue_name}")

    # Connect to AMS's non-durable queue, or create it when this script starts first.
    try:
        channel, queue_state = open_consumer_queue(connection, queue_name)
        msg_count = queue_state.method.message_count
        print(f"✓ Queue ready with {msg_count} message(s)")
    except Exception as e:
        print(f"ERROR: Failed to access/create queue: {e}", file=sys.stderr)
        return

    if msg_count == 0 and not wait:
        print("Queue is empty (no messages to consume).")
        print("Use --wait to keep connection alive and wait for messages.")
        return

    if wait:
        print("Waiting for messages (Ctrl+C to stop)...")
        consume_messages_blocking(channel, queue_name, max_messages)
    else:
        consume_messages_polling(channel, queue_name, max_messages)


def consume_messages_polling(channel, queue_name, max_messages=None):
    """Consume existing messages using polling (basic_get)."""
    consumed = 0
    while max_messages is None or consumed < max_messages:
        method, _properties, body = channel.basic_get(queue=queue_name, auto_ack=True)

        if method is None:
            print("No more messages.")
            break

        consumed += 1
        print(f"\n--- Message {consumed} ---")
        print_message_info(body, method.delivery_tag if hasattr(method, "delivery_tag") else None)

    print(f"\n✓ Consumed {consumed} message(s)")


def consume_messages_blocking(channel, queue_name, max_messages=None):
    """Consume messages using blocking consumer (basic_consume)."""
    consumed = [0]  # Use list to allow modification in nested function

    def callback(ch, method, properties, body):
        consumed[0] += 1
        print(f"\n--- Message {consumed[0]} ---")
        print_message_info(body, method.delivery_tag)

        # Auto-ack
        ch.basic_ack(delivery_tag=method.delivery_tag)

        # Stop if reached max
        if max_messages is not None and consumed[0] >= max_messages:
            print(f"\n✓ Reached max messages ({max_messages}), stopping...")
            ch.stop_consuming()

    try:
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
        channel.start_consuming()
    except KeyboardInterrupt:
        print(f"\n\n✓ Stopped by user. Consumed {consumed[0]} message(s)")
        channel.stop_consuming()


def print_message_info(body, delivery_tag=None):
    """Print information about a message."""
    print(f"Message size: {len(body)} bytes")
    if delivery_tag:
        print(f"Delivery tag: {delivery_tag}")

    # Try to decode AMS message header
    header = decode_ams_message_header(body)
    if header:
        print("AMS Header:")
        print(f"  Header size: {header['header_size']}")
        print(f"  Domain name length: {header['domain_name_length']}")
        print(f"  Number of inputs: {header['num_inputs']}")
        print(f"  Number of outputs: {header['num_outputs']}")
        print(f"  Total size: {header['total_size']}")
        print(f"  Metadata size: {header['metadata_size']}")

        # Extract domain name
        offset = header["header_size"]
        domain_name_end = offset + header["domain_name_length"]
        if domain_name_end <= len(body):
            domain_name = body[offset:domain_name_end].decode("utf-8", errors="replace")
            print(f"  Domain name: '{domain_name}'")
    else:
        print("Could not decode AMS header (message too short)")


def main():
    parser = argparse.ArgumentParser(description="Consume messages from RabbitMQ queue for AMS testing")
    parser.add_argument(
        "--config", "-c", help="Path to RabbitMQ config JSON (default: $AMS_OBJECTS or rmq-conf/pds.json)"
    )
    parser.add_argument("--queue", "-q", help='Queue name (default: uses "rabbitmq-queue-physics" from config)')
    parser.add_argument("--max", "-m", type=int, help="Maximum number of messages to consume (default: all)")
    parser.add_argument(
        "--wait", "-w", action="store_true", help="Wait for new messages (blocking mode). Press Ctrl+C to stop."
    )
    parser.add_argument("--no-ssl", action="store_true", help="Disable SSL (try if connection fails with SSL)")
    parser.add_argument(
        "--test-connection", action="store_true", help="Test connection and exit (does not consume messages)"
    )
    parser.add_argument("--cert", help="Path to TLS certificate file (default: auto-generate pds.crt next to config)")
    parser.add_argument(
        "--regenerate-cert", action="store_true", help="Force regeneration of TLS certificate even if it exists"
    )

    args = parser.parse_args()

    # Determine config path
    config_path = args.config
    if not config_path:
        config_path = os.environ.get("AMS_OBJECTS")
    if not config_path:
        config_path = "rmq-conf/pds.json"

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Determine queue name
    queue_name = args.queue
    if not queue_name:
        queue_name = config.get("rabbitmq-queue-physics")
    if not queue_name:
        print("ERROR: No queue specified and 'rabbitmq-queue-physics' not found in config", file=sys.stderr)
        sys.exit(1)

    # Handle TLS certificate for PDS
    use_ssl = not args.no_ssl
    cert_path = None

    if use_ssl:
        # Priority 1: Use cert from config if specified
        if config.get("rabbitmq-cert"):
            cert_path = Path(config["rabbitmq-cert"])
            if cert_path.exists():
                print(f"Using certificate from config: {cert_path}")
            else:
                print(f"WARNING: Certificate specified in config not found: {cert_path}", file=sys.stderr)
                cert_path = None
        # Priority 2: Use --cert flag
        elif args.cert:
            cert_path = Path(args.cert)
            if cert_path.exists():
                print(f"Using certificate from --cert flag: {cert_path}")
            else:
                print(f"WARNING: Certificate specified via --cert not found: {cert_path}", file=sys.stderr)
                cert_path = None
        # Priority 3: Auto-generate cert next to config file
        else:
            cert_path = config_path.parent / "pds.crt"
            # Generate or regenerate certificate if needed
            if args.regenerate_cert or not cert_path.exists():
                if not generate_tls_cert(config, cert_path):
                    print(
                        "WARNING: Failed to generate certificate, continuing without cert verification", file=sys.stderr
                    )
                    cert_path = None
            else:
                print(f"Using existing certificate: {cert_path}")

    # Connect to RabbitMQ
    ssl_status = "with SSL" if use_ssl else "without SSL"
    print(f"\nConnecting to RabbitMQ at {config['service-host']}:{config['service-port']} {ssl_status}")
    print(f"Virtual host: {config['rabbitmq-vhost']}")
    print(f"User: {config['rabbitmq-user']}")

    try:
        connection = connect_rabbitmq(config, cert_path=cert_path, use_ssl=use_ssl)
        print("✓ Connected successfully!")

        if args.test_connection:
            print("✓ Connection test passed")
            connection.close()
            sys.exit(0)

    except Exception as e:
        print(f"\n✗ ERROR: Failed to connect to RabbitMQ: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        if use_ssl:
            print("  - Try regenerating certificate with --regenerate-cert", file=sys.stderr)
            print("  - Try with --no-ssl flag if server doesn't require SSL", file=sys.stderr)
        else:
            print("  - Try without --no-ssl flag if server requires SSL", file=sys.stderr)
        print("  - Check firewall/network connectivity to the host", file=sys.stderr)
        print("  - Verify credentials in config file", file=sys.stderr)
        print("  - Check if RabbitMQ service is running", file=sys.stderr)
        sys.exit(1)

    try:
        consume_messages(connection, queue_name, max_messages=args.max, wait=args.wait)
    finally:
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
