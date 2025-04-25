#!/usr/bin/env bash
set -e

# 1) ensure the socket directory exists and is owned by mysql
mkdir -p /run/mysqld
chown mysql:mysql /run/mysqld

# 2) start MariaDB directly
#    this will background itself (via mysqld_safe)
exec /usr/bin/mysqld_safe --datadir=/var/lib/mysql &

# 3) wait until it's up
while ! mysqladmin ping -uroot --silent; do
  sleep 1
done
echo "MariaDB is up!"

# 4) start RabbitMQ in detached mode
rabbitmq-server -detached

# 5) drop into a shell (or run passed-in command)
if [ $# -gt 0 ]; then
  exec "$@"
else
  exec bash
fi

