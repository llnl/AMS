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

: "${MYSQL_ROOT_PASSWORD:=root}"     # default, if not passed-in
mysql -uroot <<-EOSQL
  ALTER USER 'root'@'localhost' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD}';
  FLUSH PRIVILEGES;
EOSQL
echo "Root password set to '${MYSQL_ROOT_PASSWORD}'"

# 4) start RabbitMQ in detached mode
rabbitmq-server -detached

# 5) Load the python venv
source /app/venv/bin/activate

# 6) drop into a shell (or run passed-in command)
if [ $# -gt 0 ]; then
  exec "$@"
else
  exec bash
fi

