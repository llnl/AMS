FROM fluxrm/flux-core:latest
USER root

# Install MariaDB + tools
# Install extra packages (e.g., git, python3-pip)
RUN apt-get update && \
    apt-get install -y git python3 python3-pip mariadb-server mariadb-client python3-venv libjson-glib-dev && \
    apt-get clean

RUN echo "[mysqld]\nbind-address = 127.0.0.1" > /etc/mysql/mariadb.conf.d/99-local.cnf

# Set a working directory
WORKDIR /workspace

COPY init.sql /init.sql

RUN chown -R mysql:mysql /var/lib/mysql && \
    mysqld_safe --datadir=/var/lib/mysql & \
    sleep 5 && \
    until mysqladmin ping --silent; do sleep 1; done && \
    mariadb -u root < /init.sql && \
    mariadb -u root -psecret -e 'SHOW DATABASES;'

# Detect GCC version
RUN gcc --version | head -n1 > /gcc-version.txt
# Detect CPU architecture
RUN uname -m > /arch.txt

# Create virtualenv and activate in shell
RUN python3 -m venv /venv --system-site-packages

# Set an entrypoint or default command (optional)
CMD ["bash", "-c", "\
  mysqld_safe --datadir=/var/lib/mysql & \
  until mysqladmin ping -h 127.0.0.1 --silent; do sleep 1; done && \
  echo 'MariaDB ready. Container is idle.' && \
  tail -f /dev/null"]
