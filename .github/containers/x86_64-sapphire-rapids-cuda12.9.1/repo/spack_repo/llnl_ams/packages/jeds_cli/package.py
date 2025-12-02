# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class JedsCli(PythonPackage):
    """Package for JEDS, Livermore Computing WEG job ephemeral data services"""

    homepage = "https://wci-repo.llnl.gov/repository/pypi-group/packages/jeds-cli/"
    url      = "https://wci-repo.llnl.gov/repository/pypi-group/packages/jeds-cli/0.0.5/jeds_cli-0.0.5-py3-none-any.whl"

    version("0.0.5", sha256="7ced334b648297904f0508334697de0cb3a361a13c0a2e5d7fa7a4f7c3086294", expand=False)

    depends_on("py-setuptools", type="build")
    depends_on("py-certipy@0.1.3")
    
    def setup_run_environment(self, env):
        # Needed so we can run with without legacy algorithms
        env.set("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

