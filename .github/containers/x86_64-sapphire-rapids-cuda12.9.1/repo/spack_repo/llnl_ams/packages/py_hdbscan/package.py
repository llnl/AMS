# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class PyHdbscan(PythonPackage):
    """HDBSCAN - Hierarchical Density-Based Spatial Clustering of
    Applications with Noise. Performs DBSCAN over varying epsilon
    values and integrates the result to find a clustering that gives
    the best stability over epsilon. This allows HDBSCAN to find
    clusters of varying densities (unlike DBSCAN), and be more robust
    to parameter selection. In practice this means that HDBSCAN
    returns a good clustering straight away with little or no
    parameter tuning -- and the primary parameter, minimum cluster
    size, is intuitive and easy to select.  HDBSCAN is ideal for
    exploratory data analysis; it's a fast and robust algorithm that
    you can trust to return meaningful clusters (if there are any)."""

    homepage = "https://github.com/scikit-learn-contrib/hdbscan"
    git = "https://github.com/scikit-learn-contrib/hdbscan.git"

    license("BSD-3-Clause")

    version("0.8.40", tag="release-0.8.40", commit="f0285287a62084e3a796f3a34901181972966b72")
    version("0.8.37", tag="release-0.8.37", commit="c5fcf4b3829d391eadd14598736a763952790a82")

    depends_on("py-setuptools", type="build")
    depends_on("py-cython@0.27:", type=("build", "run"))
    depends_on("py-numpy@1.16.0:", type=("build", "run"))
    depends_on("py-numpy@1.20:", type=("build", "run"), when="@0.8.29:")
    depends_on("py-scipy@0.9:", type=("build", "run"))
    depends_on("py-scipy@1.0:", type=("build", "run"), when="@0.8.29:")
    depends_on("py-scikit-learn@0.17:", type=("build", "run"))
    depends_on("py-scikit-learn@0.20:", type=("build", "run"), when="@0.8.29:")
    depends_on("py-joblib", type=("build", "run"))
    depends_on("py-joblib@1.0:", type=("build", "run"), when="@0.8.29:")
    depends_on("py-six", type=("build", "run"))
