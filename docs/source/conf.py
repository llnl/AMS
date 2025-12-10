# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
project = 'AMS'
copyright = '2023-2025, Lawrence Livermore National Security, LLC'
author = 'AMS Development Team'

# The version info for the project
# Read from VERSION file or git
version = '0.1'
# release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'breathe',  # For C++ documentation
    'exhale', # For C++ search + integration
    'myst_parser',  # For Markdown support
    'sphinx_copybutton',  # Add copy button to code blocks
    "sphinx_design" # Furo theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# html_theme = 'sphinx_rtd_theme'
html_theme = "furo"
html_static_path = ['_static']


# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"
highlight_language = "cpp"

html_context = {
    "display_github": True,
    "github_user": "LLNL",
    "github_repo": "AMS",
    "github_version": "develop",
    "conf_py_path": "/docs/source/",
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for Breathe (C++ documentation) ---------------------------------
breathe_projects = {
    "AMS": "../build/xml"
}
breathe_default_project = "AMS"

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Setup the exhale extension
exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "..",
    # Heavily encouraged optional argument (see docs)
    # "rootFileTitle":         "API",
    # "fullApiSubSectionTitle": "",
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../../src"
}

# -- Options for MyST Parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
