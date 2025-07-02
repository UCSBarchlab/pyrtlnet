# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
project = "pyrtlnet"
copyright = "2025, Jeremy Lau"
author = "Jeremy Lau"

# -- General configuration ---------------------------------------------------
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be extensions coming
# with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# Omit redundant method names in right sidebar (run() instead of NumPyInference.run()).
toc_object_entries_show_parents = "hide"

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files. This pattern also affects html_static_path and
# html_extra_path.
exclude_patterns = ["_build"]

# sphinx.ext.intersphinx configuration.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyrtl": ("https://pyrtl.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# sphinx_copybutton: exclude line numbers, prompts, and outputs.
copybutton_exclude = ".linenos, .gp, .go"

# sphinx-autodoc-typehints configuration: Always display Unions with vertical bars,
# show default values, and don't document :rtype: None.
always_use_bars_union = True
typehints_defaults = "comma"
typehints_document_rtype_none = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for a list of
# builtin themes.
html_theme = "furo"
html_title = "pyrtlnet Reference Documentation"
