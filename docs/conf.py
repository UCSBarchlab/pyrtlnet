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

html_theme_options = {
    # For view/edit this page buttons.
    "source_repository": "https://github.com/UCSBarchlab/pyrtlnet",
    "source_branch": "main",
    "source_directory": "docs/",
    # Add a GitHub repository link to the footer.
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/UCSBarchlab/pyrtlnet",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}
