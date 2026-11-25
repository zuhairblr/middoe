# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'MIDDoE'
copyright = '2025, Zuhair Tabrizi, CAPE-Lab University of Padova'
author = 'Zuhair Tabrizi, Elena Barbera, Wilson Ricardo Leal Da Silva, Fabrizio Bezzo'
# Shown version = the installed package version (pyproject.toml), so the docs never claim another one.
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('middoe')
except Exception:
    release = ''
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon settings (NumPy docstring style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True


# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

master_doc = 'index'


# -- Render the LaTeX written in docstrings ---------------------------------
# Many docstrings write maths as \( ... \) (inline) and \[ ... \] (display, delimiters on their own
# lines). In reStructuredText these are just escaped brackets, so the formulas would appear as raw
# text. This hook converts them to :math:`...` roles and ``.. math::`` blocks when the docs are built;
# the source code is not changed.
import re as _re

_BS = chr(92)                                  # a single backslash
_OPEN_D, _CLOSE_D = _BS + '[', _BS + ']'       # display-math delimiters
_INLINE = _re.compile(_re.escape(_BS + '(') + '(.+?)' + _re.escape(_BS + ')'))
_OK_BEFORE = set(' \t([{<"\'-/:')
_OK_AFTER = set(' \t)]}>"\'-/:.,;!?')


def _inline_math(line):
    out, pos = [], 0
    for m in _INLINE.finditer(line):
        start, end = m.span()
        role = ':math:`' + m.group(1).strip() + '`'
        if start > 0 and line[start - 1] not in _OK_BEFORE:
            role = _BS + ' ' + role
        if end < len(line) and line[end] not in _OK_AFTER:
            role += _BS + ' '
        out.append(line[pos:start] + role)
        pos = end
    out.append(line[pos:])
    return ''.join(out)


def _latex_to_rst(app, what, name, obj, options, lines):
    new, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == _OPEN_D:
            indent = line[:len(line) - len(line.lstrip())]
            body, i = [], i + 1
            while i < len(lines) and lines[i].strip() != _CLOSE_D:
                body.append(lines[i].strip())
                i += 1
            new += ['', indent + '.. math::', ''] + [indent + '   ' + b for b in body if b] + ['']
            i += 1  # skip the closing \]
            continue
        new.append(_inline_math(line))
        i += 1
    lines[:] = new


def setup(app):
    app.connect('autodoc-process-docstring', _latex_to_rst, priority=400)
