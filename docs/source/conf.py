project = "graph-pes"
copyright = "2023, John Gardner"
author = "John Gardner"
release = "0.0.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
autodoc_member_order = "bysource"
maximum_signature_line_length = 100
autodoc_typehints = "description"

copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True
copybutton_selector = "div.copy-button pre"

logo_highlight_colour = "#f74565"
code_color = "#f74565"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-problematic": code_color,
        "color-brand-primary": logo_highlight_colour,
        "color-brand-content": logo_highlight_colour,
    },
    "dark_css_variables": {
        "color-problematic": code_color,
        "color-brand-primary": logo_highlight_colour,
        "color-brand-content": logo_highlight_colour,
    },
}
