project = "graph-pes"
copyright = "2023-2025, John Gardner"
author = "John Gardner"
release = "0.2.4"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://ase-lib.org", None),
    "e3nn": ("https://docs.e3nn.org/en/latest/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "load-atoms": ("https://jla-gardner.github.io/load-atoms/", None),
}

html_logo = "_static/logo-square.svg"
html_title = "graph-pes"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
autodoc_member_order = "bysource"
maximum_signature_line_length = 70
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

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

nitpick_ignore = [
    ("py:class", "torch.nn.Parameter"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "e3nn.*"),
    ("py:class", "optional"),
    ("py:class", "o3.Irreps"),
    ("py:class", "graph_pes.config.training.FittingConfig"),
    ("py:class", "graph_pes.config.training.SWAConfig"),
    ("py:class", "graph_pes.config.training.GeneralConfig"),
    ("py:class", "graph_pes.config.shared.TorchConfig"),
    ("py:class", "pytorch_lightning.Callback"),
    ("py:class", "pytorch_lightning.Trainer"),
    ("py:class", "pytorch_lightning.callbacks.Callback"),
    ("py:class", "pytorch_lightning.callbacks.callback.Callback"),
    ("py:class", "TorchMetric"),
    ("py:class", "DirectForcefieldRegressor"),
    ("py:class", "ConservativeForcefieldRegressor"),
    ("py:class", "graph_pes.graph_property_model.GraphTensorModel"),
    ("ipython3", "Lexing literal_block"),
]

# override the default css to match the furo theme
nbsphinx_prolog = """
.. raw:: html

    <style>
        .jp-RenderedHTMLCommon tbody tr:nth-child(odd),
        div.rendered_html tbody tr:nth-child(odd) {
            background: var(--color-code-background);
        }
        .jp-RenderedHTMLCommon tr,
        .jp-RenderedHTMLCommon th,
        .jp-RenderedHTMLCommon td,
        div.rendered_html tr,
        div.rendered_html th,
        div.rendered_html td {
            color: var(--color-content-foreground);
        }
        .jp-RenderedHTMLCommon tbody tr:hover,
        div.rendered_html tbody tr:hover {
            background: #3c78d8aa;
        }
        div.nbinput.container div.input_area {
            /* border radius of 10px, but no outline */
            border-radius: 10px;
            border-style: none;
        }
        div.nbinput.container div.input_area > div.highlight > pre {
            padding: 10px;
            border-radius: 10px;
        }

    </style>
"""


# Add warning filter for specific lexing warning
import warnings

warnings.filterwarnings("ignore", r'.*Lexing literal_block.*as "ipython3".*')

nbsphinx_prompt_width = "0"

pygments_style = "friendly"
pygments_dark_style = "monokai"
