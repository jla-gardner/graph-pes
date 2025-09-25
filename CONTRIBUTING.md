# Contributing

Contributions to `graph-pes` via pull requests are very welcome! H
ere's how to get started.

---

**Getting started**

Fork the repo on GitHub, and clone it to your local machine.

```bash
git clone https://github.com/<your-username-here>/graph-pes.git
cd graph-pes
```

Development on `graph-pes` makes use of the `uv` package/project manager. 
You can find more information about `uv` [here](https://docs.astral.sh/uv/). 
To install the `graph-pes` dependencies, run:

```bash
uv sync --all-extras
```

Finally, install the pre-commit hooks. 
This will ensure that any code you commit is formatted in a manner consistent with the rest of the codebase.

```bash
uv run pre-commit install
```

---

**If you're making changes to the code:**

Now make your changes. 
Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
uv run pytest tests/
```

Alternatively, run a specific test with:

```bash
uv run pytest tests/<test_name>.py
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
uv run sphinx-autobuild docs/source docs/build
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.
Any time you save changes to the documentation, these will shortly be reflected in the browser!