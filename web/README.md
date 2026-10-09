# highres-TA methods page

Static page, no build step. Because it loads JSON with `fetch`, serve the folder:

    cd web && python -m http.server 8000     # then open http://localhost:8000

| File | Purpose | Edit it? |
|---|---|---|
| `content.json` | **All text**, section order and block layout | **Yes** |
| `data.json` | Numbers and member predictions from the model run | No: regenerate with `uv run python web/build_data.py` |
| `tokens.css` | RdYlBu palette and light/dark tokens | Yes, for colours |
| `style.css`, `app.js` | Layout and page renderer | Rarely |
| `uncertainty.css`, `uncertainty.js` | Interactive uncertainty explorer and charts | Rarely |

## Editing content

Each section in `content.json` has `blocks`, rendered in order. Text supports `**bold**`, `` `code` ``
and `{{path.to.value|fmt}}` placeholders that read from `data.json` (`fmt`: `int`, `pct`, or a number of decimals).

Block types: `text`, `cards`, `steps`, `table`, `callout` (`tone`: info/warn/key), `equation`, `funnel`,
`split-explorer`, `member-grid`, `uncertainty-explorer`, `variance-decomposition`, `spread-vs-error`,
`ensemble-growth`. Reorder, remove or add blocks freely.
