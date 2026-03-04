# Contributing

Thanks for your interest in contributing to Infinix-ClawAnalytics.

## How to contribute

- Search existing issues before opening a new one.
- Use clear, reproducible steps for bug reports.
- For new features, open an issue to discuss scope before sending a PR.

## Development setup

1. Create a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the dashboard locally:

```
streamlit run infinix_clawanalytics/analyzer/dashboard_app.py
```

## CSV workflow

- Use the CSV wizard to map fields and validate data.
- Save templates in the wizard for repeated schemas.
- Use the CSV diagnostics button to spot invalid dates or flags.

## Pull requests

- Keep PRs focused and small.
- Include a short summary and testing notes.
- Make sure the app runs locally before submitting.

## Code style

- Keep code readable and consistent with the existing style.
- Avoid large refactors mixed with feature changes.
