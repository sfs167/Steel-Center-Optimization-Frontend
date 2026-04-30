# Steel Inventory Streamlit Frontend

This is the public frontend-only Streamlit app.

It does not contain the optimization model code. It sends input data to the
deployed Nextmv Cloud app and displays the returned solution, tables, and
graphs.

## Streamlit Cloud

Use this as the app entry point:

```text
app.py
```

Add these secrets in Streamlit Cloud:

```toml
NEXTMV_API_KEY = "your-nextmv-api-key"
NEXTMV_APP_ID = "steel-center"
NEXTMV_INSTANCE_ID = "devint"
APP_USERNAME = "admin"
APP_PASSWORD = "your-strong-password"
```

## Public Repo Contents

Safe files for the public repo:

- `app.py`
- `default_input.json`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `README.md`

Do not commit a real `.streamlit/secrets.toml` file.
