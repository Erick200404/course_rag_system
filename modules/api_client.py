"""Streamlit talks to the backend; the backend owns documents and history."""

import os
from urllib.parse import quote

import requests

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def api_request(method: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 15)
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
    except requests.HTTPError as error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{detail}") from error
    except requests.RequestException as error:
        raise RuntimeError("无法连接问答服务，请检查 FastAPI 后端是否启动。") from error
    return response.json()


def upload_document(course_id: str, uploaded_file):
    return api_request(
        "PUT", f"/courses/{course_id}/documents/{quote(uploaded_file.name, safe='')}",
        data=uploaded_file.getvalue(), headers={"Content-Type": "application/octet-stream"}, timeout=120,
    )
