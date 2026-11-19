import os

import httpx
import streamlit as st

API_URL = os.environ.get("FINOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")


class BackendUnavailable(Exception):
    pass


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@st.cache_resource
def _client():
    if API_URL == "inprocess":
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)
    return httpx.Client(base_url=API_URL, timeout=90)


def request(method: str, path: str, json: dict = None, params: dict = None):
    try:
        response = _client().request(method, path, json=json, params=params)
    except httpx.TransportError as exc:
        raise BackendUnavailable(str(exc)) from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):
            detail = "; ".join(f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}" for item in detail)
        raise ApiError(response.status_code, str(detail))
    return response.json()


@st.cache_data(ttl=900, show_spinner=False, max_entries=200)
def _cached_get(path: str, params: tuple, version: int):
    return request("GET", path, params=dict(params))


def get(path: str, version: int, **params):
    clean = tuple(sorted((key, str(value)) for key, value in params.items() if value not in (None, "", "All")))
    return _cached_get(path, clean, version)


def state() -> dict:
    return request("GET", "/api/v1/demo/state")
