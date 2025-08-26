# public_app.py — 사용자 제출용(미리보기 + 동의 후 raw 버킷 제출)
# -----------------------------------------------------------
# ✅ Streamlit Secrets (Settings → Secrets)
# project_id = "feedback-ai-prototype-ver05"
# location   = "us-central1"
# tuned_model_name = "projects/feedback-ai-prototype-ver05/locations/us-central1/tunedModels/2731304531139756032"
#
# raw_bucket_name = "feedback-proto-ai-raw"
# raw_prefix      = "raw_submissions"
#
# [gcp_service_account]
# ...서비스계정 JSON 원문 전체...
# -----------------------------------------------------------

from __future__ import annotations
import io
import json
import uuid
from datetime import datetime, date
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd  # (미사용이지만 추후 확장 대비)
from google.oauth2 import service_account
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="🐸 개구리 학습 피드백 (Public)", page_icon="🐸", layout="centered")

PROJECT_ID = st.secrets.get("project_id")
LOCATION   = st.secrets.get("location", "us-central1")
TUNED_NAME = (st.secrets.get("tuned_model_name") or "").strip()
RAW_BUCKET = st.secrets.get("raw_bucket_name")
RAW_PREFIX = (st.secrets.get("raw_prefix") or "raw_submissions").strip().strip("/")

# Secrets 점검
if not (PROJECT_ID and LOCATION and TUNED_NAME and RAW_BUCKET):
    st.error("Secrets 설정이 부족합니다. project_id, location, tuned_model_name, raw_bucket_name를 확인하세요.")
    st.stop()

# ---------------- 인증/클라이언트 ----------------
try:
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
except Exception as e:
    st.error("Secrets의 [gcp_service_account] JSON을 확인하세요.\n" + repr(e))
    st.stop()

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

# ---------------- 공통 유틸 ----------------
def _gen_cfg() -> Dict[str, Any]:
    return {
        "max_output_tokens": 2048,     # 필요시 4096까지 올리세요
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "text/plain",
    }

def call_model_tuned(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """튜닝모델을 스트리밍으로 우선 호출 → 비면 동기 호출 폴백"""
    meta: Dict[str, Any] = {"route": []}

    # 1) 스트리밍
    try:
        gm = GenerativeModel(TUNED_NAME)
        parts: List[str] = []
        for chunk in gm.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=_gen_cfg(),
            stream=True,
        ):
            t = getattr(chunk, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts).strip()
        meta["route"].append({"name": "tuned-stream", "ok": bool(text)})
        if text:
            return text, meta
    except Exception as e:
        meta["route"].append({"name": "tuned-stream", "error": repr(e)})

    # 2) 동기
    try:
        gm = GenerativeModel(TUNED_NAME)
        r = gm.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=_gen_cfg(),
        )
        text = (getattr(r, "text", "") or "").strip()
        meta["route"].append({"name": "tuned-sync", "ok": bool(text)})
        return text, meta
    except Exception as e:
        meta["route"].append({"name": "tuned-sync", "error": repr(e)})
        return "", meta

def upload_json(bucket: str, path: str, obj: Dict[str, Any]):
    b = storage_client.bucket(bucket).blob(path)
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    b.cache_control = "no-cache"
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type="application/json")

def raw_key_for_today() -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{RAW_PREFIX}/{day}/{uuid.uuid4().hex[:10]}.json"

# ---------------- UI ----------------
with st.sidebar:
    st.markdown("**환경 정보**")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"Tuned model:\n`{TUNED_NAME}`")
    st.write(f"raw: `gs://{RAW_BUCKET}/{RAW_PREFIX}`")

st.title("🐸 개구리 학습 피드백")

st.caption("입력하신 내용은 익명으로 수집되어 서비스 개선에 활용될 수 있어요.")
prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180)

consent = st.checkbox("동의합니다. 입력 내용이 익명으로 저장되어 서비스 개선에 사용될 수 있음")

c1, c2 = st.columns([1,1])
with c1:
    preview = st.button("AI 초안 보기", use_container_width=True)
with c2:
    submit = st.button("케이스 제출", type="primary", use_container_width=True)

# 세션에 초안 보관
if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""

if preview:
    if not prompt.strip():
        st.warning("먼저 상황을 입력해주세요.")
    else:
        with st.spinner("초안 생성 중..."):
            text, meta = call_model_tuned(prompt)
            if text:
                st.session_state.draft_text = text
                st.success("초안이 생성되었습니다.")
            else:
                st.session_state.draft_text = ""
                st.warning("모델이 빈 응답을 반환했습니다.")
                with st.expander("디버그"):
                    st.json(meta)

if st.session_state.draft_text:
    st.markdown("### 👁️ AI 초안")
    st.text_area("초안 미리보기", st.session_state.draft_text, height=280)

if submit:
    if not consent:
        st.warning("제출하려면 동의에 체크해주세요.")
    elif not prompt.strip():
        st.warning("상황을 입력해주세요.")
    else:
        # 초안이 없으면 자동 생성 후 제출
        if not st.session_state.draft_text:
            text, _ = call_model_tuned(prompt)
            st.session_state.draft_text = text or ""
        record = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt": prompt.strip(),
            "ai_response": st.session_state.draft_text,
            "used_model": TUNED_NAME,
            "source_app": "public",
            "version": "v1",
        }
        key = raw_key_for_today()
        try:
            upload_json(RAW_BUCKET, key, record)
            st.success("제출 완료! 감사합니다 🙏")
            # 초기화
            st.session_state.draft_text = ""
        except Exception as e:
            st.error("제출에 실패했습니다.")
            st.exception(e)
