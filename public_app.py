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
from datetime import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
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
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
except Exception as e:
    st.error("Secrets의 [gcp_service_account] JSON을 확인하세요.\n" + repr(e))
    st.stop()

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

# ---------------- 모델 호출/유틸 ----------------
def _gen_cfg() -> Dict[str, Any]:
    # 최소 옵션만 사용 (InvalidArgument 방지)
    return {
        "max_output_tokens": 2048,   # 필요하면 4096까지
        "temperature": 0.7,
        "top_p": 0.95,
    }

def _extract_text(r) -> str:
    # r.text 우선, 없으면 candidates.parts[*].text 수집
    if getattr(r, "text", None):
        return (r.text or "").strip()
    pieces: List[str] = []
    for c in getattr(r, "candidates", []) or []:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    pieces.append(t)
    return "\n".join(pieces).strip()

def call_model_tuned(prompt: str) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"route": []}

    # 1) 튜닝모델 동기 호출
    try:
        gm = GenerativeModel(TUNED_NAME)
        r = gm.generate_content(
            contents=[{"role":"user","parts":[{"text":prompt}]}],
            generation_config=_gen_cfg(),
        )
        text = _extract_text(r)
        meta["route"].append({"name":"tuned-sync", "ok": bool(text)})
        if text:
            return text, meta
    except Exception as e:
        meta["route"].append({"name":"tuned-sync", "error": repr(e)})

    # 2) 베이스모델 폴백
    try:
        base = GenerativeModel("gemini-1.5-pro-002")
        r2 = base.generate_content(
            contents=[{"role":"user","parts":[{"text":prompt}]}],
            generation_config=_gen_cfg(),
        )
        text2 = _extract_text(r2)
        meta["route"].append({"name":"base-sync", "ok": bool(text2)})
        return text2, meta
    except Exception as e:
        meta["route"].append({"name":"base-sync", "error": repr(e)})
        return "", meta

def _upload_json(bucket: str, key: str, obj: Dict[str, Any]):
    b = storage_client.bucket(bucket).blob(key)
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    b.cache_control = "no-cache"
    b.upload_from_file(io.BytesIO(data), size=len(data), content_type="application/json")

def _raw_key_for_today() -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{RAW_PREFIX}/{day}/{uuid.uuid4().hex[:10]}.json"

# ---------------- UI ----------------
with st.sidebar:
    st.markdown("**환경 정보**")
    st.write(f"Project: `{PROJECT_ID}`")
    st.write(f"Location: `{LOCATION}`")
    st.write(f"Tuned model:\n`{TUNED_NAME}`")
    st.write(f"raw: `gs://{RAW_BUCKET}/{RAW_PREFIX}`")

st.title("🐸 개구리 학습 피드백 (Public)")
st.caption("입력하신 내용은 익명으로 수집되어 서비스 개선에 활용될 수 있어요.")

prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180, key="pub_prompt")
consent = st.checkbox("동의합니다. 입력 내용이 익명으로 저장되어 서비스 개선에 사용될 수 있음", key="pub_consent")

c1, c2 = st.columns([1,1])
with c1:
    preview = st.button("AI 초안 보기", use_container_width=True, key="pub_preview_btn")
with c2:
    submit = st.button("케이스 제출", type="primary", use_container_width=True, key="pub_submit_btn")

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
    st.text_area("초안 미리보기", st.session_state.draft_text, height=280, key="pub_draft_view")

if submit:
    if not consent:
        st.warning("제출하려면 동의에 체크해주세요.")
    elif not prompt.strip():
        st.warning("상황을 입력해주세요.")
    else:
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
        key = _raw_key_for_today()
        try:
            _upload_json(RAW_BUCKET, key, record)
            st.success("제출 완료! 감사합니다 🙏")
            st.session_state.draft_text = ""
        except Exception as e:
            st.error("제출에 실패했습니다.")
            st.exception(e)
