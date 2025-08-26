import streamlit as st
from datetime import datetime
import uuid, io, json

from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.cloud import storage

# ---------- 기본 설정 ----------
st.set_page_config(page_title="학습 피드백 AI", page_icon="🐸", layout="centered")

PROJECT_ID = st.secrets.get("project_id")
LOCATION   = st.secrets.get("location", "us-central1")
ENDPOINT   = st.secrets.get("endpoint_name", "").strip()

BUCKET     = st.secrets.get("gcs_bucket_name", "")
PREFIX     = (st.secrets.get("gcs_prefix") or "raw_submissions").strip().strip("/")
AUTO_BACKUP = bool(st.secrets.get("enable_gcs_backup", True))

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES
)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials)
storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

def gcs_upload_bytes(bucket_name: str, blob_path: str, data: bytes, content_type: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.cache_control = "no-cache"
    blob.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def call_model(model_name: str, prompt_text: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
    )
    return resp.text or ""

# ---------- UI ----------
st.title("🐸 개구리 학습 피드백")
st.caption("입력하신 내용은 익명으로 수집되어 서비스 개선에 활용될 수 있습니다.")
prompt = st.text_area("학생의 상황을 자세히 입력해주세요:", height=180)

consent = st.checkbox("동의합니다: 입력 내용이 익명으로 저장되어 서비스 개선에 사용될 수 있음")

col1, col2 = st.columns(2)
with col1:
    gen_clicked = st.button("AI 초안 보기", use_container_width=True)
with col2:
    submit_clicked = st.button("케이스 제출", use_container_width=True)

if gen_clicked:
    if not prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    else:
        with st.spinner("AI 초안 생성 중..."):
            try:
                ai_text = call_model(ENDPOINT, prompt)
                st.session_state["preview_ai"] = ai_text
                st.success("아래 초안을 확인하세요.")
            except Exception as e:
                st.error("생성 실패"); st.exception(e)

if "preview_ai" in st.session_state and st.session_state["preview_ai"]:
    st.markdown("### 🤖 AI 초안")
    st.write(st.session_state["preview_ai"])

if submit_clicked:
    if not prompt.strip():
        st.warning("학생의 상황을 입력해주세요.")
    elif not consent:
        st.warning("수집/활용 동의에 체크해주세요.")
    elif not BUCKET:
        st.error("서버 설정 오류: GCS 버킷이 설정되지 않았습니다.")
    else:
        try:
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            uid = uuid.uuid4().hex[:10]
            entry = {
                "timestamp": ts,
                "prompt": prompt,
                "ai_response": st.session_state.get("preview_ai", ""),
                "used_model": ENDPOINT,
                "submitted_by": f"public:{uid}"
            }
            day = datetime.utcnow().strftime("%Y-%m-%d")
            key = f"{PREFIX}/{day}/{ts}_{uid}.json"
            gcs_upload_bytes(BUCKET, key, json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")
            st.success("제출 완료! 감사합니다 🙏")
            st.session_state["preview_ai"] = ""
        except Exception as e:
            st.error("제출 중 오류가 발생했습니다."); st.exception(e)
