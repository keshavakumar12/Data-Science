from __future__ import annotations

import os
import tempfile
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import streamlit as st
from dotenv import load_dotenv

from mashup_core import create_mashup, is_valid_email


load_dotenv()

ROLL_NUMBER = "102303982"

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USERNAME or "no-reply@example.com")


def build_zip_bytes(file_path: Path) -> BytesIO:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=file_path.name)
    buffer.seek(0)
    return buffer


def send_email_with_zip(to_email: str, zip_bytes: BytesIO, zip_name: str) -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD):
        raise RuntimeError("SMTP credentials are missing. Set SMTP_USERNAME and SMTP_PASSWORD.")

    msg = EmailMessage()
    msg["Subject"] = "Your mashup file"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg.set_content(
        f"Hi,\n\nAttached is the mashup you requested.\n\nRoll Number: {ROLL_NUMBER}\n"
    )
    msg.add_attachment(
        zip_bytes.getvalue(),
        maintype="application",
        subtype="zip",
        filename=zip_name,
    )

    import smtplib

    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)


def main() -> None:
    st.set_page_config(page_title="Mashup Generator", page_icon="🎵", layout="centered")
    st.title("Mashup Generator")
    st.caption(f"Roll No. {ROLL_NUMBER} · Provide details to receive a zipped mashup by email.")

    with st.form("mashup_form"):
        singer = st.text_input("Singer name", placeholder="e.g., Sharry Maan")
        n_videos = st.number_input("Number of videos (>10)", min_value=11, value=12, step=1)
        duration = st.number_input("Clip duration seconds (>20)", min_value=21, value=25, step=1)
        email = st.text_input("Email to receive ZIP", placeholder="you@example.com")
        submitted = st.form_submit_button("Create & Send")

    if submitted:
        errors = []
        if not singer.strip():
            errors.append("Singer name is required.")
        if n_videos <= 10:
            errors.append("Number of videos must be greater than 10.")
        if duration <= 20:
            errors.append("Duration must be greater than 20 seconds.")
        if not is_valid_email(email):
            errors.append("Please provide a valid email address.")

        if errors:
            st.error("Please fix the issues below:\n- " + "\n- ".join(errors))
            return

        with st.spinner("Building your mashup..."):
            try:
                with tempfile.TemporaryDirectory(prefix="mashup_streamlit_") as temp_root:
                    output_mp3 = Path(temp_root) / "mashup.mp3"
                    result_path = create_mashup(
                        singer=singer,
                        n_videos=int(n_videos),
                        clip_seconds=int(duration),
                        output_path=output_mp3,
                    )
                    zip_bytes = build_zip_bytes(result_path)
                    send_email_with_zip(
                        to_email=email,
                        zip_bytes=zip_bytes,
                        zip_name=f"{ROLL_NUMBER}_mashup.zip",
                    )
                st.success(f"Success! Mashup sent to {email}.")
            except Exception as exc:
                st.error(f"Failed to create or send mashup: {exc}")


if __name__ == "__main__":
    main()
