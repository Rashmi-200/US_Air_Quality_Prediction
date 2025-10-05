import streamlit as st
import pandas as pd
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------- Helpers ----------
def send_email(smtp_server, port, sender_email, sender_pass, to_email, subject, body):
    """Send one email via SMTP with TLS (port 587)"""
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls(context=context)   # Upgrade to secure TLS
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, to_email, msg.as_string())
        return "SENT ✅"
    except Exception as e:
        return f"FAILED ❌ ({e})"

# ---------- Streamlit Tab ----------
def show_alerts_tab():
    st.header("📧 Air Quality Email Alerts")
    st.info("Upload a CSV with a column **email**. Each listed email will receive the advisory message.")

    uploaded_file = st.file_uploader("📂 Upload Email CSV", type=["csv"])
    subject = st.text_input("📌 Email Subject", placeholder="Air Quality Advisory")
    message = st.text_area("📝 Write your message", placeholder="Type advisory email here...")

    # Load email credentials from secrets.toml
    smtp_server   = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port     = int(st.secrets.get("SMTP_PORT", 587))   # default now 587
    sender_email  = st.secrets.get("SENDER_EMAIL", "")
    sender_pass   = st.secrets.get("SENDER_PASS", "")

    email_list = []
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            return

        if "email" not in df.columns:
            st.error("❌ CSV must have a column named 'email'.")
            return

        df["email"] = df["email"].astype(str).str.strip()
        email_list = df["email"].dropna().tolist()

        st.success(f"✅ Loaded {len(email_list)} email addresses.")
        st.dataframe(df.head())

    # Center the send button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📤 Send Email Alerts", use_container_width=True):
            if not smtp_server or not sender_email or not sender_pass:
                st.error("❌ Missing SMTP credentials in secrets.toml")
                return
            if not email_list:
                st.error("❌ No emails found.")
                return
            if not message.strip():
                st.error("❌ Message cannot be empty.")
                return

            results = []
            with st.spinner("📡 Sending emails..."):
                for email in email_list:
                    status = send_email(
                        smtp_server, smtp_port, sender_email, sender_pass,
                        email, subject.strip() or "Air Quality Alert", message.strip()
                    )
                    results.append({"to": email, "status": status})

            st.subheader("📊 Delivery Report")
            st.dataframe(pd.DataFrame(results))

            st.success("✅ Email broadcast finished. Check your mailbox or server logs for confirmation.")