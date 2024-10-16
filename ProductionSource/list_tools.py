from langchain.tools import tool
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


@tool
def send_email(recipient: str, subject: str, content: str) -> str:
    """Gửi email đến một người nhận với chủ đề và nội dung đã cung cấp."""

    msg = MIMEMultipart()
    msg['From'] = "teamaithacoindustries@gmail.com"
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(content, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  #  Gmail
        server.starttls()  #  TLS
        server.login("teamaithacoindustries@gmail.com", "vzes umgr myis hgqd")
        server.send_message(msg)
        print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
    finally:
        server.quit()

    return f"Email đã được gửi tới {recipient}."


list_tools_use = [send_email]
