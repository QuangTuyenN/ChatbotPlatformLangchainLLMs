from langchain.tools import tool
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import psycopg2
from psycopg2 import sql
from uuid import UUID


# MINIO CONFIG
MINIO_EPT = os.environ.get("MINIO_ENDPOINT", "10.14.16.30:31003")
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "chatbotllms")

POSTGRESQL_DB_USER = os.environ.get("POSTGRESQL_DB_USER", "postgres")
POSTGRESQL_DB_PASS = os.environ.get("POSTGRESQL_DB_PASS", "thaco%401234")
POSTGRESQL_DB_NAME = os.environ.get("POSTGRESQL_DB_NAME", "db_phong_hop_test")
POSTGRESQL_DB_HOST = os.environ.get("POSTGRESQL_DB_HOST", "10.14.16.30")
POSTGRESQL_DB_PORT = os.environ.get("POSTGRESQL_DB_PORT", 30204)

POSTGRESQL_DB_PORT = int(POSTGRESQL_DB_PORT)

PASS_DB_TEMP = os.environ.get("PASS_DB_TEMP", "thaco@1234")


@tool
def send_email(recipient: str, subject: str, content: str) -> str:
    """Gửi email đến một người nhận với chủ đề và nội dung đã cung cấp."""

    msg = MIMEMultipart()
    msg['From'] = "teamaithacoindustries@gmail.com"
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(content, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail
        server.starttls()  # TLS
        server.login("teamaithacoindustries@gmail.com", "vzes umgr myis hgqd")
        server.send_message(msg)
        print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
    finally:
        server.quit()

    return f"Email đã được gửi tới {recipient}."


@tool
def send_link_form(name_form: str) -> str:
    """Đưa ra các link của biểu mẫu hoặc hồ sơ mà người dùng yêu cầu nếu name_form viết sai chính tả hoặc viết tắt thì hãy tự sửa lại một chút cho giống với string điều kiện,
    nếu name_form hoàn toàn khác với string điều kiện thì trả về không có hồ sơ biểu mẫu, lưu ý link file trả về phải có cả phần mở rộng của file"""
    list_link = []
    name_form = name_form.lower()
    if name_form in "hồ sơ thiết kế cơ sở":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PKH01-BM01.xlsx",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM09.pptx"]
    elif name_form in "hồ sơ thiết kế kỹ thuật":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PKH01-BM03.docx",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM10.xlsx"]
    elif name_form in "biểu mẫu thiết kế kiểu dáng":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM03.doc",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM04.pptx"]
    elif name_form in "biểu mẫu mô phỏng tính bền":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM05.xlsx",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM06.pptx"]
    elif name_form in "biểu mẫu kiểm tra lắp ráp":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM13.xls",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM15.xlsx"]
    elif name_form in "biểu mẫu kiểm tra lắp đặt":
        list_link = [f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM14.xls",
                     f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/QT.TTRD.PTSP_01-BM15.xls"]
    else:
        list_link = []

    if list_link:
        rep = f"Sau đây là file để truy cập vào hồ sơ biểu mẫu: {list_link[0]}, {list_link[1]}"
    else:
        rep = "Xin lỗi nhưng tôi không có hồ sơ biểu mẫu bạn cần."

    return rep


# @tool
# def query_meeting_room():
#     """Khi người dùng yêu cầu đặt phòng họp mà không đưa ra tên phòng đặt, ngày họp, thời gian bắt đầu, kết thúc,
#     tên người đặt, nội dung cuộc họp"""
#     return "Vui lòng cung cấp cho tôi tên phòng bạn muốn đặt (RD ứng với họp ở trung tâm RD và BSP ứng với họp ở " \
#            "Ban sản phẩm), ngày đặt (Ví dụ: 02-10 ứng với ngày 2 tháng 10), thời gian bắt đầu, kết thúc (Ví dụ: Bắt đầu" \
#            "lúc 1 giờ và kết thúc lúc 3 giờ), nội dung cuộc họp (Ví dụ: Họp nội bộ), " \
#            "và tên người đặt (Ví dụ: Nguyễn Thị A)"


@tool
def check_meeting_room(room_name: str, day: str, start_hour: int, end_hour: int, meeting_content: str, name_person: str) -> str:
    """Khi người dùng muốn đặt phòng họp và tuỳ theo thông tin họ cung cấp mà đưa ra phản hồi phù hợp"""
    connection = psycopg2.connect(
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT,
        database=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=PASS_DB_TEMP
    )
    if not room_name and not day:
        return "Vui lòng cung cấp cho tôi tên phòng bạn muốn đặt (RD ứng với họp ở trung tâm RD và BSP ứng với họp ở " \
               "Ban sản phẩm), ngày đặt (Ví dụ: 02-10 ứng với ngày 2 tháng 10), thời gian bắt đầu, kết thúc (Ví dụ: Bắt đầu" \
               "lúc 1 giờ và kết thúc lúc 3 giờ), nội dung cuộc họp (Ví dụ: Họp nội bộ), " \
               "và tên người đặt (Ví dụ: Nguyễn Thị A)"
    if not room_name:
        return "Vui lòng cung cấp tên phòng họp RD hoặc BSP"
    if not day:
        return "Vui lòng cung cấp ngày họp (Ví dụ 02-10)"
    if not start_hour:
        return "Vui lòng cung cấp giờ họp"
    if not end_hour:
        return "Vui lòng cung cấp giờ kết thúc"
    if not meeting_content:
        return "Vui lòng cung cấp nội dung cuộc họp"
    if not name_person:
        return "Vui lòng cung cấp tên người đặt"

    # Tạo cursor để thực thi truy vấn
    cursor = connection.cursor()

    # Truy vấn dữ liệu từ bảng roommeeting
    check_query = sql.SQL(
        """
        SELECT room_name, day, start_hour, end_hour, meeting_content, name_person 
        FROM roommeeting
        """
    )
    cursor.execute(check_query)

    # Lấy tất cả các dòng kết quả từ cursor
    rows = cursor.fetchall()

    # Lấy tên cột từ cursor
    columns = [desc[0] for desc in cursor.description]

    # Chuyển đổi dữ liệu thành list các dict
    result = [dict(zip(columns, row)) for row in rows]

    filtered_result = [
        item for item in result
        if item['room_name'] == room_name and item['day'] == day
    ]

    if filtered_result:
        for item in filtered_result:
            if int(item['start_hour']) <= int(start_hour) <= int(item['end_hour']):
                return 'Xin lỗi nhưng khung giờ bạn đặt đã được sử dụng, chúng tôi không thể đặt cho bạn được!'
    else:
        insert_query = sql.SQL(
            """
            INSERT INTO roommeeting (id, room_name, day, start_hour, end_hour, meeting_content, name_person)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        )
        cursor.execute(insert_query, (str(UUID(bytes=os.urandom(16), version=4)), room_name, day, str(start_hour), str(end_hour), meeting_content, name_person))
        connection.commit()
        cursor.close()
        connection.close()
        return f'Thông tin của bạn đã được ghi nhận, chúng tôi xin xác nhận đã đặt phòng họp ở phòng họp {room_name} ' \
               f'vào ngày {day} từ {start_hour} giờ đến {end_hour} giờ, với người đặt là {name_person} và nội' \
               f'dung cuộc họp là {meeting_content}.'


list_tools_use = [send_email, send_link_form, check_meeting_room]
