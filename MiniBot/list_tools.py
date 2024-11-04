from langchain.tools import tool
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import psycopg2
from psycopg2 import sql
from uuid import UUID
from minio import Minio
from minio.error import S3Error
import requests


# MINIO CONFIG
MINIO_EPT = os.environ.get("MINIO_ENDPOINT", "10.14.16.30:31003")
MINIO_EPT_DOMAIN = os.environ.get("MINIO_EPT_DOMAIN", "minio.prod.bangpdk.dev")
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "chatbotllms")

POSTGRESQL_DB_USER = os.environ.get("POSTGRESQL_DB_USER", "postgres")
POSTGRESQL_DB_PASS = os.environ.get("POSTGRESQL_DB_PASS", "thaco%401234")
POSTGRESQL_DB_NAME = os.environ.get("POSTGRESQL_DB_NAME", "db_phong_hop_test")
POSTGRESQL_DB_HOST = os.environ.get("POSTGRESQL_DB_HOST", "10.14.16.30")
POSTGRESQL_DB_PORT = os.environ.get("POSTGRESQL_DB_PORT", 30204)

POSTGRESQL_DB_PORT = int(POSTGRESQL_DB_PORT)

PASS_DB_TEMP = os.environ.get("PASS_DB_TEMP", "thaco@1234")


# Config MinIO client
minio_client = Minio(
    endpoint=os.environ.get("MINIO_ENDPOINT", "minio.prod.bangpdk.dev"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", "teamaithaco"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "thaco@1234"),
    secure=False  # True: MinIO server use HTTPS
)


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
    connection = psycopg2.connect(
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT,
        database=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password="thaco@1234"
    )
    # Tạo cursor để thực thi truy vấn
    cursor = connection.cursor()

    # Truy vấn dữ liệu từ bảng roommeeting
    check_query = sql.SQL(
        """
        SELECT id, bucket_name, bucket_description 
        FROM miniofileupload
        """
    )
    cursor.execute(check_query)

    # Lấy tất cả các dòng kết quả từ cursor
    rows = cursor.fetchall()
    for row in rows:
        if name_form in row[2]:
            bucket_name = row[1]
            objects = minio_client.list_objects(bucket_name)
            for obj in objects:
                list_link.append(f"https://{MINIO_EPT_DOMAIN}/{bucket_name}/{obj.object_name}")

    if list_link:
        rep = f"Sau đây là file để truy cập vào hồ sơ biểu mẫu: {list_link}"
    else:
        rep = "Xin lỗi nhưng tôi không có hồ sơ biểu mẫu bạn cần."

    return rep


@tool
def query_meeting_room():
    """Khi người dùng yêu cầu đặt phòng họp chung chung mà không đưa ra đầy đủ thông tin về phòng họp
     trả về lệnh "Đặt phòng họp" để front end đưa ra form"""
    return "Vui lòng điền vào form bên dưới các thông tin đặt phòng họp bên dưới để chúng tôi hỗ trợ ạ."


@tool
def check_meeting_room(room_name: str, day: str, start_hour: float, end_hour: float, meeting_content: str, name_person: str):
    """Khi người dùng đưa thông tin đặt phòng họp và cung cấp đủ thông tin tên phòng họp, ngày, thời gian bắt đầu,
    kết thúc, nội dung họp, tên người đặt tuỳ theo thông tin họ cung cấp mà đưa ra phản hồi đặt được hay không"""
    connection = psycopg2.connect(
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT,
        database=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=PASS_DB_TEMP
    )
    print("start: ", start_hour)
    print("end: ", end_hour)
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
    print("filter result: ", filtered_result)
    if filtered_result:
        for item in filtered_result:
            print("item: ", item)
            print("item start_hour: ", item['start_hour'])
            print(start_hour)
            print("item end hour: ", item['end_hour'])
            if (float(item['start_hour']) <= float(start_hour)) and (float(start_hour) < float(item['end_hour'])):
                print("-------")
                return 'Xin lỗi nhưng khung giờ bạn đặt đã được sử dụng, chúng tôi không thể đặt cho bạn được!'
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


@tool
def check_available_hour_meeting_room(room_name: str, day: str) -> str:
    """Khi người dùng yêu cầu đưa ra thông tin các khung giờ trống hoặc khung giờ đã sử dụng của 1 ngày
    cụ thể thì tùy vào thông tin yêu cầu mà đưa ra phản hồi phù hợp"""
    if not room_name or not day:
        return "Vui lòng cung cấp đầy đủ tên phòng họp và ngày bạn muốn kiểm tra"
    connection = psycopg2.connect(
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT,
        database=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=PASS_DB_TEMP
    )

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

    rep = []
    for item in filtered_result:
        if float(item['start_hour']).is_integer():
            item['start_hour'] = f"{item['start_hour']} giờ"
        else:
            item['start_hour'] = f"{float(item['start_hour']) - 0.5} giờ 30"
        if float(item['end_hour']).is_integer():
            item['end_hour'] = f"{item['end_hour']} giờ"
        else:
            item['end_hour'] = f"{float(item['end_hour']) - 0.5} giờ 30"
        rep.append(f"từ {item['start_hour']} đến {item['end_hour']}")

    return f"Các khung giờ đã được đặt tại phòng họp {room_name} ngày {day} là: {rep}"


list_tools_use = [send_email, send_link_form, query_meeting_room, check_meeting_room, check_available_hour_meeting_room]
