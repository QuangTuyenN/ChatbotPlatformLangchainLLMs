# import requests
# import urllib3
#
# # Tắt cảnh báo SSL
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#
# def reload_webchat():
#     url = "https://corellms.prod.bangpdk.dev/webchat/88e0cc7a-6695-4b69-a134-2139023421c6/reload"
#     headers = {
#         'Content-Type': 'application/json'
#     }
#     try:
#         # Bỏ qua kiểm tra chứng chỉ SSL
#         response = requests.get(url, headers=headers, verify=False)
#         if response.status_code == 200:
#             print("API call successful!")
#             print("Response:", response.json())
#         else:
#             print(f"Failed to call API. Status code: {response.status_code}")
#             print("Response:", response.text)
#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred: {e}")
#
# # Gọi hàm
# reload_webchat()


import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def post_webchat():
    url = "http://chatbotllms.shop/webchat/ea999510-eb09-4bd7-ba78-8e9eea795289/process"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "text": "ai là giám đốc thaco industries",
        "items": []
    }
    try:
        response = requests.post(url, headers=headers, json=data, verify=False)
        if response.status_code == 200:
            print("API call successful!")
            print("Response:", response.json())
        else:
            print(f"Failed to call API. Status code: {response.status_code}")
            print("Response:", response.text)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


# Gọi hàm
post_webchat()

# import chromadb
#
# try:
#     client = chromadb.HttpClient(host='10.14.16.30', port=30745)
#
#     # Lấy collection theo ID
#     collection_vecto = client.get_collection('88e0cc7a-6695-4b69-a134-2139023421c6')
#
#     # Lấy tất cả các vector từ collection
#     vectors = collection_vecto.get()
#
#     # Kiểm tra xem vectors có đúng là danh sách các dictionary không
#     print("Vectors trả về:", vectors)  # Debug để kiểm tra dữ liệu trả về
#
#     ids = vectors['ids']
#     collection_vecto.delete(ids=ids)
#     print("đã xóa")
#     # # Nếu vectors là danh sách các dictionary, trích xuất ID
#     # if isinstance(vectors, list) and all(isinstance(v, dict) for v in vectors):
#     #     ids = [v['ids'] for v in vectors]  # Truy cập 'id' từ từng vector
#     #
#     #     if ids:
#     #         # Xóa tất cả vector bằng danh sách ID
#     #         collection_vecto.delete(ids=ids)
#     #         print("Đã xóa toàn bộ vector embedding trong collection.")
#     #     else:
#     #         print("Không có vector nào để xóa.")
#     # else:
#     #     print("Dữ liệu trả về không đúng định dạng mong muốn.")
#
# except chromadb.errors.CollectionNotFoundError:
#     print("Collection không tồn tại.")
# except Exception as bug:
#     print("Lỗi: ", bug)

# import chromadb
#
# # Kết nối với ChromaDB server
# client = chromadb.HttpClient(host='10.14.16.30', port=30745)
#
# try:
#     # Lấy danh sách tất cả các collections
#     collections = client.list_collections()
#
#     print("Danh sách các collections:")
#     for collection in collections:
#         # Truy cập thuộc tính id và name từ đối tượng collection
#         print(f"- ID: {collection.id}, Name: {collection.name}")
# except Exception as e:
#     print("Lỗi: ", e)




