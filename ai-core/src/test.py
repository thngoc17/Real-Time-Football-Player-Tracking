import base64
import json
import urllib.request
import urllib.error

# 1. Cấu hình thông số
ENDPOINT_URL = "https://yolo-endpoint-ver1.eastasia.inference.ml.azure.com/score"
IMAGE_PATH = r"E:\PyCharm\YOLO_football\dataset\images\test\Match_1953_2_0_subclip_frame_000847.jpg"  # Thay bằng đường dẫn ảnh thực tế trên máy


def main():
    try:
        # 2. Đọc và mã hóa ảnh sang Base64
        with open(IMAGE_PATH, "rb") as f:
            img_bytes = f.read()

        base64_str = base64.b64encode(img_bytes).decode("utf-8")

        # 3. Đóng gói payload theo chuẩn yêu cầu của online_score.py
        payload = {
            "image": base64_str
        }

        data = json.dumps(payload).encode("utf-8")

        # 4. Khởi tạo request
        req = urllib.request.Request(
            url=ENDPOINT_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        # 5. Gửi request và nhận kết quả
        print(f"Đang gửi ảnh tới {ENDPOINT_URL}...")
        with urllib.request.urlopen(req) as response:
            result = response.read().decode("utf-8")

            # Làm đẹp chuỗi JSON trả về
            parsed_result = json.loads(result)
            print("\n=== KẾT QUẢ SUY LUẬN ===")
            print(json.dumps(parsed_result, indent=2, ensure_ascii=False))

    except urllib.error.HTTPError as e:
        print(f"Lỗi HTTP: {e.code} - {e.reason}")
        print(e.read().decode("utf-8"))
    except Exception as e:
        print(f"Lỗi hệ thống: {str(e)}")


if __name__ == "__main__":
    main()