import os
import shutil
import cv2
from deepface import DeepFace

# 載入 Haar Cascade 人臉模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# 資料夾設定（配合作業題目）
# ==========================================
source_dir = "data"  # 原始包含三個資料夾的目錄
input_dir = "face_data"  # 整合所有圖片的目的地資料夾
output_dir = "face_data_ok"  # 辨識結果輸出資料夾

categories = ["Sad", "Angry", "Happy"]

# 建立需要的資料夾
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 支援的圖片格式
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# ==========================================
# 步驟 1：將 Sad, Angry, Happy 的圖片整合到 face_data
# ==========================================
print("正在整合資料夾圖片...")
for category in categories:
    category_path = os.path.join(source_dir, category)

    if os.path.exists(category_path):
        for filename in os.listdir(category_path):
            if filename.lower().endswith(image_extensions):
                source_file = os.path.join(category_path, filename)

                # 為了防止不同資料夾有同名檔案被覆蓋，新檔名加上分類前綴
                new_filename = f"{category}_{filename}"
                target_file = os.path.join(input_dir, new_filename)

                # 複製圖片到 face_data
                shutil.copy(source_file, target_file)
print("圖片整合完成！\n")

# ==========================================
# 步驟 2：逐一讀取 face_data 資料夾中的圖片進行辨識
# ==========================================
print("開始進行 DeepFace 表情辨識...")
for filename in os.listdir(input_dir):

    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(input_dir, filename)

    print(f"Processing: {filename}")

    # 讀取圖片
    img = cv2.imread(image_path)

    if img is None:
        print(f"Cannot read image: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # 分析每張臉
    for (x, y, w, h) in faces:

        face_img = img[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )

            emotion = result[0]["dominant_emotion"]

            # 畫框與文字
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print(f"Error analyzing face in {filename}: {e}")

    # 儲存結果到 face_data_ok 資料夾
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)

    print(f"Saved: {output_path}")

print("\n所有圖片處理完畢！結果已儲存至 face_data_ok 資料夾。")