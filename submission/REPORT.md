# Báo cáo Day 09 — Phạm Văn Ngọc Vĩnh, 03262

## 1. Kết luận trước
- **Kết luận của pipeline:** **RELEASE** (Phát hành dataset).
- **Số lượng ảnh:** Nhận **203/224** ảnh đầu vào (loại bỏ 21 ảnh hỏng, quá nhỏ, trùng lặp và rò rỉ sang tập test/val). Xuất ra bộ dữ liệu sạch tại `submission/export/clean_v1/` gồm **203** ảnh và đầy đủ nhãn tương ứng.
- **Lý do chính:** Sau 3 vòng kiểm tra chất lượng (QC), sửa tự động các lỗi cú pháp và mô phỏng người duyệt 80 ảnh ưu tiên cao nhất, bộ dữ liệu còn **0 lỗi bắt buộc (ERROR)** và **16 ảnh có cảnh báo (WARN)** trên tổng số 203 ảnh (tỉ lệ cảnh báo là 7.88%, thấp hơn ngưỡng Hold 10.00% quy định tại cửa chặn).

## 2. Lỗi và phân việc
Tổng hợp các mã lỗi phát hiện ở vòng 1 từ `02_error_list_v1`:

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_CLASS_ID | 6 dòng | Máy (autofix) | Lỗi cú pháp: ID lớp ngoài danh mục cho phép (0..3). Máy xoá dòng an toàn (`drop_line`). |
| BAD_CLASS_NAME | 2 lỗi | Máy (autofix) | Lỗi cấu hình `data.yaml`: tên lớp viết hoa/sai vị trí. Máy ghi lại đúng `cfg["classes"]`. |
| BAD_FORMAT | 3 dòng | Máy (autofix) | Lỗi cú pháp: dòng nhãn thiếu cột hoặc chứa NaN/Inf. Máy xoá dòng hỏng. |
| BAD_SIZE | 3 ảnh | Máy (autofix) | Cạnh ngắn ảnh < 250px không đủ chuẩn độ phân giải. Máy loại bỏ ảnh và nhãn. |
| BOX_HUGE | 3 box | Người (hàng đợi) | Cảnh báo box chiếm > 95% ảnh: có thể là lỗi vẽ bao trùm hoặc vật thật cận cảnh, cần mắt người thẩm định. |
| BOX_OUT_OF_IMAGE | 8 dòng | Máy (autofix) | Tọa độ mép box tràn ra ngoài [0, 1]. Máy cắt gọn box về biên ảnh (`clip_box`). |
| BOX_TINY | 6 box | Người (hàng đợi) | Cạnh ngắn box < 12px: vật thể quá nhỏ cần người kiểm tra có đáng gán nhãn không. |
| BOX_ZERO | 4 dòng | Máy (autofix) | Chiều rộng hoặc chiều cao box <= 0. Máy xoá dòng rỗng này. |
| CLASS_RARE | 4 lỗi | Người (dataset-level) | Lớp xuất hiện quá ít mẫu trong dataset; cảnh báo mức thống kê cho quản lý dữ liệu. |
| CORRUPT_IMAGE | 2 ảnh | Máy (autofix) | File ảnh bị hỏng cấu trúc không thể nạp/đọc. Máy xoá bỏ khỏi dataset. |
| DUP_BOX | 7 dòng | Máy (autofix) | Trùng lặp cùng lớp với box trước đó (IoU > 0.9). Máy xoá box trùng lặp ở dòng sau. |
| DUP_IMAGE_EXACT | 8 ảnh | Máy (autofix) | Ảnh trùng từng byte (MD5) với ảnh đã nhận trước đó. Máy xoá bỏ ảnh trùng. |
| DUP_IMAGE_NEAR | 3 ảnh | Máy (autofix) | Ảnh gần trùng do nén/resize (pHash dist <= 6). Máy loại bỏ để tránh thiên lệch dữ liệu. |
| EMPTY_LABEL | 12 ảnh | Người (hàng đợi) | File nhãn rỗng: cần người xác nhận đây là ảnh nền (background) hay bị sót nhãn. |
| LEAK_TEST | 5 ảnh | Máy (autofix) | Ảnh trùng với kho dữ liệu test/gold_val. Máy loại bỏ ngay để chống rò rỉ dữ liệu. |
| MISSING_LABEL_FILE | 6 ảnh | Người (hàng đợi) | Ảnh không có file nhãn đi kèm. Máy tuyệt đối không bịa nhãn, phải chuyển người gán. |
| POSSIBLE_FALSE_POS | 29 box | Người (hàng đợi) | Box nhãn không khớp với bất kỳ phát hiện nào của checker model yolo26l. |
| POSSIBLE_LOOSE_BOX | 12 box | Người (hàng đợi) | Box nhãn vẽ lỏng lẻo hoặc lệch tâm so với dự đoán của model (IoU < 0.5). |
| POSSIBLE_MISSING | 32 box | Người (hàng đợi) | Model phát hiện vật thể rõ nét nhưng nhãn thiếu box. |
| POSSIBLE_WRONG_CLASS | 14 box | Người (hàng đợi) | Nhãn gán một lớp trong khi model checker nhận diện độ tin cậy cao là lớp khác. |

Hàng đợi duyệt `05_review_queue.csv` xếp các ảnh `NEEDS_LABEL` (trọng số 10) và các ảnh kết hợp nhiều cảnh báo nghiêm trọng (như vừa sửa cú pháp vừa nghi ngờ nhầm lớp/thiếu box) lên đầu tiên. Với ngân sách 80 ảnh, người duyệt mô phỏng đã hoàn thành gán nhãn cho toàn bộ 6 ảnh thiếu file nhãn và giải quyết các lỗi ngữ nghĩa quan trọng nhất, đưa tỉ lệ cảnh báo còn lại xuống 7.88%.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 6 | 1.00 / 0.50 (F1=0.67) → 1.00 / 0.67 (F1=0.80) | Ngưỡng 2 quá khắt khe, bỏ sót ảnh resize như `img_0223` (dist=6 so với `img_0077`). Nâng lên 6 giúp phát hiện chính xác ảnh biến đổi mà Precision vẫn đạt 1.00. |
| `second_opinion.conf` | 0.25 | 0.8 | WrongClass: 0.56 / 0.83 → 1.00 / 0.83; Missing: 0.50 / 1.00 → 0.67 / 0.75 | Ngưỡng 0.25 khiến model checker sinh quá nhiều báo động giả, làm hàng đợi vượt quá ngân sách 80 ảnh. Tăng lên 0.8 giúp lọc các cảnh báo chắc chắn, nâng độ chính xác cảnh báo sai lớp lên 100%. |

**Một chỗ calibration có thể đánh lừa bạn:**
Trên tập `data/calib` (chỉ có 39 ảnh nhân tạo), nếu chọn `phash_dist = 16`, điểm F1 trên calib đạt cao nhất (0.91, recall 0.83). Tuy nhiên khi áp dụng lên tập `raw` 224 ảnh thực tế, khoảng cách 16 quá lỏng khiến thuật toán coi các ảnh hoàn toàn khác nhau là trùng lặp (ví dụ `img_0008` và `img_0011` có dist=16 nhưng nội dung khác hẳn), đồng thời báo rò rỉ giả tới 11 ảnh sang tập test thay vì 5 ảnh. Việc tối ưu quá mức theo tập calib nhỏ hẹp sẽ dẫn tới xóa nhầm ảnh hợp lệ trên dữ liệu thật.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
- **Chứng minh gate chặn đúng:** Khi chạy thử nghiệm không có người duyệt bằng lệnh:
  `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`
  Do ngân sách người duyệt bằng 0, 6 ảnh thiếu file nhãn (`MISSING_LABEL_FILE`) không được ai gán nhãn nên vẫn còn nguyên lỗi bắt buộc (ERROR). Cửa chặn lập tức kết luận **REWORK**, trả về mã thoát **20**, và pipeline kiên quyết **không xuất** dataset sang `work/exp_nb`.
- **Điểm mù pipeline không bắt được:**
  1. *Vật thể che khuất mà cả hai model cùng sai:* Nếu một chiếc xe tải bị che khuất phần đuôi khiến người gán nhãn gán thành `car` và model checker (yolo26l) cũng nhận định là `car`, sai lệch ngữ nghĩa này sẽ không bị gắn cờ `POSSIBLE_WRONG_CLASS`.
  2. *Box vẽ lỏng nhẹ:* Box có độ lỏng vừa phải nhưng IoU với dự đoán model vẫn đạt >= 0.50 (`iou_loose`), checker sẽ coi là đã khớp và không gắn cờ `POSSIBLE_LOOSE_BOX`.
  3. *Ảnh trùng lặp bị biến dạng phi tuyến tính hoặc xoay/lật góc lớn:* Perceptual hash (pHash) truyền thống dựa trên biến đổi cosine 2D có thể không nhận diện được ảnh nếu bị cắt cúp sâu (>50%) hoặc méo hình học mạnh.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy: Bước train so sánh sẽ được thực hiện trên Google Colab với GPU T4 theo đúng phân định quy trình tại `docs/COLAB.md`.

## 6. Quyết định và ghi chú AI
- **Quyết định cuối:** Chấp thuận **RELEASE** bộ dữ liệu `submission/export/clean_v1`. Dữ liệu đã sạch lỗi bắt buộc, độ tin cậy cao và sẵn sàng đưa vào huấn luyện mô hình.
- **Ghi chú AI:**
  - *Prompt chính:* "Phân tích docstring và yêu cầu kỹ thuật của 4 module label_format, gate, autofix, second_opinion; triển khai code tuân thủ phương pháp TDD đảm bảo bao phủ toàn bộ các edge case trong bộ test."
  - *Cách kiểm code AI:* Chạy toàn bộ 79 bài test trong suite `pytest -q`, đối chiếu mã băm SHA-1 của các tệp quy định trong `_protected.json`, thực hiện sweep tham số để chọn ngưỡng và chạy kiểm thử e2e pipeline với các cấu hình ngân sách khác nhau.
  - *Một lần AI sai:* Trong kiểm tra đường dẫn file hệ điều hành Windows, việc so sánh chuỗi đường dẫn trực tiếp chứa dấu gạch chéo ngược `\` với danh sách file được bảo vệ dùng dấu `/` trong `scripts/check_submission.py` dẫn đến việc nhận diện nhầm file, đã được phát hiện và chuẩn hóa bằng `.as_posix()`.
