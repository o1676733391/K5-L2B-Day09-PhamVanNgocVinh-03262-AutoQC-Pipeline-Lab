"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stem_set = set(image_stems)
    rev_set = set(reviewed) if reviewed else set()

    n_errors = sum(1 for i in issues if i.severity == ERROR)

    warn_images = {
        i.image for i in issues
        if i.severity == WARN and i.image in stem_set and i.image not in rev_set
    }

    n_images = len(image_stems)
    warn_ratio = round(len(warn_images) / n_images, 4) if n_images > 0 else 0.0

    hold_thresh = cfg["gate"]["warn_ratio_hold"]

    if n_errors > 0:
        decision = "Rework"
        reasons = [f"Còn {n_errors} lỗi bắt buộc (ERROR) chưa được khắc phục."]
    elif warn_ratio >= hold_thresh:
        decision = "Hold"
        reasons = [
            f"Tỉ lệ ảnh có cảnh báo {warn_ratio:.4f} ({len(warn_images)}/{n_images}) "
            f">= ngưỡng cho phép {hold_thresh:.4f}."
        ]
    else:
        decision = "Release"
        reasons = [
            f"Đạt chuẩn phát hành: 0 lỗi bắt buộc, tỉ lệ cảnh báo {warn_ratio:.4f} "
            f"({len(warn_images)}/{n_images}) < ngưỡng {hold_thresh:.4f}."
        ]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": len(warn_images),
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
