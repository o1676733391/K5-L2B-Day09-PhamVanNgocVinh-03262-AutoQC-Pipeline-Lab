"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    overflow_tol = cfg["label_format"]["overflow_tol"]
    min_box_px = cfg["label_format"]["min_box_px"]
    max_box_frac = cfg["label_format"]["max_box_frac"]
    dup_box_iou = cfg["label_format"]["dup_box_iou"]
    n_classes = len(ds.classes)

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        lines = ds.read_label_lines(stem)
        if lines is None:
            continue

        im_size = ds.image_size(stem)
        valid_boxes = []  # list of (line_no, class_id, (cx, cy, w, h))

        for line_no, line_text in lines:
            try:
                c, cx, cy, w, h = parse_yolo_line(line_text)
            except ValueError as e:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, f"dòng nhãn sai định dạng: {e}", line=line_no))
                continue

            if c < 0 or c >= n_classes:
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp {c} ngoài khoảng 0..{n_classes-1}", line=line_no))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"kích thước box <= 0 (w={w}, h={h})", line=line_no))
                continue

            # Hình học box
            left = cx - w / 2.0
            right = cx + w / 2.0
            top = cy - h / 2.0
            bottom = cy + h / 2.0

            overflow = max(0.0, -left, right - 1.0, -top, bottom - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem, f"mép box tràn ngoài ảnh {overflow:.4f}", line=line_no, extra={"overflow": overflow}))

            # Kiểm tra trùng box với dòng trước đó (cùng lớp)
            for prev_line_no, prev_c, prev_box in valid_boxes:
                if prev_c == c:
                    iou = iou_xywh((cx, cy, w, h), prev_box)
                    if iou > dup_box_iou:
                        issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, f"trùng box với dòng {prev_line_no} (IoU={iou:.3f})", line=line_no, extra={"line_prev": prev_line_no, "iou": iou}))
                        break

            valid_boxes.append((line_no, c, (cx, cy, w, h)))

            # Cảnh báo box quá nhỏ / quá lớn
            if im_size is not None:
                W, H = im_size
                short_side = min(w * W, h * H)
                if short_side < min_box_px:
                    issues.append(Issue("label_format", WARN, "BOX_TINY", stem, f"cạnh ngắn {short_side:.1f}px < {min_box_px}px", line=line_no, extra={"min_px": short_side}))

            area_frac = w * h
            if area_frac > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, f"diện tích box {area_frac:.3f} > {max_box_frac}", line=line_no, extra={"frac": area_frac}))

    return issues
