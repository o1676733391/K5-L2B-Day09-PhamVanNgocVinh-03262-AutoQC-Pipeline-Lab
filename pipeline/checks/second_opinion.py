"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so_cfg = cfg["second_opinion"]
    conf_th = so_cfg["conf"]
    min_area_px = so_cfg["min_area_px"]
    fp_conf = so_cfg["fp_conf"]
    iou_missing = so_cfg["iou_missing"]
    iou_match = so_cfg["iou_match"]
    iou_loose = so_cfg["iou_loose"]
    n_classes = len(ds.classes)

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds:
            continue

        raw_lines = ds.read_label_lines(stem)
        if raw_lines is None:
            continue

        # Các dòng nhãn hợp lệ, kèm số dòng gốc
        valid_labels = []  # list of (line_no, class_id, (cx, cy, w, h))
        for line_no, line_text in raw_lines:
            try:
                c, cx, cy, w, h = parse_yolo_line(line_text)
            except ValueError:
                continue
            if 0 <= c < n_classes:
                valid_labels.append((line_no, c, (cx, cy, w, h)))

        im_size = ds.image_size(stem)
        W, H = im_size if im_size is not None else (640, 640)

        pred_rows = preds.get(stem, [])

        # P: dự đoán có conf >= conf_th và diện tích px >= min_area_px
        P = []
        for p in pred_rows:
            p_cls, p_conf, p_cx, p_cy, p_w, p_h = p[0], p[1], p[2], p[3], p[4], p[5]
            if p_conf >= conf_th and (p_w * W * p_h * H) >= min_area_px:
                P.append((int(p_cls), float(p_conf), (p_cx, p_cy, p_w, p_h)))

        # Kiểm tra từng dự đoán r trong P
        for p_cls, p_conf, p_box in P:
            cands = []
            for l_no, l_cls, l_box in valid_labels:
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    cands.append((l_cls, iou))

            if not cands:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_MISSING", stem,
                    f"model dự đoán lớp {p_cls} (conf {p_conf:.2f}) nhưng nhãn thiếu box",
                    line=None, extra={"conf": p_conf, "pred_cls": p_cls}
                ))
            else:
                same_cands = [c for c in cands if c[0] == p_cls]
                if same_cands:
                    best_same_iou = max(c[1] for c in same_cands)
                    if best_same_iou < iou_loose:
                        issues.append(Issue(
                            "second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                            f"box nhãn lỏng/lệch so với model: IoU {best_same_iou:.2f} < {iou_loose}",
                            line=None, extra={"conf": p_conf, "iou": round(best_same_iou, 4), "pred_cls": p_cls}
                        ))
                    else:
                        pass  # Đã được phủ bởi nhãn đúng lớp
                else:
                    diff_cands = cands
                    best_diff_iou = max(c[1] for c in diff_cands)
                    if best_diff_iou >= iou_match:
                        issues.append(Issue(
                            "second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                            f"model dự đoán lớp {p_cls} nhưng nhãn gán lớp khác (IoU {best_diff_iou:.2f})",
                            line=None, extra={"conf": p_conf, "iou": round(best_diff_iou, 4), "pred_cls": p_cls}
                        ))
                    else:
                        pass  # Mơ hồ (iou trong [iou_missing, iou_match)): không cờ gì

        # Kiểm tra FALSE_POS: với mỗi dòng nhãn hợp lệ
        fp_preds = [p for p in pred_rows if p[1] >= fp_conf]
        for l_no, l_cls, l_box in valid_labels:
            matched = any(iou_xywh(l_box, (p[2], p[3], p[4], p[5])) >= iou_missing for p in fp_preds)
            if not matched:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                    f"box nhãn lớp {l_cls} không khớp dự đoán nào của model",
                    line=l_no, extra={"class": l_cls}
                ))

    return issues
