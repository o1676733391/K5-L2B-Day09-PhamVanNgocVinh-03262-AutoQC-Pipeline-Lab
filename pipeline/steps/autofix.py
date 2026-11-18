"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |
"""
from collections import defaultdict
from pathlib import Path

import yaml

from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
IMAGE_REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def fix(ds, issues, cfg):
    error_issues = [i for i in issues if i.severity == ERROR]
    if not error_issues:
        return []

    fixes = []
    removed_images = set()

    # 1. BAD_CLASS_NAME: sửa names trong data.yaml
    bad_yaml_issues = [i for i in error_issues if i.code == "BAD_CLASS_NAME"]
    if bad_yaml_issues:
        old_data = {}
        if ds.yaml_path.exists():
            try:
                old_data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                old_data = {}
        old_names = old_data.get("names", "")
        old_data["names"] = list(cfg["classes"])
        ds.yaml_path.write_text(yaml.safe_dump(old_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append({
            "image": "data.yaml",
            "line": "",
            "action": "rename_classes",
            "before": str(old_names),
            "after": str(cfg["classes"]),
        })

    # 2. Xoá ảnh hỏng / trùng / rò rỉ: DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE
    for i in error_issues:
        if i.code.startswith("DUP_IMAGE_") or i.code in IMAGE_REMOVE_CODES:
            stem = i.image
            if stem not in removed_images and stem not in ("data.yaml", "<dataset>"):
                im_path = ds.image_path(stem)
                if im_path and im_path.exists():
                    try:
                        im_path.unlink()
                    except Exception:
                        pass
                lbl_path = ds.label_path(stem)
                if lbl_path.exists():
                    try:
                        lbl_path.unlink()
                    except Exception:
                        pass
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "remove_image",
                    "before": im_path.name if im_path else stem,
                    "after": "đã xoá",
                })
                removed_images.add(stem)

    # 3. MISSING_LABEL_FILE: máy không bịa nhãn, ghi needs_label
    for i in error_issues:
        if i.code == "MISSING_LABEL_FILE":
            stem = i.image
            if stem not in removed_images:
                fixes.append({
                    "image": stem,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán",
                })

    # 4. Sửa trên dòng file nhãn: gom theo stem để tránh lệch số dòng
    line_issues_by_stem = defaultdict(lambda: defaultdict(list))
    for i in error_issues:
        if i.line is not None and i.image not in removed_images and i.image not in ("data.yaml", "<dataset>"):
            line_issues_by_stem[i.image][i.line].append(i)

    for stem, line_map in line_issues_by_stem.items():
        lbl_path = ds.label_path(stem)
        if not lbl_path.exists():
            continue
        raw_lines = lbl_path.read_text(encoding="utf-8", errors="replace").splitlines()

        # Xác định thao tác cho từng dòng cần xử lý (sắp xếp theo line_no)
        file_fixes = []
        lines_to_drop = set()
        lines_to_replace = {}  # line_no -> new_line_str

        for line_no in sorted(line_map.keys()):
            if line_no < 1 or line_no > len(raw_lines):
                continue
            orig_line = raw_lines[line_no - 1]
            iss_list = line_map[line_no]
            codes = {it.code for it in iss_list}

            # Quy tắc: xoá dòng thắng cắt box
            if codes & DROP_CODES:
                lines_to_drop.add(line_no)
                file_fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": orig_line,
                    "after": "",
                })
            elif "BOX_OUT_OF_IMAGE" in codes:
                try:
                    c, cx, cy, w, h = parse_yolo_line(orig_line)
                    x1, x2 = cx - w / 2.0, cx + w / 2.0
                    y1, y2 = cy - h / 2.0, cy + h / 2.0
                    nx1, nx2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
                    ny1, ny2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
                    nw, nh = nx2 - nx1, ny2 - ny1
                    if nw <= 0.0 or nh <= 0.0:
                        # Box nằm hẳn ngoài ảnh -> xoá dòng
                        lines_to_drop.add(line_no)
                        file_fixes.append({
                            "image": stem,
                            "line": line_no,
                            "action": "drop_line",
                            "before": orig_line,
                            "after": "",
                        })
                    else:
                        ncx = nx1 + nw / 2.0
                        ncy = ny1 + nh / 2.0
                        new_line_str = f"{c} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                        lines_to_replace[line_no] = new_line_str
                        file_fixes.append({
                            "image": stem,
                            "line": line_no,
                            "action": "clip_box",
                            "before": orig_line,
                            "after": new_line_str,
                        })
                except ValueError:
                    lines_to_drop.add(line_no)
                    file_fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "drop_line",
                        "before": orig_line,
                        "after": "",
                    })

        # Ghi lại file nhãn
        new_lines = []
        for idx, line_text in enumerate(raw_lines):
            cur_line = idx + 1
            if cur_line in lines_to_drop:
                continue
            elif cur_line in lines_to_replace:
                new_lines.append(lines_to_replace[cur_line])
            else:
                new_lines.append(line_text)

        lbl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        fixes.extend(file_fixes)

    return fixes
