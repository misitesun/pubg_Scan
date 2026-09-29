"""训练素材可落盘；此模块不被屏幕推理工作线程调用。"""
from pathlib import Path
import hashlib
import json
import random
import shutil
import uuid

import cv2
import numpy as np


def load_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{Path(path).name}")
    return image


def save_image(path, frame):
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("图片编码失败。")
    data.tofile(str(path))


def source_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "samples").mkdir(exist_ok=True)
        self.manifest = self.root / "project.json"
        if self.manifest.exists():
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("不支持的项目版本。")
            self.samples = data["samples"]
        else:
            self.samples = []
            self.save()

    def save(self):
        temporary = self.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "samples": self.samples},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.manifest)

    def image_path(self, sample):
        path = (self.root / "samples" / sample["image"]).resolve()
        if path.parent != (self.root / "samples").resolve():
            raise ValueError("项目中的图片路径无效。")
        return path

    def _append(self, frame, group):
        name = uuid.uuid4().hex + ".jpg"
        save_image(self.root / "samples" / name, frame)
        self.samples.append({"image": name, "group": group, "reviewed": False, "boxes": []})

    def import_images(self, paths, cancelled=lambda: False):
        # 同一批图片归为一组，避免相邻画面分散到训练集和验证集。
        group = "images-" + uuid.uuid4().hex
        count = 0
        existing = {sample.get("source_hash") for sample in self.samples}
        for path in paths:
            if cancelled():
                break
            fingerprint = source_digest(path)
            if fingerprint in existing:
                continue
            self._append(load_image(path), group)
            self.samples[-1]["source_hash"] = fingerprint
            existing.add(fingerprint)
            count += 1
        self.save()
        return count

    def extract_video(self, path, interval=2.0, limit=500, cancelled=lambda: False):
        if interval <= 0 or limit < 1:
            raise ValueError("抽帧间隔和数量必须为正数。")
        group = "video-" + source_digest(path)
        if any(sample["group"] == group for sample in self.samples):
            raise ValueError("这段录像已经导入，请勿重复抽帧造成训练/验证数据重复。")
        capture = cv2.VideoCapture(str(path))
        count = 0
        try:
            if not capture.isOpened():
                raise ValueError("无法打开录像，建议使用 MP4（H.264）格式。")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("无法读取录像帧率。")
            step = max(1, round(fps * interval))
            index = 0
            # 顺序解码，避免某些编码器 seek 返回重复帧。
            while count < limit and not cancelled():
                ok = capture.grab()
                if not ok:
                    break
                if index % step == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        self._append(frame, group)
                        count += 1
                index += 1
        finally:
            capture.release()
            self.save()
        return count

    def annotate(self, index, boxes, roles=None):
        from .identity import ROLES
        if roles is not None and (len(roles)!=len(boxes) or any(r not in ROLES for r in roles)):
            raise ValueError('每个人物都需要有效的身份分类。')
        frame = load_image(self.image_path(self.samples[index]))
        sample=self.samples[index]
        ignored_boxes={tuple(sample['boxes'][i]) for i in sample.get('ignored_box_indices',[])
                       if isinstance(i,int) and 0<=i<len(sample['boxes'])}
        ignored_reasons={tuple(sample['boxes'][i]):sample.get('ignored_box_reasons',{}).get(str(i),'用户指定忽略')
                         for i in sample.get('ignored_box_indices',[]) if isinstance(i,int) and 0<=i<len(sample['boxes'])}
        height, width = frame.shape[:2]
        cleaned = []
        cleaned_roles=[]
        for pos,(x1, y1, x2, y2) in enumerate(boxes):
            if not np.isfinite([x1,y1,x2,y2]).all():
                raise ValueError('人物框不能包含无效坐标。')
            x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                cleaned.append([x1, y1, x2, y2])
                cleaned_roles.append(roles[pos] if roles is not None else 'unknown')
        self.samples[index]["boxes"] = cleaned
        self.samples[index]['roles']=cleaned_roles
        self.samples[index]['ignored_box_indices']=[i for i,b in enumerate(cleaned) if tuple(b) in ignored_boxes]
        self.samples[index]['ignored_box_reasons']={str(i):ignored_reasons[tuple(b)] for i,b in enumerate(cleaned) if tuple(b) in ignored_boxes}
        self.samples[index]['roles_reviewed']=roles is not None or not cleaned
        self.samples[index]["reviewed"] = True
        self.save()

    def import_identity_review(self, path):
        """导入离线复核页的用户确认；先校验整份记录，再原子保存。"""
        from .identity import ROLES
        record=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(record,dict) or record.get('format')!='person-vision-review-v1':
            raise ValueError('不是本工具导出的身份复核结果。')
        entries=record.get('samples')
        if not isinstance(entries,list):raise ValueError('复核记录缺少样本列表。')
        by_image={s['image']:s for s in self.samples}
        updates=[];seen=set()
        for entry in entries:
            if not isinstance(entry,dict) or entry.get('status') not in ('confirmed','needs_correction','pending'):
                raise ValueError('复核状态无效。')
            name=entry.get('image')
            if name not in by_image or name in seen:raise ValueError('复核样本重复或不属于当前项目。')
            seen.add(name);s=by_image[name]
            if source_digest(self.image_path(s))!=entry.get('image_sha256') or s['boxes']!=entry.get('boxes'):
                raise ValueError('原图或人物框已变化，请重新生成复核页，避免覆盖新标注。')
            roles=entry.get('roles')
            if not isinstance(roles,list) or len(roles)!=len(s['boxes']) or any(r not in ROLES for r in roles):
                raise ValueError('复核身份与人物框不一致。')
            updates.append((s,entry))
        for s,entry in updates:
            s['user_confirmed']=entry['status']=='confirmed'
            s['roles_reviewed']=entry['status']=='confirmed'
            s['roles']=entry['roles']
            s['user_review_note']=str(entry.get('note',''))[:2000]
            s['user_review_status']=entry['status']
            if s['user_confirmed']:
                s['reviewed']=True
                s['reviewer']='user_review_sheet'
        self.save()
        return sum(s['user_confirmed'] for s,_ in updates)

    def export_dataset(self, output, validation_ratio=0.2, seed=42, identities=False, user_confirmed_only=False):
        from .identity import ROLES
        reviewed = [s for s in self.samples if s["reviewed"] and (not identities or s.get('roles_reviewed',False))
                    and (not user_confirmed_only or s.get('user_confirmed',False))]
        for s in reviewed:
            ignored=s.get('ignored_box_indices',[])
            if not isinstance(ignored,list) or any(type(i)!=int or not 0<=i<len(s['boxes']) for i in ignored):
                raise ValueError('忽略标注的编号无效，请重新核对。')
        if identities and any(len(s.get('roles',[]))!=len(s['boxes']) or any(r not in ROLES for r in s.get('roles',[])) for s in reviewed):
            raise ValueError('身份标签与人物框不一致，请重新复核。')
        def has_targets(sample):
            return any(i not in sample.get('ignored_box_indices',[]) for i in range(len(sample['boxes'])))
        if not reviewed or not any(has_targets(s) for s in reviewed):
            raise ValueError("请先标注人物。无人物的图片也需要手动确认。")
        groups = sorted({s["group"] for s in reviewed})
        if len(groups) < 2:
            raise ValueError("至少需要两个素材组（不同录像，或分批导入的图片），用于独立验证。")
        if not 0 < validation_ratio < 1:
            raise ValueError("验证集比例应在 0 和 1 之间。")
        # 找到两边都有人物的分组，避免验证集只有背景图。
        randomizer = random.Random(seed)
        validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_ratio)))
        for _ in range(100):
            randomizer.shuffle(groups)
            validation_groups = set(groups[:validation_count])
            validation = [s for s in reviewed if s["group"] in validation_groups]
            training = [s for s in reviewed if s["group"] not in validation_groups]
            if any(has_targets(s) for s in validation) and any(has_targets(s) for s in training):
                break
        else:
            raise ValueError("训练集与验证集都必须有人物；请标注至少两个素材组中的人物。")
        output = Path(output).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("导出目录必须为空，避免旧标签混入新数据集。")
        output.mkdir(parents=True, exist_ok=True)
        for split, samples in (("train", training), ("val", validation)):
            images, labels = output / "images" / split, output / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for sample in samples:
                frame = load_image(self.image_path(sample))
                height, width = frame.shape[:2]
                shutil.copy2(self.image_path(sample), images / sample["image"])
                lines = []
                for pos,(x1, y1, x2, y2) in enumerate(sample["boxes"]):
                    if pos in sample.get('ignored_box_indices',[]):continue
                    cid=ROLES.index(sample['roles'][pos]) if identities else 0
                    lines.append(f"{cid} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} "
                                 f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                (labels / (Path(sample["image"]).stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        yaml = output / "dataset.yaml"
        # JSON 双引号字符串也属于合法 YAML 字符串，兼容中文和 Windows 路径。
        names=''.join(f'  {i}: {r}\n' for i,r in enumerate(ROLES)) if identities else '  0: person\n'
        # 不写入本机绝对 path；Ultralytics 以 dataset.yaml 所在目录解析素材。
        # 整份数据集复制到 Windows 后不需要手动改 Mac 路径。
        yaml.write_text("train: images/train\nval: images/val\nnames:\n"+names, encoding="utf-8")
        (output / "split.json").write_text(json.dumps({
            "seed": seed, "train_groups": sorted(set(groups) - validation_groups),
            "val_groups": sorted(validation_groups), "train_images": len(training),
            "val_images": len(validation)}, indent=2), encoding="utf-8")
        return yaml, len(training), len(validation)
