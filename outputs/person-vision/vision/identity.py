"""身份过滤实验：像素队标模板、短时跟踪和未知身份抑制，不读取游戏状态。"""
from dataclasses import dataclass, replace
from pathlib import Path
import cv2
import numpy as np

ROLES = ('self', 'opponent', 'teammate', 'unknown')
ROLE_LABELS = {'self': '自身', 'opponent': '对手', 'teammate': '队友', 'unknown': '不确定'}
ROLE_COLORS = {'self': '#5aa9ff', 'opponent': '#ff6060', 'teammate': '#52e4b5', 'unknown': '#ffc56b'}


def overlap(a, b):
    intersection = max(0, min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    union = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection
    return intersection/max(union,1e-6)


class TeamMarkers:
    """匹配用户确认的队标小图；不能将远处标记绑定到另一个人物。"""
    def __init__(self, directory=None, threshold=.94):
        self.templates=[]
        self.threshold=threshold
        if directory:
            from .dataset import load_image
            root=Path(directory)
            if not root.is_dir():
                raise ValueError('队标模板目录不存在。')
            paths=sorted(p for p in root.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
            if not paths or len(paths)>32:
                raise ValueError('队标目录应包含 1–32 张已确认的 PNG/JPG 队标小图。')
            for path in paths:
                template=load_image(path)
                if not 4<=min(template.shape[:2]) or max(template.shape[:2])>128 or template.std()<5:
                    raise ValueError(f'队标模板应为 4–128 像素的清晰小图，且不能是纯色：{path.name}')
                self.templates.append(template)

    def associate(self, frame, boxes):
        if not self.templates:
            return set(),set()
        h,w=frame.shape[:2]
        matches=[]
        for i,b in enumerate(boxes):
            bw,bh=b.x2-b.x1,b.y2-b.y1
            # 标记必须贴近该人物上方；不扫描 HUD、小地图或整幅画面中的名字。
            x1=max(0,int(b.x1-.25*bw));x2=min(w,int(b.x2+.25*bw))
            y1=max(0,int(b.y1-min(.4*bh,64*h/576)));y2=min(h,int(b.y1+.12*bh))
            roi=frame[y1:y2,x1:x2]
            if roi.size==0:continue
            best=(0,None)
            for template in self.templates:
                for scale in (.8,1.,1.2):
                    tw,th=round(template.shape[1]*scale),round(template.shape[0]*scale)
                    if min(tw,th)<4 or roi.shape[0]<th or roi.shape[1]<tw:continue
                    resized=cv2.resize(template,(tw,th))
                    scores=cv2.matchTemplate(roi,resized,cv2.TM_CCOEFF_NORMED)
                    _,score,_,point=cv2.minMaxLoc(scores)
                    if np.isfinite(score) and score>best[0]:
                        best=(score,(x1+point[0]+tw/2,y1+point[1]+th/2))
            if best[0]>=self.threshold:
                matches.append((i,best[1]))
        confirmed=set();ambiguous=set()
        for i,point in matches:
            contenders=[j for j,p in matches if np.linalg.norm(np.subtract(point,p))<8]
            if len(contenders)==1:confirmed.add(i)
            else:ambiguous.update(contenders)
        return confirmed,ambiguous


@dataclass
class Track:
    box: object
    time: float
    role: str='unknown'
    streak: int=0
    excluded: bool=False


class IdentityFilter:
    def __init__(self, markers=None, minimum_confidence=.5, minimum_frames=3):
        self.markers=markers or TeamMarkers()
        self.minimum_confidence=minimum_confidence
        self.minimum_frames=minimum_frames
        self.tracks={}
        self.next_id=1
        self.previous=None
        self.decisions=[]

    def update(self, frame, boxes, now):
        # 明显切镜或大幅转向后清空身份，避免把旧人物身份带入新场景。
        tiny=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(32,18)).astype(np.float32)
        if self.previous is not None and np.mean(np.abs(tiny-self.previous))>35:
            self.tracks.clear()
        self.previous=tiny
        self.tracks={k:t for k,t in self.tracks.items() if 0<=now-t.time<=.5}
        # 全局贪心一对一匹配，不允许同一轨迹绑定两个人。
        candidates=sorted(((overlap(b,t.box),i,k) for i,b in enumerate(boxes) for k,t in self.tracks.items()),reverse=True)
        used_tracks=set();assignments={}
        for score,i,k in candidates:
            if score>=.35 and i not in assignments and k not in used_tracks:
                assignments[i]=k;used_tracks.add(k)
        team,ambiguous=self.markers.associate(frame,boxes)
        visible=[];decisions=[]
        for i,b in enumerate(boxes):
            k=assignments.get(i)
            if k is None:
                k=self.next_id;self.next_id+=1;self.tracks[k]=Track(b,now)
            track=self.tracks[k]
            role=b.role;reason='身份模型候选'
            if i in ambiguous:
                role='unknown';reason='队标与多个人物关联存在歧义'
            elif i in team:
                role='teammate';reason='人物上方匹配已确认的队标模板'
            elif role=='teammate':
                role='unknown';reason='队友候选缺少对应队标，保守排除'
            elif b.confidence<self.minimum_confidence:
                role='unknown';reason='身份候选置信度不足'
            if role in ('self','teammate'):
                track.excluded=True
            track.streak=track.streak+1 if role==track.role else 1
            track.box=b;track.time=now;track.role=role
            # 曾为自身 / 队友的连续轨迹不会因一次身份翻转成为对手。
            draw=(role=='opponent' and track.streak>=self.minimum_frames and not track.excluded)
            if draw:visible.append(b)
            decisions.append({'track_id':k,'role':role if role!='opponent' or draw else 'unknown',
                              'confidence':b.confidence,'draw_box':draw,'evidence':reason,
                              'bbox':[b.x1,b.y1,b.x2,b.y2]})
        self.decisions=decisions
        return visible
