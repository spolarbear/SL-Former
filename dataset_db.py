# dataset_db.py
"""
基于 PostgreSQL 的数据集 (PyTorch Dataset) - 解耦架构

直接从数据库读取:
    - structures 独立字段 -> frame_features (杆系结构化特征, 由 db_generate_samples 生成)
    - ground_motions 波形 -> motion (输入加速度时程)
    - samples 结果 -> disp (屋顶位移时程)

用法:
    from dataset_db import SLFDbDataset
    ds = SLFDbDataset(n_stories=None, pga_range=None)   # 支持条件过滤
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from db_manager import SLFDatabase, _bytes_to_arr


class SLFDbDataset(Dataset):
    def __init__(self, n_stories=None, pga_range=None, db=None, limit=None,
                 use_frame_feature=True):
        self.db = db or SLFDatabase()
        self.use_frame_feature = use_frame_feature
        self.rows = self.db.query_samples(n_stories=n_stories, pga_range=pga_range,
                                          limit=limit)
        self.seq_len = None
        self._items = []
        for r in self.rows:
            sid = r['sample_id']
            resp = self.db.get_sample(sid)
            if resp is None or resp.get('roof_disp') is None:
                continue
            motion = self._get_motion(resp['gm_id'])
            struct = self.db.get_structure(resp['struct_id'])
            ff = self._frame_features(struct) if struct is not None else None
            self._items.append({
                'frame_features': ff,
                'disp': resp['roof_disp'],
                'motion': motion,
                'params': self._params_from_struct(struct) if struct else None,
                'height': np.float32(struct['total_height']) if struct else np.float32(0.0),
                'E_avg': np.float32(30.0),
            })
            if self.seq_len is None:
                self.seq_len = len(resp['roof_disp'])

    # ---------- 特征 ----------
    @staticmethod
    def _params_from_struct(st):
        """从结构独立字段重建 21 维 params (结构条件注入).

        p[0:8]  : 层数/跨数X/Y/跨度X/Y/层高/平均节点质量/阻尼
        p[8]    : 平面形状 ID (0=rect, 1=T, 2=L, 3=C, 4=U)
        p[9:21] : 每层楼面平均荷载 (kPa), 每层一个值, 最多 12 层, 不足补 0
        """
        if st is None:
            return None
        from config import Config as _C
        from generate_frames import shape_to_id
        dim = int(getattr(_C, 'PARAMS_DIM', 21))
        off = int(getattr(_C, 'PARAMS_FLOOR_LOAD_OFFSET', 9))
        sidx = int(getattr(_C, 'PARAMS_SHAPE_IDX', 8))
        max_fl = int(getattr(_C, 'PARAMS_MAX_FLOORS', 12))
        masses = st.get('floor_masses') or []
        loads = st.get('floor_loads') or []
        p = np.zeros(dim, dtype=np.float32)
        p[0] = st['num_stories']; p[1] = st['num_bays_x']; p[2] = st['num_bays_y']
        p[3] = st['span_x']; p[4] = st['span_y']; p[5] = st['story_height']
        p[6] = float(np.mean(masses)) if masses else 0.0
        p[7] = 0.05
        p[sidx] = shape_to_id(st.get('plane_shape') or 'rect')
        # 每层楼面荷载 (kPa) -> p[off:off+min(ns, max_fl)]
        ns = int(st['num_stories'])
        if loads:
            for i, v in enumerate(loads[:max_fl]):
                p[off + i] = float(v)
        return p

    @staticmethod
    def _frame_features(st):
        """从结构独立字段重建杆系结构化特征 (frame_feature_encoder)"""
        if st is None:
            return None
        # 重建 frame_params dict 供 encoder 使用
        from generate_frames import generate_fixed_frame
        ns = int(st['num_stories']); nx = int(st['num_bays_x']); ny = int(st['num_bays_y'])
        sx = float(st['span_x']); sy = float(st['span_y']); sh = float(st['story_height'])
        plane_shape = str(st.get('plane_shape') or 'rect').lower()
        ms = max(sx, sy)
        bh = max(0.4, min(ms/12, 0.8)); bh = round(bh / 0.2) * 0.2   # 200mm
        bw = max(0.2, min(bh/2.5, 0.5)); bw = round(bw / 0.2) * 0.2  # 200mm
        frame = generate_fixed_frame(ns, nx, ny, sx, sy, sh, 0.6, bw, bh,
                                     plane_shape=plane_shape)
        fm = np.asarray(st.get('floor_masses') or [], dtype=np.float64)
        from frame_feature_encoder import extract_frame_features
        feat, _ = extract_frame_features(frame, fm if len(fm) else None)
        return feat

    def _get_motion(self, gm_id):
        gm = self.db.get_ground_motion(gm_id)
        if gm is None or gm.get('motion') is None:
            return None
        return gm['motion']

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        from config import Config as _C
        pdim = int(getattr(_C, 'PARAMS_DIM', 20))
        it = self._items[idx]
        sample = {
            'octree_features': torch.zeros(1),
            'disp': torch.FloatTensor(np.ascontiguousarray(it['disp'])),
            'height': torch.FloatTensor([it['height']]),
            'E_avg': torch.FloatTensor([it['E_avg']]),
            'params': torch.FloatTensor(it['params']) if it['params'] is not None else torch.zeros(pdim),
        }
        if it['frame_features'] is not None and self.use_frame_feature:
            sample['frame_features'] = torch.FloatTensor(
                np.ascontiguousarray(it['frame_features']))
        if it['motion'] is not None:
            sample['motion'] = torch.FloatTensor(np.ascontiguousarray(it['motion']))
        return sample


if __name__ == '__main__':
    ds = SLFDbDataset(limit=20)
    print("数据集样本数:", len(ds))
    if len(ds):
        s = ds[0]
        print("keys:", list(s.keys()))
        print("disp:", tuple(s['disp'].shape))
        if 'frame_features' in s:
            print("frame_features:", tuple(s['frame_features'].shape))
        if 'motion' in s:
            print("motion:", tuple(s['motion'].shape))
