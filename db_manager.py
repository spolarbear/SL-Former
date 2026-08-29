# db_manager.py
"""
SLF 样本集数据库管理器 (PostgreSQL) - 解耦架构

设计原则:
    数据完全解耦, 不用 JSON 存核心数据, 多设独立字段便于检索。
    三张主表:
        1. ground_motions : 地震波时序 + 波id (独立字段: pga/dt/duration/n_steps/source/wave_name)
        2. structures     : 建筑杆系模型 (独立字段: 层数/跨数/跨度/层高/梁柱截面等)
        3. samples        : 模型id + 地震波id + 计算结果 (独立字段: target_pga/applied_pga/结果统计)
    查重机制:
        - structures 唯一键 (num_stories, num_bays_x, num_bays_y, span_x, span_y,
                            story_height, beam_width, beam_height, slab_thickness, floor_load)
        - samples 唯一键 (struct_id, gm_id, target_pga)

用法:
    from db_manager import SLFDatabase
    db = SLFDatabase()
    db.init_schema()                    # 建表 (幂等)
    gid = db.get_or_create_ground_motion(...)  # 波查重
    sid = db.get_or_create_structure(...)      # 结构查重
    sample = db.get_or_create_sample(...)      # 样本查重

依赖: psycopg2-binary
"""
import os
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor


DEFAULT_DSN = dict(
    host=os.environ.get('SLF_PG_HOST', 'localhost'),
    port=int(os.environ.get('SLF_PG_PORT', '5432')),
    dbname=os.environ.get('SLF_PG_DB', 'slf_sim'),
    user=os.environ.get('SLF_PG_USER', 'postgres'),
    password=os.environ.get('SLF_PG_PASSWORD', 'pgsql'),  # 本地默认密码 (db_setup 重置)
)

# ============================================================
# v3.0 分支: 表名后缀
#   - 默认 '_v3' (操作 ground_motions_v3 / structures_v3 / samples_v3)
#   - 旧表 (无后缀) 完全保留, 不再被 v3 代码触碰
#   - 可用环境变量 SLF_PG_SUFFIX 整体切换 (如 'v4' -> *_v4)
# ============================================================
TABLE_SUFFIX = os.environ.get('SLF_PG_SUFFIX', 'v3')
GM_TABLE  = f"ground_motions_{TABLE_SUFFIX}"
ST_TABLE  = f"structures_{TABLE_SUFFIX}"
SP_TABLE  = f"samples_{TABLE_SUFFIX}"

# ============================================================
# 数组 <-> BYTEA 序列化
# ============================================================
def _arr_to_bytes(a):
    return np.asarray(a, dtype=np.float32).tobytes()


def _bytes_to_arr(b, shape=None):
    a = np.frombuffer(b, dtype=np.float32).copy()  # 可写副本 (torch 需要)
    if shape is not None:
        return a.reshape(shape)
    return a


# ============================================================
# Schema (解耦, 独立字段, 无 JSON 核心数据) — 表名带 v3 后缀
# ============================================================
def _schema_sql():
    """根据 TABLE_SUFFIX 生成带后缀的建表 SQL."""
    return f"""
-- 1. 地震波时序表 ({GM_TABLE})
CREATE TABLE IF NOT EXISTS {GM_TABLE} (
    gm_id       SERIAL PRIMARY KEY,
    wave_name   TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'synthetic',
    pga         DOUBLE PRECISION NOT NULL,
    dt          DOUBLE PRECISION NOT NULL DEFAULT 0.02,
    duration    DOUBLE PRECISION,
    n_steps     INTEGER,
    motion_bytes BYTEA,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (wave_name, source, pga)
);

-- 2. 建筑杆系模型表 ({ST_TABLE}, 全部独立字段)
CREATE TABLE IF NOT EXISTS {ST_TABLE} (
    struct_id       SERIAL PRIMARY KEY,
    num_stories     INTEGER NOT NULL,
    num_bays_x      INTEGER NOT NULL,
    num_bays_y      INTEGER NOT NULL,
    span_x          DOUBLE PRECISION NOT NULL,
    span_y          DOUBLE PRECISION NOT NULL,
    story_height    DOUBLE PRECISION NOT NULL,
    total_height    DOUBLE PRECISION NOT NULL,
    beam_width      DOUBLE PRECISION NOT NULL,
    beam_height     DOUBLE PRECISION NOT NULL,
    slab_thickness  DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    col_sections    TEXT,
    beam_sections   TEXT,
    floor_loads     TEXT,
    floor_masses    TEXT,
    total_mass_kg   DOUBLE PRECISION,
    n_columns       INTEGER,
    n_beams         INTEGER,
    plane_shape     TEXT NOT NULL DEFAULT 'rect',
    created_at      TIMESTAMPTZ DEFAULT now()
    -- 注: 不再设几何 UNIQUE 约束 (截面随机化后, 几何相同但截面不同的结构需共存;
    --      查重由应用层 get_or_create_structure 按几何+截面完成)
);

-- 3. 样本表 ({SP_TABLE}, 模型id + 波id + 结果)
CREATE TABLE IF NOT EXISTS {SP_TABLE} (
    sample_id       SERIAL PRIMARY KEY,
    struct_id       INTEGER NOT NULL REFERENCES {ST_TABLE}(struct_id),
    gm_id           INTEGER NOT NULL REFERENCES {GM_TABLE}(gm_id),
    target_pga      DOUBLE PRECISION NOT NULL,
    applied_pga     DOUBLE PRECISION,
    motion_scale    DOUBLE PRECISION,
    roof_disp_bytes BYTEA,
    disp_std        DOUBLE PRECISION,
    disp_peak       DOUBLE PRECISION,
    disp_final      DOUBLE PRECISION,
    n_steps         INTEGER,
    sim_status      TEXT DEFAULT 'pending',
    sim_time_s      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (struct_id, gm_id, target_pga)
);

CREATE INDEX IF NOT EXISTS idx_samples_struct ON {SP_TABLE}(struct_id);
CREATE INDEX IF NOT EXISTS idx_samples_gm     ON {SP_TABLE}(gm_id);
CREATE INDEX IF NOT EXISTS idx_samples_pga    ON {SP_TABLE}(target_pga);
CREATE INDEX IF NOT EXISTS idx_structures_params ON {ST_TABLE}(num_stories, num_bays_x, num_bays_y);
"""


SCHEMA_SQL = _schema_sql()


class SLFDatabase:
    """SLF 样本集数据库操作类 (解耦架构)"""

    def __init__(self, dsn=None, autocommit=True):
        self.dsn = dsn or dict(DEFAULT_DSN)
        self.conn = psycopg2.connect(**self.dsn)
        self.conn.autocommit = autocommit
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)

    # ------------------------------------------------------------
    # 建表 / 通用
    # ------------------------------------------------------------
    def init_schema(self):
        self.cur.execute(SCHEMA_SQL)
        # 迁移: 新表无 beam_sections 列则补充
        self.cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=%s AND column_name='beam_sections'", (ST_TABLE,))
        if self.cur.fetchone() is None:
            self.cur.execute(f"ALTER TABLE {ST_TABLE} ADD COLUMN beam_sections TEXT")
            print(f"[MIGRATE] {ST_TABLE} 表新增 beam_sections 列")
        # 迁移: 新表无 plane_shape 列则补充 (默认 rect 兼容旧数据)
        self.cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=%s AND column_name='plane_shape'", (ST_TABLE,))
        if self.cur.fetchone() is None:
            self.cur.execute(
                f"ALTER TABLE {ST_TABLE} ADD COLUMN plane_shape TEXT NOT NULL DEFAULT 'rect'")
            print(f"[MIGRATE] {ST_TABLE} 表新增 plane_shape 列")
        # 迁移: 删除几何 UNIQUE 约束 (截面随机化后需允许同几何不同截面共存)
        self.cur.execute(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid=%s::regclass AND contype='u'", (ST_TABLE,))
        for r in self.cur.fetchall():
            self.cur.execute(f"ALTER TABLE {ST_TABLE} DROP CONSTRAINT {r['conname']}")
            print(f"[MIGRATE] 删除 UNIQUE 约束 {r['conname']}")
        self.conn.commit()
        return True

    def stats(self):
        out = {}
        for t in [GM_TABLE, ST_TABLE, SP_TABLE]:
            self.cur.execute(f"SELECT count(*) AS n FROM {t}")
            out[t] = self.cur.fetchone()['n']
        return out

    # ------------------------------------------------------------
    # 地震波表 (查重: wave_name+source+pga)
    # ------------------------------------------------------------
    def get_or_create_ground_motion(self, motion, wave_name, pga, dt=0.02,
                                    source='synthetic'):
        motion = np.asarray(motion, dtype=np.float32)
        self.cur.execute(
            f"SELECT gm_id FROM {GM_TABLE} WHERE wave_name=%s AND source=%s AND ABS(pga-%s) < 1e-9",
            (wave_name, source, float(pga)))
        row = self.cur.fetchone()
        if row:
            return row['gm_id']
        self.cur.execute(
            f"INSERT INTO {GM_TABLE} (wave_name, source, pga, dt, duration, n_steps, motion_bytes) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING gm_id",
            (wave_name, source, float(pga), float(dt),
             float(len(motion)*dt), len(motion), _arr_to_bytes(motion)))
        gid = self.cur.fetchone()['gm_id']
        self.conn.commit()
        return gid

    def count_structures(self):
        self.cur.execute(f"SELECT count(*) AS n FROM {ST_TABLE}")
        return self.cur.fetchone()['n']

    def count_samples(self):
        self.cur.execute(f"SELECT count(*) AS n FROM {SP_TABLE}")
        return self.cur.fetchone()['n']

    def count_done_samples(self):
        """只统计有计算结果的样本 (sim_status='done'), 排除 pending/failed"""
        self.cur.execute(f"SELECT count(*) AS n FROM {SP_TABLE} WHERE sim_status='done'")
        return self.cur.fetchone()['n']

    def count_pending_samples(self):
        """统计未完成/被中断的样本 (sim_status='pending')"""
        self.cur.execute(f"SELECT count(*) AS n FROM {SP_TABLE} WHERE sim_status='pending'")
        return self.cur.fetchone()['n']

    def count_ground_motions(self):
        self.cur.execute(f"SELECT count(*) AS n FROM {GM_TABLE}")
        return self.cur.fetchone()['n']

    def get_ground_motion(self, gm_id):
        self.cur.execute(f"SELECT * FROM {GM_TABLE} WHERE gm_id=%s", (gm_id,))
        row = self.cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d['motion'] = _bytes_to_arr(d.pop('motion_bytes'))
        return d

    def get_all_ground_motions(self):
        """读取全部地震波 (供仿真使用, 不重复入库)"""
        self.cur.execute(
            f"SELECT gm_id, wave_name, pga, dt, motion_bytes FROM {GM_TABLE} "
            "ORDER BY gm_id")
        waves = []
        for r in self.cur.fetchall():
            d = dict(r)
            d['motion'] = _bytes_to_arr(d.pop('motion_bytes'))
            waves.append(d)
        return waves

    # ------------------------------------------------------------
    # 结构模型表 (查重: 全部几何参数)
    # ------------------------------------------------------------
    def get_or_create_structure(self, frame_params, floor_loads=None,
                                floor_masses=None):
        ns = int(frame_params['num_stories'])
        nx = int(frame_params['num_spans_x'])
        ny = int(frame_params['num_spans_y'])
        sx = float(frame_params['span_x'])
        sy = float(frame_params['span_y'])
        sh = float(frame_params['story_height'])
        bw = float(frame_params['beam_width'])
        bh = float(frame_params['beam_height'])
        st = float(frame_params.get('slab_thickness', 0.2))
        plane_shape = str(frame_params.get('plane_shape') or 'rect').lower()
        col_sections = ','.join(f"{x:.4f}" for x in frame_params.get('col_sections', []))
        # 逐层梁截面: [(w,h), ...] -> "w,h;w,h;..."
        bs = frame_params.get('beam_sections', [])
        if bs:
            beam_sections = ';'.join(f"{w:.4f},{h:.4f}" for w, h in bs)
        else:
            beam_sections = f"{bw:.4f},{bh:.4f}"
        fl = ','.join(f"{x:.3f}" for x in floor_loads) if floor_loads is not None else ''
        fm = ','.join(f"{x:.1f}" for x in floor_masses) if floor_masses is not None else ''
        total_mass = float(np.sum(floor_masses)) if floor_masses is not None else 0.0
        n_cols = len(frame_params.get('columns', []))
        n_beams = len(frame_params.get('beams', []))

        self.cur.execute(
            f"""SELECT struct_id, col_sections, beam_sections FROM {ST_TABLE} WHERE
               num_stories=%s AND num_bays_x=%s AND num_bays_y=%s AND
               span_x=%s AND span_y=%s AND story_height=%s AND
               beam_width=%s AND beam_height=%s AND slab_thickness=%s
               AND plane_shape=%s""",
            (ns, nx, ny, sx, sy, sh, bw, bh, st, plane_shape))
        for row in self.cur.fetchall():
            # 几何相同还需截面一致才复用 (截面随机化后)
            same_col = (row['col_sections'] or '') == col_sections
            same_beam = (row['beam_sections'] or '') == beam_sections
            if same_col and same_beam:
                return row['struct_id']

        self.cur.execute(
            f"""INSERT INTO {ST_TABLE}
               (num_stories, num_bays_x, num_bays_y, span_x, span_y,
                story_height, total_height, beam_width, beam_height,
                slab_thickness, col_sections, beam_sections, floor_loads,
                floor_masses, total_mass_kg, n_columns, n_beams, plane_shape)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               RETURNING struct_id""",
            (ns, nx, ny, sx, sy, sh, float(ns*sh), bw, bh, st,
             col_sections, beam_sections, fl, fm, total_mass, n_cols, n_beams,
             plane_shape))
        sid = self.cur.fetchone()['struct_id']
        self.conn.commit()
        return sid

    def get_structure(self, struct_id):
        self.cur.execute(f"SELECT * FROM {ST_TABLE} WHERE struct_id=%s", (struct_id,))
        row = self.cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d['col_sections'] = [float(x) for x in d['col_sections'].split(',')] if d['col_sections'] else []
        d['beam_sections'] = ([tuple(float(y) for y in x.split(','))
                               for x in d['beam_sections'].split(';')]
                              if d['beam_sections'] else [])
        d['floor_loads'] = [float(x) for x in d['floor_loads'].split(',')] if d['floor_loads'] else []
        d['floor_masses'] = [float(x) for x in d['floor_masses'].split(',')] if d['floor_masses'] else []
        return d

    # ------------------------------------------------------------
    # 样本表 (查重: struct_id + gm_id + target_pga)
    # ------------------------------------------------------------
    def get_or_create_sample(self, struct_id, gm_id, target_pga):
        self.cur.execute(
            f"SELECT sample_id FROM {SP_TABLE} WHERE struct_id=%s AND gm_id=%s AND ABS(target_pga-%s) < 1e-9",
            (struct_id, gm_id, float(target_pga)))
        row = self.cur.fetchone()
        if row:
            return row['sample_id'], False
        self.cur.execute(
            f"INSERT INTO {SP_TABLE} (struct_id, gm_id, target_pga, sim_status) "
            "VALUES (%s,%s,%s,'pending') RETURNING sample_id",
            (struct_id, gm_id, float(target_pga)))
        sid = self.cur.fetchone()['sample_id']
        self.conn.commit()
        return sid, True

    def save_sample_result(self, sample_id, roof_disp, applied_pga=None,
                           motion_scale=None, sim_status='done', sim_time_s=None,
                           commit=True):
        roof_disp = np.asarray(roof_disp, dtype=np.float32)
        self.cur.execute(
            f"""UPDATE {SP_TABLE} SET
               roof_disp_bytes=%s, applied_pga=%s, motion_scale=%s,
               disp_std=%s, disp_peak=%s, disp_final=%s, n_steps=%s,
               sim_status=%s, sim_time_s=%s, updated_at=now()
               WHERE sample_id=%s""",
            (_arr_to_bytes(roof_disp),
             float(applied_pga) if applied_pga is not None else None,
             float(motion_scale) if motion_scale is not None else None,
             float(roof_disp.std()), float(np.abs(roof_disp).max()),
             float(roof_disp[-1]), len(roof_disp),
             sim_status, sim_time_s, sample_id))
        if commit:
            self.conn.commit()

    def get_sample(self, sample_id):
        self.cur.execute(f"SELECT * FROM {SP_TABLE} WHERE sample_id=%s", (sample_id,))
        row = self.cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get('roof_disp_bytes'):
            d['roof_disp'] = _bytes_to_arr(d.pop('roof_disp_bytes'))
        return d

    def get_pending_samples(self, limit=None):
        q = f"SELECT * FROM {SP_TABLE} WHERE sim_status='pending'"
        if limit:
            q += f" LIMIT {int(limit)}"
        self.cur.execute(q)
        return [dict(r) for r in self.cur.fetchall()]

    def get_sample_status(self, struct_id, gm_id, target_pga):
        """查询 (结构,波,PGA) 组合的样本状态; 返回 sample_id + sim_status 或 None

        Returns:
            dict {'sample_id': int, 'sim_status': str} 或 None (不存在)
        """
        self.cur.execute(
            f"SELECT sample_id, sim_status FROM {SP_TABLE} "
            "WHERE struct_id=%s AND gm_id=%s AND ABS(target_pga-%s) < 1e-9",
            (struct_id, gm_id, float(target_pga)))
        row = self.cur.fetchone()
        if row is None:
            return None
        return {'sample_id': row['sample_id'], 'sim_status': row['sim_status']}

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------
    def query_samples(self, n_stories=None, pga_range=None, struct_ids=None,
                      limit=None, plane_shape=None, floor_load_kpa=None):
        """查询已完成仿真的样本 (可加数据子集过滤).

        Args:
            n_stories: 仅查指定层数.
            pga_range: (min, max) 目标 PGA 范围.
            struct_ids: 仅查指定结构 id 列表.
            limit: 返回条数上限.
            plane_shape: 仅查该平面形状的结构 (rect/t/l/c/u, None=不限).
            floor_load_kpa: 仅查楼层荷载**每层均等于**该值 (kPa) 的结构,
                            None=不限 (需 floor_loads 每层都==该值).
        """
        q = f"""
            SELECT s.sample_id, s.struct_id, s.gm_id, s.target_pga, s.applied_pga,
                   s.disp_std, s.disp_peak, st.num_stories, st.num_bays_x,
                   st.num_bays_y, st.span_x, st.span_y, st.story_height,
                   st.total_height, st.beam_width, st.beam_height
            FROM {SP_TABLE} s
            JOIN {ST_TABLE} st ON st.struct_id = s.struct_id
            WHERE s.sim_status='done'
        """
        args = []
        if n_stories is not None:
            q += " AND st.num_stories=%s"; args.append(int(n_stories))
        if pga_range is not None:
            q += " AND s.target_pga BETWEEN %s AND %s"
            args += [float(pga_range[0]), float(pga_range[1])]
        if struct_ids:
            q += " AND s.struct_id = ANY(%s)"; args.append(list(struct_ids))
        if plane_shape is not None:
            q += " AND LOWER(st.plane_shape)=%s"
            args.append(str(plane_shape).lower())
        if floor_load_kpa is not None:
            # floor_loads 存为逗号分隔 (如 "20.000,15.000"); 每层均等于该值
            v = f"{float(floor_load_kpa):.3f}"
            # 正则: 整串都是该值(允许前后空格/0补齐), 用 split 后全等判断
            q += (" AND st.floor_loads IS NOT NULL"
                  " AND array_length(string_to_array(st.floor_loads, ','),1) > 0"
                  " AND NOT EXISTS ("
                  "   SELECT 1 FROM unnest(string_to_array(st.floor_loads, ',')) AS fl"
                  "   WHERE ABS(fl::float8 - %s) > 1e-6)")
            args.append(float(floor_load_kpa))
        q += " ORDER BY s.sample_id"
        if limit:
            q += " LIMIT %s"; args.append(int(limit))
        self.cur.execute(q, args)
        return [dict(r) for r in self.cur.fetchall()]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == '__main__':
    db = SLFDatabase()
    print("连接成功, 统计:", db.stats())
    db.close()
