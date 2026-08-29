# db_generate_samples.py
"""
批量生成样本脚本 (与训练完全脱开)

功能:
    1. 读取生成规则 (结构参数范围 / 地震动池 / PGA 范围)
    2. 批量生成结构 -> 查重 (structures 表唯一键)
    3. 对每个结构 x 地震波 x PGA 组合 -> 查重 (samples 表唯一键)
    4. 只对缺失的样本运行 OpenSees 仿真, 结果写入 samples
    5. 重复运行不会重复仿真 (查重机制)

用法:
    python db_generate_samples.py --num 5000 --n_waves 50 --workers 12
    python db_generate_samples.py --force   # 忽略已完成样本, 强制重算 (谨慎)

依赖:
    - PostgreSQL (slf_sim 库)
    - openseespy (sam3 环境)
"""
import os
import sys
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from db_manager import SLFDatabase, SP_TABLE
from simulation_cache import run_single_simulation


# ============================================================
# 生成规则 (简单模式: 全部离散固定步长, 保证组合可重复/查重生效)
# ============================================================
def make_simple_rule(num_samples=1000, n_waves=50, seed=42):
    cfg = Config()
    # 离散 PGA 取值 (固定步长): 0.1 / 0.15 / 0.2 g
    pga_options = list(getattr(cfg, 'PGA_OPTIONS', [0.10]))
    return dict(
        num_samples=num_samples,
        n_waves=n_waves,
        seed=seed,                     # 随机种子: 同一 seed 生成同一批结构 (可重复/查重生效); 换 seed 生成全新样本
        pga_options=pga_options,
        num_stories=[2,3,4,5,6,7,8,9,10,11],         # 离散层数状态值
        span_widths=[3.0, 4.0, 5.0, 6.0, 8.0],
        story_heights=[3.0, 4.0, 5.0, 6.0],
        floor_load_options=[10.0, 15.0, 20.0, 25.0],   # 离散荷载选项
        target_dt=cfg.TARGET_DT,
    )


# ============================================================
# 结构生成 (简单模式, 与 simulation_cache 一致的确定性重建)
# ============================================================
def build_frame_from_params(p, col_sections=None, beam_sections=None):
    from generate_frames import generate_fixed_frame, id_to_shape
    ns = int(p[0]); nx = int(p[1]); ny = int(p[2])
    sx = float(p[3]); sy = float(p[4]); sh = float(p[5])
    # 平面形状 (第 8 位 ID, 兼容旧 8 维)
    plane_shape = id_to_shape(p[8]) if len(p) > 8 else 'rect'
    ms = max(sx, sy)
    bh = max(0.4, min(ms/12, 0.8)); bh = round(bh / 0.2) * 0.2   # 200mm 模数
    bw = max(0.2, min(bh/2.5, 0.5)); bw = round(bw / 0.2) * 0.2  # 200mm 模数
    return generate_fixed_frame(ns, nx, ny, sx, sy, sh, 0.6, bw, bh,
                                col_sections=col_sections,
                                beam_sections=beam_sections,
                                plane_shape=plane_shape)


# 梁截面选项 (宽×高, m) — 200mm 模数
BEAM_SECTION_OPTIONS = [(0.2, 0.4), (0.2, 0.6), (0.4, 0.6), (0.4, 0.8)]


def _random_col_sections(ns, nx, ny, sx, sy, sh, load_per_area=20.0):
    """生成逐层柱截面 (正方形, 200mm 模数, 渐变+扰动)。

    轴压比 0.3~0.9, 截面 200×200~1400×1400mm。
    底部大顶部小 (按楼层渐变), 加随机扰动。
    直接按轴压比公式生成, 但底部截面在 0.2~1.4m 范围随机锚定,
    保证样本覆盖整个尺寸范围。
    """
    import random as _r
    tributary = sx * sy
    # 底部柱: 在 0.2~1.4m 随机锚定 (200mm 模数), 使样本覆盖全范围
    bottom = _r.choice([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    # 按层数渐变到顶部 (顶部约 0.3~0.5 倍底部, 最小 0.2)
    ratio_top = _r.uniform(0.3, 0.5)
    top = max(0.2, round(bottom * ratio_top / 0.2) * 0.2)
    if top >= bottom:
        top = max(0.2, bottom - 0.2)
    # 线性渐变 + 每层扰动 (200mm 模数)
    base = np.linspace(bottom, top, ns).tolist()
    col_secs = []
    for i, cs in enumerate(base):
        perturb = _r.choice([-0.2, 0.0, 0.0, 0.2])   # ±1 档扰动
        v = max(0.2, min(1.4, cs + perturb))
        v = round(v / 0.2) * 0.2
        col_secs.append(round(v, 2))
    # 单调化 (底部>=顶部)
    for i in range(1, ns):
        if col_secs[i] > col_secs[i-1]:
            col_secs[i] = col_secs[i-1]
    return col_secs


def _random_beam_sections(ns):
    """生成逐层梁截面: 每层从 4 种类型独立随机选 (同层一致, 层间可不同)"""
    return [list(BEAM_SECTION_OPTIONS[int(np.random.randint(len(BEAM_SECTION_OPTIONS)))])
            for _ in range(ns)]


def make_structure_params(rule):
    """按规则生成结构参数 + 每层荷载 + 节点质量 + 随机截面 + 平面形状
    (全部离散固定步长, 固定种子可复现)

    Returns:
        params_list    : [N, 20] 结构参数向量
                         p[0:8]  原 8 维 (层数/跨数/跨度/层高/平均质量/阻尼)
                         p[8:20] 每层楼面荷载 (kPa), 最多 12 层, 不足补 0
        fl_list        : [N] 每层荷载
        fm_list        : [N] 每层节点质量
        col_sec_list   : [N] 每结构逐层柱截面 [[ns], ...]
        beam_sec_list  : [N] 每结构逐层梁截面 [[(w,h)]*ns, ...]
    """
    from simulation_cache import generate_floor_loads, compute_floor_node_masses
    from generate_frames import shape_to_id
    from config import Config as _C
    cfg = Config()
    pdim = int(getattr(_C, 'PARAMS_DIM', 21))
    off = int(getattr(_C, 'PARAMS_FLOOR_LOAD_OFFSET', 9))
    max_fl = int(getattr(_C, 'PARAMS_MAX_FLOORS', 12))
    np.random.seed(rule.get('seed', 42))   # 固定种子: 同一规则生成相同结构组合
    n = rule['num_samples']
    params_list, fl_list, fm_list = [], [], []
    col_sec_list, beam_sec_list = [], []
    for _ in range(n):
        ns = int(np.random.choice(rule['num_stories']))
        sh = float(np.random.choice(rule['story_heights']))
        sx = float(np.random.choice(rule['span_widths']))
        sy = float(np.random.choice(rule['span_widths']))
        max_sx = max(1, int(cfg.SPACE_X // sx)); max_sy = max(1, int(cfg.SPACE_Y // sy))
        # 避免单跨薄弱结构: 跨数从 [2,3,4] 选 (去掉 1)
        nx = int(np.random.choice([2, 3, 4]))
        ny = int(np.random.choice([2, 3, 4]))
        nx = min(nx, max_sx); ny = min(ny, max_sy)
        # 随机平面形状 (避免薄弱连接):
        #   - C/U 形: 横条(连接左右翼)中间段 = nx-2 需 >=2 跨 -> nx>=4
        #   - T/L 形: 至少 2×2
        #   - rect : 任意
        if nx >= 4 and ny >= 2:
            # 可安全选所有形状
            plane_shape = str(np.random.choice(['rect', 'T', 'L', 'C', 'U']))
        elif nx >= 2 and ny >= 2:
            # 尺寸不足 C/U: 只选 rect/T/L (T/L 的竖条连接不需横向 2 跨)
            plane_shape = str(np.random.choice(['rect', 'T', 'L']))
        else:
            plane_shape = 'rect'
        # 离散荷载 (各层可选相同或不同, 从固定选项选)
        loads = generate_floor_loads(ns, rule['floor_load_options'])
        masses = compute_floor_node_masses(ns, nx, ny, sx, sy, loads,
                                           plane_shape=plane_shape)
        # 离散质量: 取整到 100 kg 步长, 保证查重稳定
        masses_rounded = np.round(masses / 100.0) * 100.0
        # 随机柱/梁截面
        col_secs = _random_col_sections(ns, nx, ny, sx, sy, sh)
        beam_secs = _random_beam_sections(ns)
        # 21 维 params: 前 8 维原结构参数, 第 8 位形状 ID, 第 9 位起每层荷载 (kPa)
        p = np.zeros(pdim, dtype=np.float32)
        p[0] = ns; p[1] = nx; p[2] = ny
        p[3] = sx; p[4] = sy; p[5] = sh
        p[6] = float(np.mean(masses_rounded)); p[7] = 0.05
        p[8] = shape_to_id(plane_shape)
        for i, v in enumerate(np.asarray(loads, dtype=np.float32)[:max_fl]):
            p[off + i] = float(v)
        params_list.append(p)
        fl_list.append(np.asarray(loads, dtype=np.float32))
        fm_list.append(masses_rounded.astype(np.float32))
        col_sec_list.append(col_secs)
        beam_sec_list.append(beam_secs)
    return params_list, fl_list, fm_list, col_sec_list, beam_sec_list


# ============================================================
# 单样本仿真 worker
# ============================================================
def _run_one_sample(args):
    (sample_id, params, motion, seq_len, target_dt, pga_target,
     floor_masses, col_sections, beam_sections) = args
    try:
        result = run_single_simulation((
            params, motion, seq_len, target_dt,
            col_sections, beam_sections, Config.E_CONCRETE,
            floor_masses, pga_target))
        return {
            'sample_id': sample_id,
            'roof_disp': result['displacement'],
            'motion': result['motion'],
            'applied_pga': result['pga'],
            'failed': result['failed'],
        }
    except Exception as e:
        return {'sample_id': sample_id, 'failed': True, 'error': str(e)}


# ============================================================
# 主生成逻辑 (查重)
# ============================================================
def generate_samples(rule=None, force=False, workers=4):
    rule = rule or make_simple_rule()
    cfg = Config()
    dt = rule['target_dt']
    seq_len = cfg.get_seq_len()

    db = SLFDatabase()

    # 1. 地震动池: 统一由数据库管理 (ground_motions 唯一波库)
    #    - 表为空时: 一次性从 dzb 加载所有波入库 (用源文件名)
    #    - 之后: 所有生成/仿真只用数据库里的波, 不再重新读取/编号
    waves = db.get_all_ground_motions()
    if len(waves) == 0:
        # 从 dzb 文件夹一次性加载全部真实波 (不用 pool/motion_pool, 统一波库)
        from earthquake_simulator_3d import EarthquakeLoader3D, SimConfig3D
        sc = SimConfig3D()
        sc.TARGET_DT = dt
        sc.WINDOW_BEFORE = cfg.WINDOW_BEFORE
        sc.WINDOW_AFTER = cfg.WINDOW_AFTER
        sc.TARGET_PGA = cfg.TARGET_PGA
        loader = EarthquakeLoader3D(sc)
        motions, names, n_loaded = loader.load_earthquake_files(
            cfg.EARTHQUAKE_FOLDER, cfg.TARGET_PGA, dt,
            cfg.WINDOW_BEFORE, cfg.WINDOW_AFTER,
            max_files=100000)   # 加载 dzb 全部波
        if n_loaded <= 0:
            print("[ERR] dzb 文件夹无可用地震波")
            return 0
        print(f"[WAVE] 从 dzb 一次性加载全部真实波: {n_loaded} 条 -> 入库")
        for i, m in enumerate(motions):
            wname = names[i] if i < len(names) else f"dzb_{i}"
            db.get_or_create_ground_motion(m, wname, cfg.TARGET_PGA,
                                           dt=dt, source='dzb')
        waves = db.get_all_ground_motions()
        print(f"[WAVE] 波库入库完成, 共 {len(waves)} 条")
    else:
        print(f"[WAVE] 从数据库波库读取 {len(waves)} 条固定地震波 (统一管理)")

    # 取前 n_waves 条 (仅用数据库中的波, 不再重新读取/编号)
    n_waves = min(rule['n_waves'], len(waves)) if len(waves) > 0 else 1
    motion_pool = [w['motion'] for w in waves[:n_waves]]
    wave_ids = [w['gm_id'] for w in waves[:n_waves]]
    print(f"[WAVE] 使用数据库波库前 {len(motion_pool)} 条 (gm_id {wave_ids[0]}~{wave_ids[-1]})")

    # 3. 结构参数 + 查重 (make_structure_params 内部已固定 seed)
    params_list, fl_list, fm_list, col_sec_list, beam_sec_list = \
        make_structure_params(rule)
    struct_ids = []
    for i, p in enumerate(params_list):
        frame = build_frame_from_params(p, col_sections=col_sec_list[i],
                                        beam_sections=beam_sec_list[i])
        sid = db.get_or_create_structure(frame, fl_list[i], fm_list[i])
        struct_ids.append(sid)
    print(f"[STRUCT] 结构查重后: {db.count_structures()} 唯一结构")

    # 4. 样本组合查重 (PGA 从离散选项选, 固定 seed 可复现 -> 查重生效)
    #    状态恢复策略:
    #      - 样本不存在      -> 新建 pending 并加入待仿真
    #      - sim_status='pending' -> 恢复仿真 (上次创建后未完成/人工中断)
    #      - sim_status='failed'  -> 重新仿真
    #      - sim_status='done'    -> 已算完, 跳过
    np.random.seed(rule.get('seed', 42))
    todo = []   # (sample_id, params, motion, pga, floor_masses, col_sec, beam_sec)
    made = skipped = resumed = refailed = 0
    pga_options = rule.get('pga_options', [0.10, 0.15, 0.20])
    for i in range(rule['num_samples']):
        pga = float(np.random.choice(pga_options))
        wid = wave_ids[i % len(wave_ids)]
        st = db.get_sample_status(struct_ids[i], wid, pga)
        if st is None:
            # 不存在: 新建 pending 记录, 加入待仿真
            sample_id, _ = db.get_or_create_sample(struct_ids[i], wid, pga)
            todo.append((sample_id, params_list[i], motion_pool[i % len(motion_pool)],
                         pga, fm_list[i], col_sec_list[i], beam_sec_list[i]))
            made += 1
        elif st['sim_status'] == 'done':
            # 已算完: 跳过 (除非 force)
            if force:
                todo.append((st['sample_id'], params_list[i],
                             motion_pool[i % len(motion_pool)], pga, fm_list[i],
                             col_sec_list[i], beam_sec_list[i]))
                made += 1
            else:
                skipped += 1
        elif st['sim_status'] == 'pending':
            # 上次创建后未完成/人工中断: 恢复仿真
            todo.append((st['sample_id'], params_list[i],
                         motion_pool[i % len(motion_pool)], pga, fm_list[i],
                         col_sec_list[i], beam_sec_list[i]))
            made += 1
            resumed += 1
        elif st['sim_status'] == 'failed':
            # 上次失败: 重新仿真
            todo.append((st['sample_id'], params_list[i],
                         motion_pool[i % len(motion_pool)], pga, fm_list[i],
                         col_sec_list[i], beam_sec_list[i]))
            made += 1
            refailed += 1
        else:
            # 其他未知状态, 保守处理: 加入待仿真
            todo.append((st['sample_id'], params_list[i],
                         motion_pool[i % len(motion_pool)], pga, fm_list[i],
                         col_sec_list[i], beam_sec_list[i]))
            made += 1
    print(f"[SAMPLE] 待仿真: {made} (恢复pending {resumed}, 重试failed {refailed}), "
          f"已存在done跳过: {skipped}")

    if made == 0:
        print("[DONE] 无新增样本 (全部命中查重)")
        return 0

    # 5. 并行仿真 + 边算边存 (每完成 100 个样本立即写库 commit, 避免中断丢数据)
    tasks = [(sid, p, m, seq_len, dt, pga, fm, cs, bs)
             for (sid, p, m, pga, fm, cs, bs) in todo]
    # sample_id -> 对应 todo 项的 pga (写库需要)
    todo_by_sid = {t[0]: t for t in todo}
    print(f"[SIM] 启动 {workers} 进程仿真 {made} 个样本...")
    t0 = time.time()
    done = failed = 0
    save_interval = 100
    results = {}
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_sample, a): a[0] for a in tasks}
        with tqdm(total=made, desc="  [SIM] 仿真进度", unit="样本") as pbar:
            for fu in as_completed(futs):
                r = fu.result()
                results[r['sample_id']] = r
                pbar.update(1)
                # 每完成 save_interval 个, 立即写库 commit (边算边存)
                if len(results) >= save_interval and len(results) % save_interval == 0:
                    n_flush = _flush_results_to_db(db, results, todo_by_sid,
                                                   cfg, elapsed=None, commit=True)
                    done += n_flush[0]; failed += n_flush[1]
                    print(f"  💾 已保存 {len(results)} 个样本 (累计 done={done}, failed={failed})")
                    results.clear()
    elapsed = time.time() - t0

    # 6. 剩余结果写库 (不足 100 个的也提交)
    if results:
        n_flush = _flush_results_to_db(db, results, todo_by_sid, cfg,
                                       elapsed=elapsed, commit=True)
        done += n_flush[0]; failed += n_flush[1]
        print(f"  💾 已保存 {len(results)} 个样本 (累计 done={done}, failed={failed})")

    print(f"[SIM] 完成: {done}, 失败: {failed}, 用时 {elapsed:.1f}s")
    print("[DONE] 数据库统计:", db.stats())
    return done


def _flush_results_to_db(db, results, todo_by_sid, cfg, elapsed=None, commit=True):
    """把已完成的仿真结果写入数据库 (返回 (done, failed))"""
    done = failed = 0
    for sid, r in results.items():
        t = todo_by_sid.get(sid)
        if r is None or r.get('failed'):
            failed += 1
            db.cur.execute(f"UPDATE {SP_TABLE} SET sim_status='failed' WHERE sample_id=%s", (sid,))
        else:
            pga = t[3] if t else 0.0
            db.save_sample_result(sid, r['roof_disp'], applied_pga=r['applied_pga'],
                                  motion_scale=pga / cfg.TARGET_PGA if pga > 0 else None,
                                  sim_status='done', sim_time_s=elapsed if elapsed else None,
                                  commit=False)
            done += 1
    if commit:
        db.conn.commit()
    return (done, failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量生成样本到数据库 (与训练脱开)')
    parser.add_argument('--num', type=int, default=5000, help='样本数')
    parser.add_argument('--n_waves', type=int, default=50, help='有限波形池数量')
    parser.add_argument('--workers', type=int, default=12, help='仿真并行进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子: 同一 seed 生成同一批样本(查重跳过); 换 seed 生成全新样本 (默认42)')
    parser.add_argument('--force', action='store_true', help='强制重算已完成样本')
    args = parser.parse_args()

    rule = make_simple_rule(num_samples=args.num, n_waves=args.n_waves, seed=args.seed)
    print(f"[SEED] 使用随机种子: {rule['seed']}")
    generate_samples(rule=rule, force=args.force, workers=args.workers)
