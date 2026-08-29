# structure_check.py
"""
结构模型健康检查脚本 (抽查)

用途:
    从数据库抽查若干结构模型, 用 OpenSees 重建并检查:
    1. 模态分析: 前 6 阶自振频率, 检查是否有异常局部振动 (频率异常低/阶间跳变)
    2. 重力加载: 施加自重, 检查基底反力 vs 总自重是否平衡
    3. 时程位移检查: 跑一条地震时程, 检查各节点位移是否有异常 (孤立节点位移爆表)

抽查策略:
    - 从数据库均匀采样若干结构 (覆盖不同层数/跨数)
    - 每结构重建规则网格模型 (与 run_analysis 一致)
    - 输出检查报告 (控制台 + CSV)

用法:
    python structure_check.py --num 10                  # 抽查 10 个结构
    python structure_check.py --num 10 --output ./plots/sci/struct_check
    python structure_check.py --stories 5,7,9           # 指定层数抽查
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_manager import SLFDatabase, ST_TABLE
from config import Config


# ============================================================
# OpenSees 建模 (与 run_analysis 一致的规则网格)
# ============================================================
def build_model(ops, num_stories, num_bays_x, num_bays_y,
                bay_width_x, bay_width_y, story_height,
                col_sections, beam_b, beam_h, E, nu, floor_node_masses=None,
                mass_per_node=35000.0):
    """构建 3D 框架模型, 返回 node_tags 和 elem_id 总数"""
    G = E / (2 * (1 + nu))
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    node_tags = {}
    node_id = 0
    for floor in range(num_stories + 1):
        for iy in range(num_bays_y + 1):
            for ix in range(num_bays_x + 1):
                node_id += 1
                ops.node(node_id, ix * bay_width_x, iy * bay_width_y, floor * story_height)
                node_tags[(floor, ix, iy)] = node_id
    # 固定底部
    for iy in range(num_bays_y + 1):
        for ix in range(num_bays_x + 1):
            ops.fix(node_tags[(0, ix, iy)], 1, 1, 1, 1, 1, 1)

    ops.uniaxialMaterial('Elastic', 1, E)
    ops.uniaxialMaterial('Elastic', 2, G)
    ops.geomTransf('Linear', 1, 1, 0, 0)  # 柱
    ops.geomTransf('Linear', 2, 0, 1, 0)  # X 梁
    ops.geomTransf('Linear', 3, 1, 0, 0)  # Y 梁

    # 柱
    elem_id = 0
    for floor in range(1, num_stories + 1):
        col_size = col_sections[min(floor - 1, len(col_sections) - 1)]
        A = col_size ** 2; I = col_size ** 4 / 12
        for iy in range(num_bays_y + 1):
            for ix in range(num_bays_x + 1):
                elem_id += 1
                ops.element('elasticBeamColumn', elem_id,
                            node_tags[(floor - 1, ix, iy)], node_tags[(floor, ix, iy)],
                            A, E, G, I, I, I, 1)
    # X 梁
    A_b = beam_b * beam_h; I_b = beam_b * beam_h ** 3 / 12
    for floor in range(1, num_stories + 1):
        for iy in range(num_bays_y + 1):
            for ix in range(num_bays_x):
                elem_id += 1
                ops.element('elasticBeamColumn', elem_id,
                            node_tags[(floor, ix, iy)], node_tags[(floor, ix + 1, iy)],
                            A_b, E, G, I_b, I_b, I_b, 2)
    # Y 梁
    for floor in range(1, num_stories + 1):
        for iy in range(num_bays_y):
            for ix in range(num_bays_x + 1):
                elem_id += 1
                ops.element('elasticBeamColumn', elem_id,
                            node_tags[(floor, ix, iy)], node_tags[(floor, ix, iy + 1)],
                            A_b, E, G, I_b, I_b, I_b, 3)
    return node_tags, elem_id


def apply_masses(ops, node_tags, num_stories, floor_node_masses):
    """按楼层赋节点质量"""
    # 该层节点 (同一楼层所有网格节点)
    n_per_floor = {}
    for (fl, ix, iy), nid in node_tags.items():
        n_per_floor.setdefault(fl, []).append(nid)
    for floor in range(1, num_stories + 1):
        m = floor_node_masses[min(floor - 1, len(floor_node_masses) - 1)]
        for nid in n_per_floor.get(floor, []):
            ops.mass(nid, m, m, m, 0, 0, 0)


# ============================================================
# 1. 模态分析
# ============================================================
def check_modes(ops, num_stories, num_bays_x, num_bays_y,
                bay_width_x, bay_width_y, story_height,
                col_sections, beam_b, beam_h, E, nu, floor_node_masses):
    """模态分析: 前 6 阶频率, 检查局部振动"""
    node_tags, _ = build_model(ops, num_stories, num_bays_x, num_bays_y,
                               bay_width_x, bay_width_y, story_height,
                               col_sections, beam_b, beam_h, E, nu,
                               floor_node_masses)
    apply_masses(ops, node_tags, num_stories, floor_node_masses)
    ops.numberer('RCM')
    ops.system('BandGen')
    try:
        omega2 = ops.eigen('-fullGenLapack', 6)
        freqs = np.sqrt(np.abs(np.asarray(omega2, dtype=np.float64))) / (2 * np.pi)
    except Exception as e:
        return {'error': str(e)}
    freqs = np.sort(freqs)
    # 检查: 首阶是否异常低 (<0.3Hz 可能是数值问题/局部振动)
    issues = []
    if len(freqs) > 0:
        if freqs[0] < 0.3:
            issues.append(f"首阶频率过低 {freqs[0]:.3f} Hz (疑似数值/局部振动)")
        # 检查相邻频率是否异常接近 (可能是虚假自由度)
        for i in range(1, len(freqs)):
            if freqs[i] - freqs[i-1] < 1e-3:
                issues.append(f"第{i}和{i+1}阶频率几乎相同 ({freqs[i-1]:.3f} vs {freqs[i]:.3f} Hz)")
    return {'freqs': freqs.tolist(), 'issues': issues}


# ============================================================
# 2. 重力加载检查
# ============================================================
def check_gravity(ops, num_stories, num_bays_x, num_bays_y,
                  bay_width_x, bay_width_y, story_height,
                  col_sections, beam_b, beam_h, E, nu, floor_node_masses):
    """重力加载: 施加每层自重, 检查基底反力 vs 总自重"""
    node_tags, _ = build_model(ops, num_stories, num_bays_x, num_bays_y,
                               bay_width_x, bay_width_y, story_height,
                               col_sections, beam_b, beam_h, E, nu,
                               floor_node_masses)
    apply_masses(ops, node_tags, num_stories, floor_node_masses)
    g = 9.81
    # 施加每层节点竖向重力 (F = m*g, 向下)
    n_per_floor = {}
    for (fl, ix, iy) in node_tags:
        n_per_floor.setdefault(fl, []).append(node_tags[(fl, ix, iy)])
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    total_w = 0.0
    for floor in range(1, num_stories + 1):
        m = floor_node_masses[min(floor - 1, len(floor_node_masses) - 1)]
        w_node = m * g
        total_w += w_node * len(n_per_floor.get(floor, []))
        for nid in n_per_floor.get(floor, []):
            ops.load(nid, 0.0, 0.0, -w_node, 0, 0, 0)
    # 静态求解
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGen')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    try:
        ops.analyze(1)
    except Exception as e:
        return {'error': str(e), 'total_weight': total_w}
    # 基底反力 (节点 1..n_base 的竖向反力)
    base_nodes = [nid for (fl, ix, iy), nid in node_tags.items() if fl == 0]
    try:
        ops.reactions()  # 先计算反力
    except Exception:
        pass
    react_sum = 0.0
    react_x = 0.0
    react_y = 0.0
    react_z = 0.0
    for nid in base_nodes:
        react_x += ops.nodeReaction(nid, 1)
        react_y += ops.nodeReaction(nid, 2)
        react_z += ops.nodeReaction(nid, 3)
    # 竖向反力应为 +total_w (向下荷载, 反力向上为正? 约定)
    balance = react_z / (total_w + 1e-12)
    issues = []
    if abs(balance - 1.0) > 0.05:
        issues.append(f"竖向反力/自重 = {balance:.4f} (应≈1, 不平衡)")
    return {
        'total_weight_N': float(total_w),
        'react_z_N': float(react_z),
        'react_xy_N': float(np.hypot(react_x, react_y)),
        'balance_ratio': float(balance),
        'issues': issues,
    }


# ============================================================
# 3. 时程位移检查
# ============================================================
def check_timehistory(ops, motion, dt, num_stories, num_bays_x, num_bays_y,
                      bay_width_x, bay_width_y, story_height,
                      col_sections, beam_b, beam_h, E, nu, floor_node_masses,
                      damping_ratio=0.05, max_steps=None):
    """时程分析: 检查各节点位移, 找异常节点 (max_steps 可限制步数加速)"""
    node_tags, _ = build_model(ops, num_stories, num_bays_x, num_bays_y,
                               bay_width_x, bay_width_y, story_height,
                               col_sections, beam_b, beam_h, E, nu,
                               floor_node_masses)
    apply_masses(ops, node_tags, num_stories, floor_node_masses)
    n_steps = min(len(motion), max_steps) if max_steps else len(motion)

    # 阻尼 (Rayleigh)
    m_ref = float(np.mean(floor_node_masses[:num_stories]))
    E0 = E
    col0 = col_sections[0]
    I0 = col0 ** 4 / 12
    k_est = (num_bays_x + 1) * (num_bays_y + 1) * 12 * E0 * I0 / story_height ** 3
    m_eff = (num_bays_x + 1) * (num_bays_y + 1) * m_ref
    omega1 = np.sqrt(k_est / m_eff) if m_eff > 0 else 1.0
    omega2 = omega1 * 2.5
    alpha_m = 2 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
    beta_k = 2 * damping_ratio / (omega1 + omega2)
    ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)

    # 地震动 (X 向)
    motion_arr = (-1.0 * np.asarray(motion, dtype=np.float64) * 9.81).tolist()
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *motion_arr)
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1, '-dir', 1)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGen')
    ops.test('NormDispIncr', 1e-6, 100)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    # 记录各节点 X 位移峰值
    nids = list(node_tags.values())
    peak_disp = {nid: 0.0 for nid in nids}
    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            break
        for nid in nids:
            d = ops.nodeDisp(nid)
            if len(d) > 0:
                peak_disp[nid] = max(peak_disp[nid], abs(d[0] * 1000.0))  # mm
    # 按楼层分组, 找异常节点 (该层位移远超同层中位数的 3 倍)
    floor_of = {}
    for (fl, ix, iy), nid in node_tags.items():
        floor_of[nid] = fl
    issues = []
    for fl in range(1, num_stories + 1):
        vals = np.array([peak_disp[n] for n in nids if floor_of[n] == fl])
        if len(vals) == 0:
            continue
        med = np.median(vals)
        for n in nids:
            if floor_of[n] == fl and peak_disp[n] > 3 * med + 1e-6 and med > 1e-6:
                issues.append(f"楼层{fl} 节点{n}: 峰值位移 {peak_disp[n]:.2f}mm "
                              f"远超该层中位 {med:.2f}mm")
    top_vals = [peak_disp[n] for n in nids if floor_of[n] == num_stories]
    return {
        'top_peak_mean_mm': float(np.mean(top_vals)) if top_vals else 0.0,
        'top_peak_std_mm': float(np.std(top_vals)) if top_vals else 0.0,
        'max_node_disp_mm': float(max(peak_disp.values())),
        'n_nodes': len(nids),
        'n_base_nodes': len([n for n in nids if floor_of[n] == 0]),
        'issues': issues,
    }


# ============================================================
# 抽查主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='结构模型健康检查 (模态/重力/时程)')
    parser.add_argument('--num', type=int, default=10, help='抽查结构数')
    parser.add_argument('--stories', type=str, default=None,
                        help='指定层数 (逗号分隔, 如 5,7,9); 默认均匀采样')
    parser.add_argument('--output', type=str, default='./plots/sci/struct_check',
                        help='输出目录')
    parser.add_argument('--use_first_wave', type=bool, default=True,
                        help='时程检查用数据库第一条波')
    parser.add_argument('--th_steps', type=int, default=200,
                        help='时程检查步数上限 (默认200, 加速抽查; None=全部500)')
    args = parser.parse_args()

    import openseespy.opensees as ops
    os.makedirs(args.output, exist_ok=True)
    cfg = Config()
    db = SLFDatabase()

    # 选取抽查结构
    if args.stories:
        stories = [int(x) for x in args.stories.split(',')]
        raw_ids = []
        for s in stories:
            db.cur.execute(f"SELECT struct_id FROM {ST_TABLE} WHERE num_stories=%s "
                           "ORDER BY struct_id LIMIT 5", (s,))
            raw_ids.extend([r['struct_id'] for r in db.cur.fetchall()])
        structs = [db.get_structure(sid) for sid in raw_ids]
        structs = [s for s in structs if s is not None]
    else:
        db.cur.execute(f"SELECT struct_id FROM {ST_TABLE} ORDER BY struct_id")
        all_ids = [r['struct_id'] for r in db.cur.fetchall()]
        if not all_ids:
            print('无结构数据')
            return
        # 均匀采样
        idx = np.linspace(0, len(all_ids) - 1, args.num).astype(int)
        idx = np.unique(idx)
        structs = [db.get_structure(all_ids[i]) for i in idx]
        structs = [s for s in structs if s is not None]

    print(f"抽查 {len(structs)} 个结构...")

    # 取一条地震波 (时程检查用)
    waves = db.get_all_ground_motions()
    motion = waves[0]['motion'] if waves else None
    dt = cfg.TARGET_DT

    E = cfg.E_CONCRETE
    nu = 0.2
    beam_b = 0.3; beam_h = 0.6
    col_sections = cfg.COL_SECTIONS

    records = []
    for si, st in enumerate(structs):
        ns = int(st['num_stories'])
        nx = int(st['num_bays_x'])
        ny = int(st['num_bays_y'])
        sx = float(st['span_x'])
        sy = float(st['span_y'])
        sh = float(st['story_height'])
        fm = st.get('floor_masses') or []
        floor_masses = [float(x) for x in fm] if fm else [35000.0] * ns
        print(f"\n--- 结构 {si+1}/{len(structs)}: {ns}层 {nx}x{ny}跨 "
              f"{sx}x{sy}m h={sh}m ---")

        rec = {'struct_id': st['struct_id'], 'num_stories': ns,
               'num_bays': f"{nx}x{ny}", 'span': f"{sx}x{sy}"}

        # 1. 模态
        r = check_modes(ops, ns, nx, ny, sx, sy, sh, col_sections,
                        beam_b, beam_h, E, nu, floor_masses)
        if 'error' in r:
            rec['mode_error'] = r['error']
            print(f"  模态: 失败 {r['error']}")
        else:
            rec['f1'] = r['freqs'][0] if len(r['freqs']) > 0 else 0
            rec['f2'] = r['freqs'][1] if len(r['freqs']) > 1 else 0
            rec['f6'] = r['freqs'][-1] if len(r['freqs']) > 0 else 0
            rec['mode_issues'] = '; '.join(r['issues']) if r['issues'] else '无'
            print(f"  模态: f1={r['freqs'][0]:.3f}Hz f2={r['freqs'][1]:.3f}Hz "
                  f"f6={r['freqs'][-1]:.3f}Hz {'⚠️'+str(r['issues']) if r['issues'] else 'OK'}")

        # 2. 重力
        r = check_gravity(ops, ns, nx, ny, sx, sy, sh, col_sections,
                          beam_b, beam_h, E, nu, floor_masses)
        if 'error' in r:
            rec['grav_error'] = r['error']
            print(f"  重力: 失败 {r['error']}")
        else:
            rec['total_w_kN'] = r['total_weight_N'] / 1000
            rec['react_z_kN'] = r['react_z_N'] / 1000
            rec['balance'] = r['balance_ratio']
            rec['grav_issues'] = '; '.join(r['issues']) if r['issues'] else '无'
            print(f"  重力: 总自重={r['total_weight_N']/1000:.1f}kN "
                  f"反力Z={r['react_z_N']/1000:.1f}kN "
                  f"平衡比={r['balance_ratio']:.3f} {'⚠️'+str(r['issues']) if r['issues'] else 'OK'}")

        # 3. 时程
        if motion is not None:
            r = check_timehistory(ops, motion, dt, ns, nx, ny, sx, sy, sh,
                                  col_sections, beam_b, beam_h, E, nu,
                                  floor_masses, max_steps=args.th_steps)
            rec['top_peak_mean_mm'] = r['top_peak_mean_mm']
            rec['max_node_disp_mm'] = r['max_node_disp_mm']
            rec['n_nodes'] = r['n_nodes']
            rec['th_issues'] = '; '.join(r['issues']) if r['issues'] else '无'
            print(f"  时程: 顶层峰值均值={r['top_peak_mean_mm']:.2f}mm "
                  f"最大节点={r['max_node_disp_mm']:.2f}mm "
                  f"{'⚠️'+str(r['issues']) if r['issues'] else 'OK'}")
        else:
            rec['th_issues'] = '无地震波'
        records.append(rec)

    # 汇总
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output, 'structure_check.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n\n✅ 检查完成! 汇总: {csv_path}")
    # 打印问题汇总
    n_issues = sum(1 for r in records
                   if r.get('mode_issues') not in (None, '无', '')
                   or r.get('grav_issues') not in (None, '无', '')
                   or r.get('th_issues') not in (None, '无', ''))
    print(f"有问题的结构: {n_issues}/{len(records)}")
    db.close()


if __name__ == '__main__':
    main()
