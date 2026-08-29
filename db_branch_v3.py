# db_branch_v3.py
"""
v3.0 分支脚本: 把现有三张表 (ground_motions / structures / samples)
复制出一套新的 *_v3 表 (ground_motions_v3 / structures_v3 / samples_v3)。

之后 v3 版本所有代码默认操作 *_v3 表, 旧的 ground_motions / structures / samples
完全保留不动 (作为 v2 及以前的数据), 互不影响。

用法:
    python db_branch_v3.py [--password pgsql] [--host localhost] [--port 5432]
                           [--user postgres] [--dbname slf_sim]

特性:
    - 幂等: 目标 _v3 表已存在则跳过, 不会重复复制/覆盖
    - 复制包含: 表结构 (列/默认值/注释)、全部数据、约束/索引/外键、独立主键序列
    - 表名和后缀可通过环境变量/参数覆盖:
        SLF_PG_SUFFIX=v3  /  --suffix v3
"""
import os
import sys
import argparse
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# 默认分支后缀 (所有 v3 代码统一从这里读取, 便于整体改后缀)
DEFAULT_SUFFIX = os.environ.get('SLF_PG_SUFFIX', 'v3')

# 需要复制的三张表
SOURCE_TABLES = ['ground_motions', 'structures', 'samples']


def table_exists(cur, table):
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s", (table,))
    return cur.fetchone() is not None


def _copy_table(cur, src, dst):
    """把 src 表完整复制为 dst (结构 + 数据 + 独立主键序列), 幂等.

    实现要点:
      1. 建表时用 LIKE INCLUDING DEFAULTS INCLUDING STORAGE INCLUDING COMMENTS
         (不含约束/索引, 避免新旧表在同一个 schema 下索引/约束名冲突)
      2. 复制数据
      3. 为 SERIAL 自增列重建独立序列 (起点 max+1, 保证新旧表 ID 不重叠)
      4. 重建约束/索引/外键 (主键 + UNIQUE + 外键 + 普通索引)
    """
    if table_exists(cur, dst):
        print(f"  [SKIP] {dst} 已存在, 跳过 (如需重建请先手动 DROP)")
        return False

    print(f"  [OK] 复制 {src} -> {dst}")
    # 1. 建表结构 (默认值/存储/注释, 不含约束与索引)
    cur.execute(
        sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS "
                "INCLUDING STORAGE INCLUDING COMMENTS)")
        .format(sql.Identifier(dst), sql.Identifier(src)))

    # 2. 复制数据
    cur.execute(sql.SQL("INSERT INTO {} SELECT * FROM {}")
                .format(sql.Identifier(dst), sql.Identifier(src)))

    # 3. 为 SERIAL 自增列建独立序列 (起点 = max+1), 避免新旧表主键冲突
    cur.execute(
        """SELECT a.attname
           FROM pg_attribute a
           WHERE a.attrelid = %s::regclass AND a.attnum > 0
             AND NOT a.attisdropped
             AND pg_get_serial_sequence(%s, a.attname) IS NOT NULL""",
        (src, src))
    for row in cur.fetchall():
        col = row['attname']
        new_seq = f"{dst}_{col}_seq"
        cur.execute(
            sql.SQL("CREATE SEQUENCE IF NOT EXISTS {}").format(sql.Identifier(new_seq)))
        cur.execute(
            sql.SQL("SELECT setval({}, (SELECT COALESCE(MAX({}), 0) + 1 FROM {}))")
            .format(sql.Literal(new_seq), sql.Identifier(col), sql.Identifier(dst)))
        cur.execute(
            sql.SQL("ALTER TABLE {} ALTER COLUMN {} SET DEFAULT nextval({}::regclass)")
            .format(sql.Identifier(dst), sql.Identifier(col), sql.Literal(new_seq)))
        cur.execute(
            sql.SQL("ALTER SEQUENCE {} OWNED BY {}.{}")
            .format(sql.Identifier(new_seq), sql.Identifier(dst), sql.Identifier(col)))
        print(f"    + 序列 {new_seq} (起点 max({col})+1)")

    # 4. 重建约束/索引/外键 (主键 / UNIQUE / 外键 / 普通索引)
    _recreate_pk(cur, src, dst)
    _recreate_unique(cur, src, dst)
    _recreate_fks(cur, src, dst)
    _recreate_indexes(cur, src, dst)
    return True


def _recreate_pk(cur, src, dst):
    """复制主键 (新表独立命名)."""
    cur.execute(
        """SELECT a.attname
           FROM pg_index i
           JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
           WHERE i.indrelid = %s::regclass AND i.indisprimary
           ORDER BY array_position(i.indkey, a.attnum)""", (src,))
    cols = [r['attname'] for r in cur.fetchall()]
    if not cols:
        return
    cur.execute(
        sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})")
        .format(sql.Identifier(dst),
                sql.SQL(', ').join(sql.Identifier(c) for c in cols)))
    print(f"    + 主键 ({', '.join(cols)})")


def _recreate_unique(cur, src, dst):
    """复制 UNIQUE 约束 (排除主键, 新表独立命名)."""
    cur.execute(
        """SELECT con.conname, con.conkey,
                  pg_get_constraintdef(con.oid) AS def
           FROM pg_constraint con
           WHERE con.conrelid = %s::regclass AND con.contype='u'""", (src,))
    for r in cur.fetchall():
        name = f"{dst}_{r['conname']}"
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} {}")
            .format(sql.Identifier(dst), sql.Identifier(name),
                    sql.SQL(r['def'])))
        print(f"    + UNIQUE {r['conname']}")


def _recreate_fks(cur, src, dst):
    """复制外键 (新表独立命名).

    关键: 外键的 REFERENCES 目标也指向带后缀的新表 (如 samples_v3 外键
    指向 structures_v3 / ground_motions_v3), 使 v3 表与旧表完全解耦。
    仅当被引用的目标新表存在时才重建。
    """
    suffix = dst[len(src) + 1:]  # 从 dst 名推断后缀 (src + '_' + suffix)
    cur.execute(
        """SELECT con.conname, pg_get_constraintdef(con.oid) AS def,
                  ref.relname AS ref_table
           FROM pg_constraint con
           JOIN pg_class ref ON ref.oid = con.confrelid
           WHERE con.conrelid = %s::regclass AND con.contype='f'""", (src,))
    for r in cur.fetchall():
        ref_name = r['ref_table']
        ref_target = f"{ref_name}_{suffix}"
        if not table_exists(cur, ref_target):
            # 被引用表无对应新表, 跳过 (保持兼容)
            print(f"    - 外键 {r['conname']} 引用的 {ref_target} 不存在, 跳过")
            continue
        name = f"{dst}_{r['conname']}"
        new_def = r['def'].replace(f"REFERENCES {ref_name}",
                                   f"REFERENCES {ref_target}")
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} {}")
            .format(sql.Identifier(dst), sql.Identifier(name),
                    sql.SQL(new_def)))
        print(f"    + 外键 {r['conname']} -> {ref_target}")


def _recreate_indexes(cur, src, dst):
    """复制普通索引 (含表达式索引, 新表独立命名)."""
    cur.execute(
        """SELECT i.relname AS idx_name,
                  pg_get_indexdef(idx.indexrelid) AS def
           FROM pg_index idx
           JOIN pg_class i ON i.oid = idx.indexrelid
           WHERE idx.indrelid = %s::regclass
             AND NOT idx.indisprimary
             AND NOT idx.indisunique
           ORDER BY i.relname""", (src,))
    for r in cur.fetchall():
        # pg_get_indexdef 返回 "CREATE [UNIQUE] INDEX name ON schema.table ..."
        # 替换成带 dst 表名 + 独立索引名的版本
        old_def = r['def']
        new_def = old_def.replace(r['idx_name'], f"{dst}_{r['idx_name']}")
        new_def = new_def.replace(f"ON public.{src}", f"ON public.{dst}")
        new_def = new_def.replace(f"ON {src}", f"ON {dst}")
        cur.execute(sql.SQL(new_def))
        print(f"    + 索引 {r['idx_name']}")


def main():
    parser = argparse.ArgumentParser(description="v3.0 分支: 复制三张表为 *_v3 新表")
    parser.add_argument("--host", default=os.environ.get("SLF_PG_HOST", "localhost"))
    parser.add_argument("--port", default=int(os.environ.get("SLF_PG_PORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("SLF_PG_USER", "postgres"))
    parser.add_argument("--password", default=None,
                        help="postgres 密码 (或环境变量 SLF_PG_PASSWORD)")
    parser.add_argument("--dbname", default=os.environ.get("SLF_PG_DB", "slf_sim"))
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX,
                        help=f"新表后缀 (默认 {DEFAULT_SUFFIX})")
    args = parser.parse_args()

    password = args.password or os.environ.get("SLF_PG_PASSWORD", "pgsql")
    dsn = dict(host=args.host, port=args.port, dbname=args.dbname,
               user=args.user, password=password)
    suffix = args.suffix

    conn = psycopg2.connect(**dsn, connect_timeout=10)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print(f"[v3 分支] 数据库 {args.dbname} / 后缀 '{suffix}'")
    for t in SOURCE_TABLES:
        dst = f"{t}_{suffix}"
        if not table_exists(cur, t):
            print(f"  [WARN] 源表 {t} 不存在, 跳过")
            continue
        _copy_table(cur, t, dst)

    # 汇总
    print("\n[统计]")
    for t in SOURCE_TABLES + [f"{t}_{suffix}" for t in SOURCE_TABLES]:
        if table_exists(cur, t):
            cur.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(t)))
            print(f"  {t:<22} {cur.fetchone()['n']} 行")
    conn.close()
    print("[DONE] v3 分支表就绪, 旧表未改动")


if __name__ == '__main__':
    main()
