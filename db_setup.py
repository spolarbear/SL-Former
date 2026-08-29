# db_setup.py
"""
初始化 SLF 样本集数据库 (解耦架构):
    1. 创建数据库 slf_sim (若不存在)
    2. 建表 (ground_motions / structures / samples)

用法:
    python db_setup.py --password pgsql
    # 或设置环境变量: $env:SLF_PG_PASSWORD="pgsql"
"""
import os
import sys
import argparse
import psycopg2
from psycopg2 import sql


def main():
    parser = argparse.ArgumentParser(description="初始化 SLF 数据库 (解耦架构)")
    parser.add_argument("--host", default=os.environ.get("SLF_PG_HOST", "localhost"))
    parser.add_argument("--port", default=int(os.environ.get("SLF_PG_PORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("SLF_PG_USER", "postgres"))
    parser.add_argument("--password", default=None,
                        help="postgres 超级用户密码 (或环境变量 SLF_PG_PASSWORD)")
    parser.add_argument("--dbname", default=os.environ.get("SLF_PG_DB", "slf_sim"))
    args = parser.parse_args()

    password = args.password or os.environ.get("SLF_PG_PASSWORD", "pgsql")

    admin_dsn = dict(host=args.host, port=args.port, dbname="postgres",
                     user=args.user, password=password)

    # 1. 创建数据库
    try:
        conn = psycopg2.connect(**admin_dsn, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (args.dbname,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(args.dbname)))
            print("[OK] database created: " + args.dbname)
        else:
            print("[OK] database exists: " + args.dbname)
        conn.close()
    except Exception as e:
        print("[ERR] connect postgres failed:", e)
        sys.exit(1)

    # 2. 建表
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ["SLF_PG_PASSWORD"] = password
    os.environ["SLF_PG_DB"] = args.dbname
    os.environ["SLF_PG_HOST"] = args.host
    os.environ["SLF_PG_PORT"] = str(args.port)
    os.environ["SLF_PG_USER"] = args.user

    from db_manager import SLFDatabase
    db = SLFDatabase()
    db.init_schema()
    print("[OK] schema initialized, stats:", db.stats())
    db.close()
    print("[DONE] SLF database ready (decoupled)")


if __name__ == "__main__":
    main()
