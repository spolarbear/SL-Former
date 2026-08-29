# db_setup_secure.py
# 安全引导 (Python版):
#   由 PowerShell 解密 .pgpass.tmp -> 通过 stdin 传密码 -> 本脚本初始化数据库
#   用法 (PowerShell):
#     $sec = Get-Content '.pgpass.tmp' | ConvertTo-SecureString
#     $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
#     $pw = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b)
#     $pw | & python db_setup_secure.py
import sys, os, io

def main():
    # 从 stdin 读密码 (一行)
    password = sys.stdin.readline().strip()
    if not password:
        print("[ERR] no password from stdin")
        sys.exit(1)
    print(f"[OK] password received (len={len(password)})")

    os.environ['SLF_PG_PASSWORD'] = password
    os.environ['SLF_PG_DB'] = os.environ.get('SLF_PG_DB', 'slf_sim')
    os.environ['SLF_PG_HOST'] = os.environ.get('SLF_PG_HOST', 'localhost')
    os.environ['SLF_PG_PORT'] = os.environ.get('SLF_PG_PORT', '5432')
    os.environ['SLF_PG_USER'] = os.environ.get('SLF_PG_USER', 'postgres')

    import psycopg2
    from psycopg2 import sql

    host = os.environ['SLF_PG_HOST']; port = int(os.environ['SLF_PG_PORT'])
    user = os.environ['SLF_PG_USER']; db = os.environ['SLF_PG_DB']

    # 1. 创建数据库
    try:
        conn = psycopg2.connect(host=host, port=port, dbname='postgres',
                                user=user, password=password, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
            print(f"[OK] database created: {db}")
        else:
            print(f"[OK] database exists: {db}")
        conn.close()
    except Exception as e:
        print("[ERR] connect postgres failed:", e)
        sys.exit(1)

    # 2. 建表
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from db_manager import SLFDatabase
    d = SLFDatabase()
    d.init_schema()
    print("[OK] schema initialized. stats:", d.stats())
    d.close()
    print("[DONE] SLF database initialized!")

if __name__ == '__main__':
    main()
