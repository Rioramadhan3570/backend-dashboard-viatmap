from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'viatmap_05-05-26'),
        cursorclass=pymysql.cursors.DictCursor
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXCLUDED USERS
# ═══════════════════════════════════════════════════════════════════════════════

# User ID yang tidak boleh muncul di seluruh pengambilan data
EXCLUDED_USER_IDS = (4, 5, 7, 8, 9)

def excluded_users_filter(table_alias='u', col='id'):
    """
    Kembalikan snippet WHERE/AND untuk mengecualikan EXCLUDED_USER_IDS.
    Contoh: excluded_users_filter('u') → "AND u.id NOT IN (4,5,7,8,9)"
    """
    ids_str = ','.join(str(i) for i in EXCLUDED_USER_IDS)
    return f"AND {table_alias}.{col} NOT IN ({ids_str})"

def excluded_users_filter_col(col='user_id'):
    """
    Versi tanpa alias tabel — untuk subquery yang langsung pakai nama kolom.
    Contoh: excluded_users_filter_col('ujian_user_id')
    """
    ids_str = ','.join(str(i) for i in EXCLUDED_USER_IDS)
    return f"AND {col} NOT IN ({ids_str})"


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_level_filter(kelas_filter, table_alias='u'):
    col = f"{table_alias}.level"
    excl = excluded_users_filter(table_alias)
    if kelas_filter:
        return f"{col} NOT IN ('1','2') AND {col} = %s {excl}", (kelas_filter,)
    return f"{col} NOT IN ('1','2') {excl}", ()


def build_material_filter(material_id, table_alias='l', col='id_material'):
    """
    Tambahkan filter id_material ke query.
    Kembalikan (where_snippet, params_tuple).
    Jika material_id None/kosong → tidak ada filter tambahan.
    """
    if material_id:
        return f"AND {table_alias}.{col} = %s", (int(material_id),)
    return "", ()


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

POIN_MAP = {'Ideal': 8, 'Normal': 6, 'Gaming': 4, 'Struggling': 2}

def get_kategori(steps, waktu):
    if   steps <= 4 and waktu <= 58: return 'Ideal'
    elif steps <= 5 and waktu <= 80: return 'Gaming'
    elif steps <= 5 and waktu <= 81: return 'Normal'
    else:                            return 'Struggling'


def _safe_in_clause(id_list):
    if len(id_list) == 1:
        return f"({id_list[0]})", ()
    return "%s", (tuple(id_list),)


SUMPOIN_NODE_ORDER = ['1-2', '3-4', '5-6', '7-8']

def sumpoin_to_node(raw_avg):
    v = float(raw_avg or 0)
    if v <= 2.0:   return '1-2'
    elif v <= 4.0: return '3-4'
    elif v <= 6.0: return '5-6'
    else:          return '7-8'


def compute_behavior_map(cursor, level_where, level_params, valid_user_ids, material_id=None):
    """
    Hitung behavior dominan per user menggunakan majority vote per latihan.
    Filter id_material jika diberikan.
    """
    if not valid_user_ids:
        return {}

    uid_list   = list(valid_user_ids)
    in_clause, in_params = _safe_in_clause(uid_list)

    mat_where, mat_params = build_material_filter(material_id, table_alias='l')

    cursor.execute(f"""
        SELECT
            u.id                        AS user_id,
            l.id_material,
            l.id_latihan,
            COUNT(DISTINCT l.id)        AS step_per_latihan,
            COALESCE(SUM(l.time), 0)    AS waktu_per_latihan
        FROM users u
        INNER JOIN log l ON l.email = u.email
        WHERE {level_where}
          AND u.id IN {in_clause}
          {mat_where}
        GROUP BY u.id, l.id_material, l.id_latihan
    """, level_params + in_params + mat_params)

    log_rows = cursor.fetchall()

    user_kat_count = defaultdict(lambda: defaultdict(int))
    for row in log_rows:
        uid   = row['user_id']
        steps = int(row['step_per_latihan'] or 0)
        waktu = float(row['waktu_per_latihan'] or 0)
        kat   = get_kategori(steps, waktu)
        user_kat_count[uid][kat] += 1

    behavior_map = {}
    for uid in valid_user_ids:
        if uid not in user_kat_count or not user_kat_count[uid]:
            behavior_map[uid] = 'Struggling'
            continue
        kat_counts = user_kat_count[uid]
        final_kat  = max(
            kat_counts.items(),
            key=lambda x: (x[1], POIN_MAP[x[0]])
        )[0]
        behavior_map[uid] = final_kat

    return behavior_map


def compute_sumpoin_map(cursor, level_where, level_params, valid_user_ids,
                        mode_where='', mode_params=(), material_id=None):
    """
    Ambil nilai TERBARU (1 baris) per user dari result_adaptive.
    Filter id_material jika diberikan.
    """
    if not valid_user_ids:
        return {}

    uid_list             = list(valid_user_ids)
    in_clause, in_params = _safe_in_clause(uid_list)

    excl_col = excluded_users_filter_col('ujian_user_id')

    # Filter material untuk result_adaptive
    mat_where_ra  = f"AND ra.id_material = %s" if material_id else ""
    mat_params_ra = (int(material_id),) if material_id else ()

    mat_where_sub  = f"AND id_material = %s" if material_id else ""
    mat_params_sub = (int(material_id),) if material_id else ()

    cursor.execute(f"""
        SELECT
            ra.ujian_user_id AS user_id,
            ra.nilai         AS sumpoin
        FROM result_adaptive ra
        INNER JOIN ujian_peserta jp ON jp.id = ra.id_ujian_peserta
        INNER JOIN ujian uj         ON uj.id = jp.ujian_id
        INNER JOIN users u          ON u.id  = ra.ujian_user_id
        WHERE ra.id IN (
            SELECT MAX(id)
            FROM result_adaptive
            WHERE ujian_user_id IN {in_clause}
            {mat_where_sub}
            {excl_col}
            GROUP BY ujian_user_id
        )
        AND {level_where}
        AND ra.ujian_user_id IN {in_clause}
        {mat_where_ra}
        {mode_where}
    """, in_params + mat_params_sub + level_params + in_params + mat_params_ra + mode_params)

    return {
        r['user_id']: float(r['sumpoin'] or 0)
        for r in cursor.fetchall()
    }


def fetch_user_test_map(cursor, material_id=None, level_where=None, level_params=()):
    """
    Ambil semua user yang punya pretest DAN posttest valid (1-20).
    Filter kelas (level_where/level_params) dan id_material jika diberikan.
    """
    mat_pre  = "AND pre.id_material = %s"  if material_id else ""
    mat_post = "AND post.id_material = %s" if material_id else ""
    mat_p    = ()
    if material_id:
        mat_p = (int(material_id), int(material_id))

    # Filter kelas via JOIN ke users
    if level_where:
        kelas_join   = "INNER JOIN users u ON u.id = pre.user_id"
        kelas_filter = f"AND {level_where}"
        params       = mat_p + level_params
    else:
        kelas_join   = ""
        kelas_filter = ""
        params       = mat_p

    # Filter excluded users (pre.user_id sudah cukup karena JOIN ke posttest)
    excl = excluded_users_filter_col('pre.user_id')

    cursor.execute(f"""
        SELECT
            pre.user_id,
            ROUND(pre.pretest)   AS nilai_pre,
            ROUND(post.posttest) AS nilai_post
        FROM pretest pre
        INNER JOIN posttest post ON post.user_id = pre.user_id
        {kelas_join}
        WHERE pre.pretest  BETWEEN 1 AND 20
          AND post.posttest BETWEEN 1 AND 20
          {mat_pre}
          {mat_post}
          {excl}
          {kelas_filter}
    """, params)

    rows = cursor.fetchall()
    return {
        r['user_id']: (int(r['nilai_pre']), int(r['nilai_post']))
        for r in rows
    }


def fetch_peserta_detail(cursor, matched_ids, pre_sub, post_sub):
    """
    Ambil detail peserta berdasarkan list user_id.
    """
    if not matched_ids:
        return []

    # Buang excluded IDs dari matched_ids untuk keamanan berlapis
    matched_ids = [uid for uid in matched_ids if uid not in EXCLUDED_USER_IDS]
    if not matched_ids:
        return []

    in_clause, in_params = _safe_in_clause(list(matched_ids))

    cursor.execute(f"""
        SELECT
            u.id,
            u.nama,
            u.level              AS kelas,
            COUNT(DISTINCT l.id) AS total_step,
            ROUND(AVG(l.time))   AS avg_waktu,
            pre.nilai            AS pre_test,
            post.nilai           AS post_test
        FROM users u
        LEFT  JOIN log l          ON l.email     = u.email
        INNER JOIN {pre_sub}  pre  ON pre.user_id  = u.id
        INNER JOIN {post_sub} post ON post.user_id = u.id
        WHERE u.id IN {in_clause}
        GROUP BY u.id, u.nama, u.level, pre.nilai, post.nilai
        ORDER BY u.nama
    """, in_params)

    return cursor.fetchall()


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD SANKEY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_sankey_3col(cursor, right_table, right_col, level_where, level_params, material_id=None):
    mat_log     = "AND l.id_material = %s" if material_id else ""
    mat_log_p   = (int(material_id),)      if material_id else ()

    mat_max_sub = "AND id_material = %s"   if material_id else ""
    mat_max_p   = (int(material_id),)      if material_id else ()

    mat_outer   = "AND t.id_material = %s" if material_id else ""
    mat_outer_p = (int(material_id),)      if material_id else ()

    excl_sub   = excluded_users_filter_col('user_id')
    excl_outer = excluded_users_filter('t', 'user_id')

    cursor.execute(f"""
        SELECT
            CASE
                WHEN total_step BETWEEN 1  AND 10  THEN '1-10'
                WHEN total_step BETWEEN 11 AND 50  THEN '11-50'
                WHEN total_step BETWEEN 51 AND 100 THEN '51-100'
                ELSE '100+'
            END AS step_node,
            CASE
                WHEN avg_time < 30                THEN '<30 dtk'
                WHEN avg_time BETWEEN 30  AND 60  THEN '30-60 dtk'
                WHEN avg_time BETWEEN 61  AND 120 THEN '61-120 dtk'
                ELSE '>120 dtk'
            END AS time_node,
            t.nilai AS nilai_node,
            COUNT(*) AS jumlah
        FROM (
            SELECT
                u.id,
                COUNT(DISTINCT l.id)  AS total_step,
                ROUND(AVG(l.time))    AS avg_time
            FROM users u
            INNER JOIN log l ON l.email = u.email
            WHERE {level_where}
            {mat_log}
            GROUP BY u.id
        ) sub
        INNER JOIN (
            SELECT t.user_id, ROUND(t.{right_col}) AS nilai
            FROM {right_table} t
            INNER JOIN users u ON u.id = t.user_id
            WHERE t.id IN (
                SELECT MAX(id)
                FROM {right_table}
                WHERE 1=1
                {mat_max_sub}
                {excl_sub}
                GROUP BY user_id
            )
            AND t.{right_col} BETWEEN 1 AND 20
            AND {level_where}
            {mat_outer}
            {excl_outer}
        ) t ON t.user_id = sub.id
        GROUP BY step_node, time_node, nilai_node
        HAVING jumlah > 0
        ORDER BY step_node, time_node, nilai_node
    """, level_params + mat_log_p + mat_max_p + level_params + mat_outer_p)

    rows = cursor.fetchall()

    link_step_time  = defaultdict(int)
    link_time_nilai = defaultdict(int)

    for row in rows:
        link_step_time[ (str(row['step_node']), str(row['time_node'])  )] += int(row['jumlah'])
        link_time_nilai[(str(row['time_node']), str(row['nilai_node']))] += int(row['jumlah'])

    links = []
    for (src, tgt), val in link_step_time.items():
        links.append({'source': src, 'target': tgt, 'value': val})
    for (src, tgt), val in link_time_nilai.items():
        links.append({'source': src, 'target': tgt, 'value': val})

    return links


def build_sankey_new(cursor, mode, right_table, right_col, level_where, level_params, material_id=None):
    if mode == 'step':
        left_expr   = "COUNT(DISTINCT l.id)"
        ranges_case = """
            CASE
                WHEN left_val BETWEEN 1   AND 10    THEN '1-10'
                WHEN left_val BETWEEN 11  AND 50    THEN '11-50'
                WHEN left_val BETWEEN 51  AND 100   THEN '51-100'
                ELSE '100+'
            END
        """
    else:
        left_expr   = "ROUND(AVG(l.time))"
        ranges_case = """
            CASE
                WHEN left_val < 30                 THEN '<30 dtk'
                WHEN left_val BETWEEN 30  AND 60   THEN '30-60 dtk'
                WHEN left_val BETWEEN 61  AND 120  THEN '61-120 dtk'
                ELSE '>120 dtk'
            END
        """

    mat_log      = "AND l.id_material = %s" if material_id else ""
    mat_log_p    = (int(material_id),)      if material_id else ()

    mat_max_sub  = "AND id_material = %s"   if material_id else ""
    mat_max_p    = (int(material_id),)      if material_id else ()

    mat_outer    = "AND t.id_material = %s" if material_id else ""
    mat_outer_p  = (int(material_id),)      if material_id else ()

    excl_sub   = excluded_users_filter_col('user_id')
    excl_outer = excluded_users_filter('t', 'user_id')

    cursor.execute(f"""
        SELECT
            {ranges_case} AS left_node,
            t.nilai        AS right_node,
            COUNT(*)       AS jumlah
        FROM (
            SELECT
                u.id,
                {left_expr} AS left_val
            FROM users u
            INNER JOIN log l ON l.email = u.email
            WHERE {level_where}
            {mat_log}
            GROUP BY u.id
        ) sub
        INNER JOIN (
            SELECT t.user_id, ROUND(t.{right_col}) AS nilai
            FROM {right_table} t
            INNER JOIN users u ON u.id = t.user_id
            WHERE t.id IN (
                SELECT MAX(id)
                FROM {right_table}
                WHERE 1=1
                {mat_max_sub}
                {excl_sub}
                GROUP BY user_id
            )
            AND t.{right_col} BETWEEN 1 AND 20
            AND {level_where}
            {mat_outer}
            {excl_outer}
        ) t ON t.user_id = sub.id
        GROUP BY left_node, right_node
        HAVING jumlah > 0
        ORDER BY left_node, right_node
    """, level_params + mat_log_p + mat_max_p + level_params + mat_outer_p)

    rows = cursor.fetchall()
    return [
        {'source': str(r['left_node']), 'target': str(r['right_node']), 'value': int(r['jumlah'])}
        for r in rows
    ]


def build_sankey_pre_behavior_post(cursor, level_where, level_params, material_id=None):
    user_test_map = fetch_user_test_map(cursor, material_id, level_where, level_params)
    if not user_test_map:
        return []

    valid_user_ids = set(user_test_map.keys())
    user_behavior  = compute_behavior_map(cursor, level_where, level_params, valid_user_ids, material_id)

    link_pre_behavior  = defaultdict(int)
    link_behavior_post = defaultdict(int)

    for uid, behavior in user_behavior.items():
        if uid in user_test_map:
            pre, post = user_test_map[uid]
            link_pre_behavior[(str(pre), behavior)] += 1
            link_behavior_post[(behavior, str(post))] += 1

    links = []
    for (src, tgt), val in link_pre_behavior.items():
        if val > 0:
            links.append({'source': src, 'target': tgt, 'value': val})
    for (src, tgt), val in link_behavior_post.items():
        if val > 0:
            links.append({'source': src, 'target': tgt, 'value': val})

    return links


def build_sankey_pre_sumpoin_post(cursor, level_where, level_params,
                                   mode_where='', mode_params=(), material_id=None):
    user_test_map = fetch_user_test_map(cursor, material_id, level_where, level_params)
    if not user_test_map:
        return []

    valid_user_ids = set(user_test_map.keys())
    sumpoin_map    = compute_sumpoin_map(
        cursor, level_where, level_params, valid_user_ids,
        mode_where, mode_params, material_id
    )

    link_pre_sumpoin  = defaultdict(int)
    link_sumpoin_post = defaultdict(int)

    for uid, (pre, post) in user_test_map.items():
        if uid not in sumpoin_map:
            continue
        if sumpoin_map[uid] == 0:
            continue

        raw_val = sumpoin_map[uid]
        node    = sumpoin_to_node(raw_val)
        link_pre_sumpoin[(str(pre), node)] += 1
        link_sumpoin_post[(node, str(post))] += 1

    links = []
    for (src, tgt), val in link_pre_sumpoin.items():
        if val > 0:
            links.append({'source': src, 'target': tgt, 'value': val})
    for (src, tgt), val in link_sumpoin_post.items():
        if val > 0:
            links.append({'source': src, 'target': tgt, 'value': val})

    return links


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: List Materials (untuk dropdown filter)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/materials', methods=['GET'])
def get_materials():
    """
    Kembalikan daftar material yang benar-benar punya data log/pretest/posttest.
    Hanya material yang tidak ter-soft-delete (deleted_at IS NULL).
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            excl = excluded_users_filter_col('l.user_id')
            cursor.execute(f"""
                SELECT DISTINCT m.id, m.name
                FROM materials m
                WHERE m.deleted_at IS NULL
                  AND (
                      EXISTS (
                          SELECT 1 FROM log l
                          INNER JOIN users u ON u.email = l.email
                          WHERE l.id_material = m.id
                            {excluded_users_filter('u')}
                      )
                   OR EXISTS (
                          SELECT 1 FROM pretest pt
                          WHERE pt.id_material = m.id
                            {excluded_users_filter_col('pt.user_id')}
                      )
                   OR EXISTS (
                          SELECT 1 FROM posttest po
                          WHERE po.id_material = m.id
                            {excluded_users_filter_col('po.user_id')}
                      )
                  )
                ORDER BY m.id ASC
            """)
            rows = cursor.fetchall()
        return jsonify({'status': 'success', 'data': rows})
    finally:
        conn.close()


# ─── Endpoint 1: Data dari tabel `log` ──────────────────────────────────────
@app.route('/api/log-stats', methods=['GET'])
def log_stats():
    excl_email = f"""
        AND email NOT IN (
            SELECT email FROM users WHERE id IN ({','.join(str(i) for i in EXCLUDED_USER_IDS)})
        )
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT id, email, nama, ground, warrant, time, created_date
                FROM log
                WHERE 1=1 {excl_email}
                ORDER BY created_date DESC
            """)
            rows = cursor.fetchall()

            cursor.execute(f"""
                SELECT
                    COUNT(*) AS total_log,
                    AVG(time) AS rata_rata_waktu,
                    MIN(time) AS waktu_tercepat,
                    MAX(time) AS waktu_terlama,
                    COUNT(DISTINCT email) AS total_peserta
                FROM log
                WHERE 1=1 {excl_email}
            """)
            stats = cursor.fetchone()

            cursor.execute(f"""
                SELECT ground, COUNT(*) AS jumlah
                FROM log
                WHERE 1=1 {excl_email}
                GROUP BY ground
                ORDER BY jumlah DESC
            """)
            ground_dist = cursor.fetchall()

            cursor.execute(f"""
                SELECT warrant, COUNT(*) AS jumlah
                FROM log
                WHERE 1=1 {excl_email}
                GROUP BY warrant
                ORDER BY jumlah DESC
            """)
            warrant_dist = cursor.fetchall()

        return jsonify({
            'status': 'success',
            'data': rows,
            'statistik': stats,
            'distribusi_ground': ground_dist,
            'distribusi_warrant': warrant_dist,
        })
    finally:
        conn.close()


# ─── Endpoint 2: Data nilai ──────────────────────────────────────────────────
@app.route('/api/nilai-stats', methods=['GET'])
def nilai_stats():
    excl = excluded_users_filter_col('ujian_user_id')
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT id, id_ujian_peserta, ujian_user_id,
                       id_material, session, nilai, created_at
                FROM ujian_peserta_jawaban
                WHERE 1=1 {excl}
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()

            cursor.execute(f"""
                SELECT
                    COUNT(*) AS total_jawaban,
                    SUM(CASE WHEN nilai = 1 THEN 1 ELSE 0 END) AS total_benar,
                    SUM(CASE WHEN nilai = 0 THEN 1 ELSE 0 END) AS total_salah,
                    ROUND(AVG(nilai) * 100, 2) AS persentase_benar
                FROM ujian_peserta_jawaban
                WHERE 1=1 {excl}
            """)
            stats = cursor.fetchone()

            cursor.execute(f"""
                SELECT
                    id_material,
                    COUNT(*) AS total_soal,
                    SUM(CASE WHEN nilai = 1 THEN 1 ELSE 0 END) AS benar,
                    ROUND(AVG(nilai) * 100, 2) AS persentase_benar
                FROM ujian_peserta_jawaban
                WHERE 1=1 {excl}
                GROUP BY id_material
                ORDER BY id_material
            """)
            per_material = cursor.fetchall()

        return jsonify({
            'status': 'success',
            'data': rows,
            'statistik': stats,
            'per_material': per_material,
        })
    finally:
        conn.close()


# ─── Endpoint 3: Ringkasan Dashboard ─────────────────────────────────────────
@app.route('/api/dashboard-summary', methods=['GET'])
def dashboard_summary():
    excl_email = f"""
        AND email NOT IN (
            SELECT email FROM users WHERE id IN ({','.join(str(i) for i in EXCLUDED_USER_IDS)})
        )
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT
                    COUNT(*) AS total_log,
                    COUNT(DISTINCT email) AS total_peserta,
                    ROUND(AVG(time), 2) AS rata_rata_waktu_detik,
                    MIN(time) AS waktu_tercepat,
                    MAX(time) AS waktu_terlama
                FROM log
                WHERE 1=1 {excl_email}
            """)
            log_summary = cursor.fetchone()

            cursor.execute(f"""
                SELECT DATE(created_date) AS tanggal, COUNT(*) AS jumlah
                FROM log
                WHERE created_date >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                  {excl_email}
                GROUP BY DATE(created_date)
                ORDER BY tanggal ASC
            """)
            log_trend = cursor.fetchall()

            excl_ujian = excluded_users_filter_col('ujian_user_id')
            cursor.execute(f"""
                SELECT
                    COUNT(*) AS total_jawaban,
                    SUM(CASE WHEN nilai = 1 THEN 1 ELSE 0 END) AS total_benar,
                    ROUND(AVG(nilai) * 100, 2) AS persentase_benar
                FROM ujian_peserta_jawaban
                WHERE 1=1 {excl_ujian}
            """)
            nilai_summary = cursor.fetchone()

        return jsonify({
            'status': 'success',
            'log_summary': log_summary,
            'log_trend': log_trend,
            'nilai_summary': nilai_summary,
        })
    finally:
        conn.close()


# ─── Endpoint 4: Total Step per User ─────────────────────────────────────────
@app.route('/api/total-step', methods=['GET'])
def total_step():
    kelas_filter   = request.args.get('kelas', None)
    email_filter   = request.args.get('email', None)
    material_id    = request.args.get('material_id', None)
    limit          = int(request.args.get('limit', 50))

    level_where, level_params = build_level_filter(kelas_filter)
    mat_where, mat_params     = build_material_filter(material_id)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            # ── Summary: hitung dari subquery, bukan outer JOIN ke raw log ──
            cursor.execute(f"""
                SELECT
                    SUM(sub.total_step)           AS grand_total_step,
                    COUNT(sub.email)              AS total_user_aktif,
                    ROUND(AVG(sub.total_step), 2) AS rata_rata_step_per_user,
                    MAX(sub.total_step)           AS step_terbanyak,
                    MIN(sub.total_step)           AS step_tersedikit
                FROM (
                    SELECT u.email, COUNT(DISTINCT l.id) AS total_step
                    FROM users u
                    LEFT JOIN log l ON l.email = u.email
                    WHERE {level_where}
                    {mat_where}
                    GROUP BY u.email
                ) sub
            """, level_params + mat_params)
            summary = cursor.fetchone()

            # Ubah build_material_filter untuk bisa dipakai di JOIN
            mat_join = f"AND l.id_material = %s" if material_id else ""

            if email_filter:
                cursor.execute(f"""
                    SELECT
                        u.id         AS user_id,
                        u.nama,
                        u.email,
                        u.level,
                        COUNT(DISTINCT l.id)  AS total_step,
                        SUM(l.time)           AS total_waktu_detik,
                        ROUND(AVG(l.time), 2) AS rata_rata_waktu_detik,
                        MIN(l.created_date)   AS pertama_aktif,
                        MAX(l.created_date)   AS terakhir_aktif
                    FROM users u
                    LEFT JOIN log l ON l.email = u.email {mat_join}
                    WHERE u.email = %s
                    AND u.level NOT IN ('1','2')
                    {excluded_users_filter('u')}
                    GROUP BY u.id, u.nama, u.email, u.level
                """, mat_params + (email_filter,))
            else:
                cursor.execute(f"""
                    SELECT
                        u.id         AS user_id,
                        u.nama,
                        u.email,
                        u.level,
                        COUNT(DISTINCT l.id)  AS total_step,
                        SUM(l.time)           AS total_waktu_detik,
                        ROUND(AVG(l.time), 2) AS rata_rata_waktu_detik,
                        MIN(l.created_date)   AS pertama_aktif,
                        MAX(l.created_date)   AS terakhir_aktif
                    FROM users u
                    LEFT JOIN log l ON l.email = u.email {mat_join}
                    WHERE {level_where}
                    {excluded_users_filter('u')}
                    GROUP BY u.id, u.nama, u.email, u.level
                    ORDER BY total_step DESC
                    LIMIT %s
                """, mat_params + level_params + (limit,))
            per_user = cursor.fetchall()

            # ── Distribusi step: sudah ikut level_where + mat_where ──
            cursor.execute(f"""
                SELECT
                    CASE
                        WHEN total_step = 0                  THEN '0'
                        WHEN total_step BETWEEN 1  AND 10   THEN '1-10'
                        WHEN total_step BETWEEN 11 AND 50   THEN '11-50'
                        WHEN total_step BETWEEN 51 AND 100  THEN '51-100'
                        WHEN total_step BETWEEN 101 AND 500 THEN '101-500'
                        ELSE '500+'
                    END AS rentang,
                    COUNT(*) AS jumlah_user
                FROM (
                    SELECT u.email, COUNT(DISTINCT l.id) AS total_step
                    FROM users u
                    LEFT JOIN log l ON l.email = u.email
                    WHERE {level_where}
                    {mat_where}
                    GROUP BY u.email
                ) sub
                GROUP BY rentang
                ORDER BY MIN(sub.total_step)
            """, level_params + mat_params)
            distribusi = cursor.fetchall()

            # ── Top 10 aktif: sudah ikut level_where + mat_where ──
            cursor.execute(f"""
                SELECT
                    u.nama,
                    u.email,
                    u.level,
                    COUNT(DISTINCT l.id) AS total_step
                FROM users u
                JOIN log l ON l.email = u.email
                WHERE {level_where}
                {mat_where}
                GROUP BY u.id, u.nama, u.email, u.level
                ORDER BY total_step DESC
                LIMIT 10
            """, level_params + mat_params)
            top_aktif = cursor.fetchall()

        return jsonify({
            'status'    : 'success',
            'summary'   : summary,
            'per_user'  : per_user,
            'distribusi': distribusi,
            'top_aktif' : top_aktif,
        })
    finally:
        conn.close()


# ─── Endpoint 5: Chart Data ───────────────────────────────────────────────────
@app.route('/api/chart-data', methods=['GET'])
def chart_data():
    kelas_filter = request.args.get('kelas', None)
    material_id  = request.args.get('material_id', None)
    level_where, level_params = build_level_filter(kelas_filter)
    mat_log_where, mat_log_p  = build_material_filter(material_id, table_alias='l')
    mat_upj_where, mat_upj_p  = build_material_filter(material_id, table_alias='upj')

    excl_ujian = excluded_users_filter_col('ujian_user_id')

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            # 1. Pie Chart: Kategori Peserta
            cursor.execute(f"""
                SELECT
                    l.email,
                    l.id_material,
                    l.id_latihan,
                    COUNT(DISTINCT l.id)     AS step_per_latihan,
                    COALESCE(SUM(l.time), 0) AS waktu_per_latihan
                FROM log l
                JOIN users u ON u.email = l.email
                WHERE {level_where}
                {mat_log_where}
                GROUP BY l.email, l.id_material, l.id_latihan
            """, level_params + mat_log_p)
            rows = cursor.fetchall()

            user_kat_count = defaultdict(lambda: defaultdict(int))
            for row in rows:
                steps = int(row['step_per_latihan'] or 0)
                waktu = float(row['waktu_per_latihan'] or 0)
                kat   = get_kategori(steps, waktu)
                user_kat_count[row['email']][kat] += 1

            kategori_count = {
                'Ideal':      {'jumlah': 0, 'total_latihan': 0},
                'Normal':     {'jumlah': 0, 'total_latihan': 0},
                'Gaming':     {'jumlah': 0, 'total_latihan': 0},
                'Struggling': {'jumlah': 0, 'total_latihan': 0},
            }

            for email, kat_counts in user_kat_count.items():
                final_kat = max(
                    kat_counts.items(),
                    key=lambda x: (x[1], POIN_MAP[x[0]])
                )[0]
                total_latihan = sum(kat_counts.values())
                kategori_count[final_kat]['jumlah']        += 1
                kategori_count[final_kat]['total_latihan'] += total_latihan

            viz_kategori_peserta = {
                kat: {'jumlah': data['jumlah']}
                for kat, data in kategori_count.items()
            }

            # 2. Bar Graph
            mat_upj_bare = f"AND id_material = %s" if material_id else ""
            mat_upj_bp   = (int(material_id),) if material_id else ()

            cursor.execute(f"""
                SELECT
                    id_material,
                    session,
                    COUNT(*)                                     AS total_soal,
                    SUM(CASE WHEN nilai = 1 THEN 1 ELSE 0 END)  AS total_benar,
                    ROUND(AVG(nilai) * 100, 2)                   AS persen_benar,
                    COUNT(DISTINCT ujian_user_id)                AS jumlah_peserta
                FROM ujian_peserta_jawaban
                WHERE 1=1 {mat_upj_bare} {excl_ujian}
                GROUP BY id_material, session
                ORDER BY id_material, session
            """, mat_upj_bp)
            bar_material = cursor.fetchall()

            # 3. Sankey lama
            mat_log_sankey = f"AND l.id_material = %s" if material_id else ""
            mat_log_sp     = (int(material_id),) if material_id else ()
            n_sankey_rep   = 4

            sankey_query = f"""
                SELECT step_node, tipe_arg, nilai_node, SUM(jumlah) AS jumlah
                FROM (
                    SELECT
                        CASE
                            WHEN total_step BETWEEN 1  AND 10  THEN 'Step 1-10'
                            WHEN total_step BETWEEN 11 AND 50  THEN 'Step 11-50'
                            WHEN total_step BETWEEN 51 AND 100 THEN 'Step 51-100'
                            ELSE 'Step 100+'
                        END AS step_node,
                        'Ground Benar' AS tipe_arg,
                        CASE
                            WHEN avg_nilai >= 8 THEN 'Nilai Tinggi (≥8)'
                            WHEN avg_nilai >= 6 THEN 'Nilai Sedang (6-7)'
                            ELSE 'Nilai Rendah (<6)'
                        END AS nilai_node,
                        jumlah_benar AS jumlah
                    FROM (
                        SELECT u.id, COUNT(l.id) AS total_step,
                            COUNT(CASE WHEN l.confirm = 'true' THEN 1 END) AS jumlah_benar,
                            COALESCE((SELECT ROUND(AVG(v.nilai),2) FROM (
                                SELECT it.inittest AS nilai FROM inittest it WHERE it.user_id=u.id
                                UNION ALL SELECT pr.pretest FROM pretest pr WHERE pr.user_id=u.id
                                UNION ALL SELECT po.posttest FROM posttest po WHERE po.user_id=u.id
                            ) v),0) AS avg_nilai
                        FROM users u INNER JOIN log l ON l.email=u.email
                        WHERE {level_where} {mat_log_sankey} GROUP BY u.id HAVING COUNT(l.id)>0
                    ) sub WHERE jumlah_benar > 0
                    UNION ALL
                    SELECT
                        CASE WHEN total_step BETWEEN 1 AND 10 THEN 'Step 1-10'
                             WHEN total_step BETWEEN 11 AND 50 THEN 'Step 11-50'
                             WHEN total_step BETWEEN 51 AND 100 THEN 'Step 51-100'
                             ELSE 'Step 100+' END AS step_node,
                        'Ground Salah' AS tipe_arg,
                        CASE WHEN avg_nilai>=8 THEN 'Nilai Tinggi (≥8)'
                             WHEN avg_nilai>=6 THEN 'Nilai Sedang (6-7)'
                             ELSE 'Nilai Rendah (<6)' END AS nilai_node,
                        jumlah_salah AS jumlah
                    FROM (
                        SELECT u.id, COUNT(l.id) AS total_step,
                            COUNT(CASE WHEN l.confirm!='true' THEN 1 END) AS jumlah_salah,
                            COALESCE((SELECT ROUND(AVG(v.nilai),2) FROM (
                                SELECT it.inittest AS nilai FROM inittest it WHERE it.user_id=u.id
                                UNION ALL SELECT pr.pretest FROM pretest pr WHERE pr.user_id=u.id
                                UNION ALL SELECT po.posttest FROM posttest po WHERE po.user_id=u.id
                            ) v),0) AS avg_nilai
                        FROM users u INNER JOIN log l ON l.email=u.email
                        WHERE {level_where} {mat_log_sankey} GROUP BY u.id HAVING COUNT(l.id)>0
                    ) sub WHERE jumlah_salah > 0
                    UNION ALL
                    SELECT
                        CASE WHEN total_step BETWEEN 1 AND 10 THEN 'Step 1-10'
                             WHEN total_step BETWEEN 11 AND 50 THEN 'Step 11-50'
                             WHEN total_step BETWEEN 51 AND 100 THEN 'Step 51-100'
                             ELSE 'Step 100+' END AS step_node,
                        'Warrant Benar' AS tipe_arg,
                        CASE WHEN avg_nilai>=8 THEN 'Nilai Tinggi (≥8)'
                             WHEN avg_nilai>=6 THEN 'Nilai Sedang (6-7)'
                             ELSE 'Nilai Rendah (<6)' END AS nilai_node,
                        jumlah_war_benar AS jumlah
                    FROM (
                        SELECT u.id, COUNT(l.id) AS total_step,
                            COUNT(CASE WHEN l.war_conf='sure' THEN 1 END) AS jumlah_war_benar,
                            COALESCE((SELECT ROUND(AVG(v.nilai),2) FROM (
                                SELECT it.inittest AS nilai FROM inittest it WHERE it.user_id=u.id
                                UNION ALL SELECT pr.pretest FROM pretest pr WHERE pr.user_id=u.id
                                UNION ALL SELECT po.posttest FROM posttest po WHERE po.user_id=u.id
                            ) v),0) AS avg_nilai
                        FROM users u INNER JOIN log l ON l.email=u.email
                        WHERE {level_where} {mat_log_sankey} GROUP BY u.id HAVING COUNT(l.id)>0
                    ) sub WHERE jumlah_war_benar > 0
                    UNION ALL
                    SELECT
                        CASE WHEN total_step BETWEEN 1 AND 10 THEN 'Step 1-10'
                             WHEN total_step BETWEEN 11 AND 50 THEN 'Step 11-50'
                             WHEN total_step BETWEEN 51 AND 100 THEN 'Step 51-100'
                             ELSE 'Step 100+' END AS step_node,
                        'Warrant Salah' AS tipe_arg,
                        CASE WHEN avg_nilai>=8 THEN 'Nilai Tinggi (≥8)'
                             WHEN avg_nilai>=6 THEN 'Nilai Sedang (6-7)'
                             ELSE 'Nilai Rendah (<6)' END AS nilai_node,
                        jumlah_war_salah AS jumlah
                    FROM (
                        SELECT u.id, COUNT(l.id) AS total_step,
                            COUNT(CASE WHEN l.war_conf!='sure' THEN 1 END) AS jumlah_war_salah,
                            COALESCE((SELECT ROUND(AVG(v.nilai),2) FROM (
                                SELECT it.inittest AS nilai FROM inittest it WHERE it.user_id=u.id
                                UNION ALL SELECT pr.pretest FROM pretest pr WHERE pr.user_id=u.id
                                UNION ALL SELECT po.posttest FROM posttest po WHERE po.user_id=u.id
                            ) v),0) AS avg_nilai
                        FROM users u INNER JOIN log l ON l.email=u.email
                        WHERE {level_where} {mat_log_sankey} GROUP BY u.id HAVING COUNT(l.id)>0
                    ) sub WHERE jumlah_war_salah > 0
                ) combined
                GROUP BY step_node, tipe_arg, nilai_node
                HAVING jumlah > 0
                ORDER BY step_node, tipe_arg, nilai_node
            """
            cursor.execute(sankey_query, (level_params + mat_log_sp) * n_sankey_rep)
            sankey_raw = cursor.fetchall()

            link_step_arg  = defaultdict(int)
            link_arg_nilai = defaultdict(int)
            for row in sankey_raw:
                link_step_arg[ (row['step_node'], row['tipe_arg']  )] += int(row['jumlah'])
                link_arg_nilai[(row['tipe_arg'],  row['nilai_node'])] += int(row['jumlah'])

            sankey_links = []
            for (src, tgt), val in link_step_arg.items():
                sankey_links.append({'source': src, 'target': tgt, 'value': val})
            for (src, tgt), val in link_arg_nilai.items():
                sankey_links.append({'source': src, 'target': tgt, 'value': val})

            # 4. Regresi
            cursor.execute(f"""
                SELECT sub.total_step,
                    ROUND(sub.avg_waktu, 2)    AS avg_waktu_detik,
                    ROUND(sub.persen_nilai, 2) AS persen_nilai
                FROM (
                    SELECT u.email,
                        COUNT(DISTINCT l.id)     AS total_step,
                        COALESCE(AVG(l.time), 0) AS avg_waktu,
                        COALESCE((SELECT ROUND(AVG(upj.nilai)*100,2)
                                  FROM ujian_peserta_jawaban upj
                                  WHERE upj.ujian_user_id=u.email),0) AS persen_nilai
                    FROM users u
                    INNER JOIN log l ON l.email=u.email
                    WHERE {level_where}
                    {mat_log_where}
                    GROUP BY u.email
                    HAVING total_step > 0
                ) sub
                WHERE sub.persen_nilai > 0
                ORDER BY sub.total_step ASC
            """, level_params + mat_log_p)
            regresi_raw = cursor.fetchall()

            # 5. Sankey Baru
            sankey_step_post      = build_sankey_new(cursor, 'step', 'posttest', 'posttest', level_where, level_params, material_id)
            sankey_time_post      = build_sankey_new(cursor, 'time', 'posttest', 'posttest', level_where, level_params, material_id)
            sankey_step_time_post = build_sankey_3col(cursor, 'posttest', 'posttest', level_where, level_params, material_id)

            # 6. Pre→Behavior→Post & Pre→Sumpoin→Post
            sankey_pre_behavior_post = build_sankey_pre_behavior_post(cursor, level_where, level_params, material_id)
            sankey_pre_sumpoin_post  = build_sankey_pre_sumpoin_post(cursor, level_where, level_params, material_id=material_id)

        return jsonify({
            'status'                  : 'success',
            'viz_kategori_peserta'    : viz_kategori_peserta,
            'bar_material'            : bar_material,
            'sankey_links'            : sankey_links,
            'regresi_data'            : regresi_raw,
            'sankey_step_post'        : sankey_step_post,
            'sankey_time_post'        : sankey_time_post,
            'sankey_step_time_post'   : sankey_step_time_post,
            'sankey_pre_behavior_post': sankey_pre_behavior_post,
            'sankey_pre_sumpoin_post' : sankey_pre_sumpoin_post,
        })
    finally:
        conn.close()


# ─── Endpoint 6: Detail peserta per band Sankey ──────────────────────────────
@app.route('/api/sankey-detail', methods=['GET'])
def sankey_detail():
    source       = request.args.get('source', '')
    target       = request.args.get('target', '')
    tab          = request.args.get('tab', '')
    kelas_filter = request.args.get('kelas', None)
    material_id  = request.args.get('material_id', None)

    if not source or not target or not tab:
        return jsonify({'status': 'error', 'message': 'Parameter source, target, tab wajib diisi'}), 400

    level_where, level_params = build_level_filter(kelas_filter)

    mat_pre_sub  = f"AND id_material = {int(material_id)}"  if material_id else ""
    mat_post_sub = f"AND id_material = {int(material_id)}"  if material_id else ""
    mat_log_join = f"AND l.id_material = {int(material_id)}" if material_id else ""

    ids_str = ','.join(str(i) for i in EXCLUDED_USER_IDS)

    def latest_val_subquery(table, col):
        mat_filter = f"AND id_material = {int(material_id)}" if material_id else ""
        return f"""
            (
                SELECT user_id, ROUND({col}) AS nilai
                FROM {table}
                WHERE id IN (
                    SELECT MAX(id) FROM {table}
                    WHERE 1=1 {mat_filter}
                    AND user_id NOT IN ({ids_str})
                    GROUP BY user_id
                )
                AND {col} BETWEEN 1 AND 20
                {mat_filter}
                AND user_id NOT IN ({ids_str})
            )
        """

    STEP_CASE = f"""
        CASE
            WHEN COUNT(DISTINCT l.id) BETWEEN 1  AND 10  THEN '1-10'
            WHEN COUNT(DISTINCT l.id) BETWEEN 11 AND 50  THEN '11-50'
            WHEN COUNT(DISTINCT l.id) BETWEEN 51 AND 100 THEN '51-100'
            ELSE '100+'
        END
    """
    TIME_CASE = f"""
        CASE
            WHEN ROUND(AVG(l.time)) < 30                THEN '<30 dtk'
            WHEN ROUND(AVG(l.time)) BETWEEN 30  AND 60  THEN '30-60 dtk'
            WHEN ROUND(AVG(l.time)) BETWEEN 61  AND 120 THEN '61-120 dtk'
            ELSE '>120 dtk'
        END
    """

    rows = []

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            if tab == 'step-post':
                cursor.execute(f"""
                    SELECT u.nama, u.level AS kelas,
                        COUNT(DISTINCT l.id) AS total_step,
                        ROUND(AVG(l.time))   AS avg_waktu,
                        t.nilai
                    FROM users u
                    INNER JOIN log l ON l.email = u.email
                    INNER JOIN {latest_val_subquery('posttest', 'posttest')} t ON t.user_id = u.id
                    WHERE {level_where}
                    {mat_log_join}
                    GROUP BY u.id, u.nama, u.level, t.nilai
                    HAVING {STEP_CASE} = %s AND t.nilai = %s
                    ORDER BY u.nama
                """, level_params + (source, int(target)))
                rows = cursor.fetchall()

            elif tab == 'time-post':
                cursor.execute(f"""
                    SELECT u.nama, u.level AS kelas,
                        COUNT(DISTINCT l.id) AS total_step,
                        ROUND(AVG(l.time))   AS avg_waktu,
                        t.nilai
                    FROM users u
                    INNER JOIN log l ON l.email = u.email
                    INNER JOIN {latest_val_subquery('posttest', 'posttest')} t ON t.user_id = u.id
                    WHERE {level_where}
                    {mat_log_join}
                    GROUP BY u.id, u.nama, u.level, t.nilai
                    HAVING {TIME_CASE} = %s AND t.nilai = %s
                    ORDER BY u.nama
                """, level_params + (source, int(target)))
                rows = cursor.fetchall()

            elif tab == 'step-time-post':
                TIME_LABELS = ['<30 dtk', '30-60 dtk', '61-120 dtk', '>120 dtk']
                if target in TIME_LABELS:
                    cursor.execute(f"""
                        SELECT u.nama, u.level AS kelas,
                            COUNT(DISTINCT l.id) AS total_step,
                            ROUND(AVG(l.time))   AS avg_waktu,
                            t.nilai
                        FROM users u
                        INNER JOIN log l ON l.email = u.email
                        INNER JOIN {latest_val_subquery('posttest', 'posttest')} t ON t.user_id = u.id
                        WHERE {level_where}
                        {mat_log_join}
                        GROUP BY u.id, u.nama, u.level, t.nilai
                        HAVING {STEP_CASE} = %s AND {TIME_CASE} = %s
                        ORDER BY u.nama
                    """, level_params + (source, target))
                else:
                    cursor.execute(f"""
                        SELECT u.nama, u.level AS kelas,
                            COUNT(DISTINCT l.id) AS total_step,
                            ROUND(AVG(l.time))   AS avg_waktu,
                            t.nilai
                        FROM users u
                        INNER JOIN log l ON l.email = u.email
                        INNER JOIN {latest_val_subquery('posttest', 'posttest')} t ON t.user_id = u.id
                        WHERE {level_where}
                        {mat_log_join}
                        GROUP BY u.id, u.nama, u.level, t.nilai
                        HAVING {TIME_CASE} = %s AND t.nilai = %s
                        ORDER BY u.nama
                    """, level_params + (source, int(target)))
                rows = cursor.fetchall()

            elif tab == 'pre-behavior-post':
                BEHAVIOR_LABELS = {'Ideal', 'Normal', 'Gaming', 'Struggling'}
                user_test_map = fetch_user_test_map(cursor, material_id, level_where, level_params)
                if not user_test_map:
                    rows = []
                else:
                    valid_user_ids = set(user_test_map.keys())
                    behavior_map = compute_behavior_map(cursor, level_where, level_params, valid_user_ids, material_id)

                    if target in BEHAVIOR_LABELS:
                        matched = [
                            uid for uid, (pre, _) in user_test_map.items()
                            if str(pre) == source and behavior_map.get(uid) == target
                        ]
                    else:
                        matched = [
                            uid for uid, (_, post) in user_test_map.items()
                            if behavior_map.get(uid) == source and str(post) == target
                        ]

                    pre_sub  = latest_val_subquery('pretest',  'pretest')
                    post_sub = latest_val_subquery('posttest', 'posttest')
                    raw_rows = fetch_peserta_detail(cursor, matched, pre_sub, post_sub)

                    rows = []
                    for r in raw_rows:
                        r = dict(r)
                        r['behavior'] = behavior_map.get(r['id'], 'Struggling')
                        rows.append(r)

            elif tab == 'pre-sumpoin-post':
                SUMPOIN_NODES = set(SUMPOIN_NODE_ORDER)
                user_test_map = fetch_user_test_map(cursor, material_id, level_where, level_params)
                if not user_test_map:
                    rows = []
                else:
                    valid_user_ids = set(user_test_map.keys())
                    sumpoin_map = compute_sumpoin_map(cursor, level_where, level_params, valid_user_ids, material_id=material_id)

                    side = request.args.get('side', None)
                    if side not in ('left', 'right'):
                        if target in SUMPOIN_NODES:
                            side = 'left'
                        elif source in SUMPOIN_NODES:
                            side = 'right'
                        else:
                            side = 'right'

                    if side == 'left':
                        matched = [
                            uid for uid, (pre, _) in user_test_map.items()
                            if uid in sumpoin_map
                            and sumpoin_map[uid] > 0
                            and str(pre) == source
                            and sumpoin_to_node(sumpoin_map[uid]) == target
                        ]
                    else:
                        matched = [
                            uid for uid, (_, post) in user_test_map.items()
                            if uid in sumpoin_map
                            and sumpoin_to_node(sumpoin_map[uid]) == source
                            and str(post) == target
                        ]

                    pre_sub  = latest_val_subquery('pretest',  'pretest')
                    post_sub = latest_val_subquery('posttest', 'posttest')
                    raw_rows = fetch_peserta_detail(cursor, matched, pre_sub, post_sub)

                    rows = []
                    for r in raw_rows:
                        r = dict(r)
                        uid = r['id']
                        raw_float = sumpoin_map.get(uid)
                        r['sum_poin'] = f"{raw_float:.4f}" if raw_float is not None else None
                        rows.append(r)

            else:
                return jsonify({'status': 'error', 'message': 'Tab tidak dikenali'}), 400

        return jsonify({
            'status'  : 'success',
            'source'  : source,
            'target'  : target,
            'tab'     : tab,
            'total'   : len(rows),
            'peserta' : rows,
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status' : 'error',
            'message': str(e),
            'trace'  : traceback.format_exc(),
        }), 500
    finally:
        conn.close()


# ─── Endpoint 7: Regresi ─────────────────────────────────────────────────────
@app.route('/api/regresi-nilai', methods=['GET'])
def regresi_nilai():
    kelas_filter = request.args.get('kelas', None)
    mode         = request.args.get('mode', '')
    material_id  = request.args.get('material_id', None)
    level_where, level_params = build_level_filter(kelas_filter)
    level_where_inner, level_params_inner = build_level_filter(kelas_filter, table_alias='u')

    mat_log_where, mat_log_p = build_material_filter(material_id, table_alias='l')
    mat_ra_where             = f"AND ra_sub.id_material = %s" if material_id else ""
    mat_ra_p                 = (int(material_id),) if material_id else ()
    mat_ra_latest            = f"AND ra_latest.id_material = %s" if material_id else ""
    mat_ra_lp                = (int(material_id),) if material_id else ()

    if mode == 'adaptive':
        mode_where = "AND uj.adaptive_learning = 1"
    elif mode == 'viatmap':
        mode_where = """
            AND (
                uj.type_latihan = 1
                OR (jp.type_ujian = 'viat_map' AND uj.adaptive_learning = 0)
            )
        """
    else:
        mode_where = """
            AND (
                uj.adaptive_learning = 1
                OR uj.type_latihan = 1
                OR jp.type_ujian = 'viat_map'
            )
        """

    excl_ra = excluded_users_filter_col('ra_sub.ujian_user_id')
    excl_ra_latest = excluded_users_filter_col('ra_latest.ujian_user_id')

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT
                    u.id                            AS user_id,
                    u.nama,
                    u.level,
                    COUNT(DISTINCT l.id)            AS total_step,
                    ROUND(AVG(l.time), 2)           AS avg_waktu_detik,
                    ROUND(latest_ra.avg_nilai, 4)   AS sum_poin
                FROM users u
                INNER JOIN log l ON l.email = u.email
                INNER JOIN (
                    SELECT
                        ra_latest.ujian_user_id,
                        AVG(ra_latest.nilai) AS avg_nilai
                    FROM result_adaptive ra_latest
                    INNER JOIN (
                        SELECT
                            ra_sub.ujian_user_id,
                            MAX(ra_sub.session) AS max_session
                        FROM result_adaptive ra_sub
                        INNER JOIN ujian_peserta jp ON jp.id = ra_sub.id_ujian_peserta
                        INNER JOIN ujian uj         ON uj.id = jp.ujian_id
                        INNER JOIN users u_sub      ON u_sub.id = ra_sub.ujian_user_id
                        WHERE 1=1 {mode_where} {mat_ra_where} {excl_ra}
                        GROUP BY ra_sub.ujian_user_id
                    ) latest_session
                        ON  latest_session.ujian_user_id = ra_latest.ujian_user_id
                        AND latest_session.max_session   = ra_latest.session
                    INNER JOIN ujian_peserta jp ON jp.id = ra_latest.id_ujian_peserta
                    INNER JOIN ujian uj         ON uj.id = jp.ujian_id
                    WHERE 1=1 {mode_where} {mat_ra_latest} {excl_ra_latest}
                    GROUP BY ra_latest.ujian_user_id
                ) latest_ra ON latest_ra.ujian_user_id = u.id
                WHERE {level_where_inner}
                {mat_log_where}
                GROUP BY u.id, u.nama, u.level, latest_ra.avg_nilai
                HAVING total_step > 0 AND sum_poin IS NOT NULL
                ORDER BY total_step ASC
            """, mat_ra_p + mat_ra_lp + level_params_inner + mat_log_p)

            rows = cursor.fetchall()

        return jsonify({
            'status' : 'success',
            'mode'   : mode or 'all',
            'total'  : len(rows),
            'data'   : rows,
        })
    finally:
        conn.close()


# ─── Endpoint 8: K-Means Clustering ──────────────────────────────────────────
@app.route('/api/kmeans-clustering', methods=['GET'])
def kmeans_clustering():
    kelas_filter = request.args.get('kelas', None)
    material_id  = request.args.get('material_id', None)
    n_clusters   = int(request.args.get('k', 3))
    level_where, level_params = build_level_filter(kelas_filter)

    mat_log_where, mat_log_p = build_material_filter(material_id, table_alias='l')
    mat_pre_where            = f"AND id_material = %s" if material_id else ""
    mat_pre_p                = (int(material_id),) if material_id else ()
    mat_post_where           = f"AND id_material = %s" if material_id else ""
    mat_post_p               = (int(material_id),) if material_id else ()
    mat_ra_where             = f"AND id_material = %s" if material_id else ""
    mat_ra_p                 = (int(material_id),) if material_id else ()

    excl_ra  = excluded_users_filter_col('ujian_user_id')
    excl_pre = excluded_users_filter_col('user_id')

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT
                    u.id                            AS user_id,
                    u.nama,
                    u.level                         AS kelas,
                    COUNT(DISTINCT l.id)            AS total_step,
                    ROUND(COALESCE(AVG(l.time), 0), 2) AS avg_waktu,
                    latest_ra.nilai                 AS sum_poin,
                    pre.nilai                       AS pretest,
                    post.nilai                      AS posttest
                FROM users u
                INNER JOIN log l ON l.email = u.email
                INNER JOIN (
                    SELECT ra.ujian_user_id, ra.nilai
                    FROM result_adaptive ra
                    WHERE ra.id IN (
                        SELECT MAX(ra2.id)
                        FROM result_adaptive ra2
                        WHERE 1=1 {mat_ra_where} {excl_ra}
                        GROUP BY ra2.ujian_user_id
                    )
                    {mat_ra_where}
                    {excl_ra}
                ) latest_ra ON latest_ra.ujian_user_id = u.id
                INNER JOIN (
                    SELECT user_id, ROUND(pretest) AS nilai
                    FROM pretest
                    WHERE id IN (
                        SELECT MAX(id) FROM pretest
                        WHERE 1=1 {mat_pre_where} {excl_pre}
                        GROUP BY user_id
                    )
                    AND pretest BETWEEN 1 AND 20
                    {mat_pre_where}
                    {excl_pre}
                ) pre ON pre.user_id = u.id
                INNER JOIN (
                    SELECT user_id, ROUND(posttest) AS nilai
                    FROM posttest
                    WHERE id IN (
                        SELECT MAX(id) FROM posttest
                        WHERE 1=1 {mat_post_where} {excl_pre}
                        GROUP BY user_id
                    )
                    AND posttest BETWEEN 1 AND 20
                    {mat_post_where}
                    {excl_pre}
                ) post ON post.user_id = u.id
                WHERE {level_where}
                {mat_log_where}
                GROUP BY u.id, u.nama, u.level, latest_ra.nilai, pre.nilai, post.nilai
                HAVING total_step > 0 AND sum_poin > 0
            """, mat_ra_p + mat_ra_p + mat_pre_p + mat_pre_p + mat_post_p + mat_post_p + level_params + mat_log_p)

            rows = cursor.fetchall()

        if len(rows) < n_clusters:
            return jsonify({
                'status' : 'error',
                'message': f'Data tidak cukup untuk {n_clusters} cluster (hanya {len(rows)} user)',
            }), 400

        features = ['total_step', 'avg_waktu', 'sum_poin', 'pretest', 'posttest']
        X_raw = np.array([
            [float(r[f] or 0) for f in features]
            for r in rows
        ])

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        centroids_scaled = kmeans.cluster_centers_
        centroids_real   = scaler.inverse_transform(centroids_scaled)

        centroid_list = []
        for i, c in enumerate(centroids_real):
            centroid_list.append({
                'cluster'    : i,
                'total_step' : round(float(c[0]), 2),
                'avg_waktu'  : round(float(c[1]), 2),
                'sum_poin'   : round(float(c[2]), 2),
                'pretest'    : round(float(c[3]), 2),
                'posttest'   : round(float(c[4]), 2),
            })

        from sklearn.metrics import silhouette_score
        sil_score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

        result = []
        for i, row in enumerate(rows):
            result.append({
                'user_id'    : row['user_id'],
                'nama'       : row['nama'],
                'kelas'      : row['kelas'],
                'total_step' : int(row['total_step'] or 0),
                'avg_waktu'  : float(row['avg_waktu'] or 0),
                'sum_poin'   : float(row['sum_poin'] or 0),
                'pretest'    : int(row['pretest'] or 0),
                'posttest'   : int(row['posttest'] or 0),
                'cluster'    : int(labels[i]),
            })

        cluster_stats = {}
        for i in range(n_clusters):
            members = [r for r in result if r['cluster'] == i]
            if not members:
                continue
            cluster_stats[i] = {
                'jumlah'         : len(members),
                'avg_total_step' : round(sum(m['total_step'] for m in members) / len(members), 2),
                'avg_waktu'      : round(sum(m['avg_waktu']  for m in members) / len(members), 2),
                'avg_sum_poin'   : round(sum(m['sum_poin']   for m in members) / len(members), 2),
                'avg_pretest'    : round(sum(m['pretest']     for m in members) / len(members), 2),
                'avg_posttest'   : round(sum(m['posttest']    for m in members) / len(members), 2),
            }

        return jsonify({
            'status'       : 'success',
            'k'            : n_clusters,
            'total_user'   : len(result),
            'silhouette'   : round(sil_score, 4),
            'inertia'      : round(float(kmeans.inertia_), 4),
            'centroids'    : centroid_list,
            'cluster_stats': cluster_stats,
            'data'         : result,
        })

    finally:
        conn.close()


# ─── Endpoint 9: Nilai per Kelas ─────────────────────────────────────────────
@app.route('/api/nilai-per-kelas', methods=['GET'])
def nilai_per_kelas():
    kelas_filter = request.args.get('kelas', None)
    material_id  = request.args.get('material_id', None)
    if material_id:
        material_id = int(material_id)

    excl = excluded_users_filter('u')

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:

            if kelas_filter:
                cursor.execute(f"""
                    SELECT DISTINCT level
                    FROM users u
                    WHERE level NOT IN ('1','2')
                      AND level = %s
                      AND level IS NOT NULL
                      {excl}
                    ORDER BY level
                """, (kelas_filter,))
            else:
                cursor.execute(f"""
                    SELECT DISTINCT level
                    FROM users u
                    WHERE level NOT IN ('1','2')
                      AND level IS NOT NULL
                      {excl}
                    ORDER BY level
                """)
            kelas_rows = cursor.fetchall()
            kelas_list = [r['level'] for r in kelas_rows]

            result = []
            for kelas in kelas_list:

                if material_id:
                    cursor.execute(f"""
                        SELECT ROUND(AVG(p.pretest), 2) AS avg_val
                        FROM pretest p
                        INNER JOIN users u ON u.id = p.user_id
                        WHERE u.level = %s
                          AND u.level NOT IN ('1','2')
                          AND p.id_material = %s
                          {excl}
                    """, (kelas, material_id))
                else:
                    cursor.execute(f"""
                        SELECT ROUND(AVG(p.pretest), 2) AS avg_val
                        FROM pretest p
                        INNER JOIN users u ON u.id = p.user_id
                        WHERE u.level = %s
                          AND u.level NOT IN ('1','2')
                          {excl}
                    """, (kelas,))
                row = cursor.fetchone()
                pretest_avg = float(row['avg_val']) if row and row['avg_val'] else None

                if material_id:
                    cursor.execute(f"""
                        SELECT ROUND(AVG(p.posttest), 2) AS avg_val
                        FROM posttest p
                        INNER JOIN users u ON u.id = p.user_id
                        WHERE u.level = %s
                          AND u.level NOT IN ('1','2')
                          AND p.id_material = %s
                          {excl}
                    """, (kelas, material_id))
                else:
                    cursor.execute(f"""
                        SELECT ROUND(AVG(p.posttest), 2) AS avg_val
                        FROM posttest p
                        INNER JOIN users u ON u.id = p.user_id
                        WHERE u.level = %s
                          AND u.level NOT IN ('1','2')
                          {excl}
                    """, (kelas,))
                row = cursor.fetchone()
                posttest_avg = float(row['avg_val']) if row and row['avg_val'] else None

                result.append({
                    'level'    : kelas,
                    'pretest'  : pretest_avg,
                    'posttest' : posttest_avg,
                })

        return jsonify({
            'status'      : 'success',
            'kelas_filter': kelas_filter,
            'material_id' : material_id,
            'data'        : result,
        })
    finally:
        conn.close()


# ─── Health check ────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Python API berjalan'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)