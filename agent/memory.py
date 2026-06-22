"""Step 6 — memory write-back: the system READS and WRITES, so it learns.

Without this the KG is read-only and goes stale. After a task finishes, the
orchestrator persists what it learned: a decision record ("chose X because Y") and
any new edges it discovered. Memories live in three tiers with a promotion policy:

    working  -> project  : proven useful (used >= PROMOTE_USES times)
    project  -> archive  : gone cold (untouched for ARCHIVE_AFTER_DAYS)

Nothing is ever silently dropped — cold memories are ARCHIVED (recoverable), not
deleted. Access is what promotes: recall() touches a row, so things that keep
proving useful rise, and things that don't sink. Plain stdlib sqlite3.
"""
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "runtime", "memory.db")

PROMOTE_USES = 3                     # working -> project after this many uses
ARCHIVE_AFTER_DAYS = 30              # project -> archive after this long untouched
_DAY = 86400

TIERS = ("working", "project", "archive")


def _conn(db_path=None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tier TEXT NOT NULL DEFAULT 'working',
        kind TEXT NOT NULL DEFAULT 'decision',
        content TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_used_at REAL NOT NULL,
        use_count INTEGER NOT NULL DEFAULT 0,
        archived INTEGER NOT NULL DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS edges(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE(src, dst, rel))""")
    return c


class Memory:
    def __init__(self, db_path=None):
        self.c = _conn(db_path)

    def close(self):
        self.c.close()

    # ---- write path ----
    def record(self, content, kind="decision", tier="working", now=None):
        now = time.time() if now is None else now
        cur = self.c.execute(
            "INSERT INTO memories(tier,kind,content,created_at,last_used_at) "
            "VALUES(?,?,?,?,?)", (tier, kind, content, now, now))
        self.c.commit()
        return cur.lastrowid

    def add_edge(self, src, dst, rel, now=None):
        now = time.time() if now is None else now
        self.c.execute(
            "INSERT OR IGNORE INTO edges(src,dst,rel,created_at) VALUES(?,?,?,?)",
            (src, dst, rel, now))
        self.c.commit()

    # ---- read path (access promotes) ----
    def recall(self, substring="", limit=10, now=None):
        now = time.time() if now is None else now
        rows = self.c.execute(
            "SELECT id,tier,kind,content,use_count FROM memories "
            "WHERE archived=0 AND content LIKE ? "
            "ORDER BY tier='project' DESC, use_count DESC, last_used_at DESC LIMIT ?",
            (f"%{substring}%", limit)).fetchall()
        for r in rows:
            self.c.execute(
                "UPDATE memories SET use_count=use_count+1, last_used_at=? WHERE id=?",
                (now, r[0]))
        self.c.commit()
        return [{"id": r[0], "tier": r[1], "kind": r[2],
                 "content": r[3], "use_count": r[4] + 1} for r in rows]

    # ---- promotion / eviction (never deletes) ----
    def promote(self, now=None):
        now = time.time() if now is None else now
        # working -> project: it keeps proving useful.
        up = self.c.execute(
            "UPDATE memories SET tier='project' "
            "WHERE tier='working' AND archived=0 AND use_count>=?",
            (PROMOTE_USES,)).rowcount
        # project -> archive: gone cold. Archived, NOT deleted.
        cutoff = now - ARCHIVE_AFTER_DAYS * _DAY
        arch = self.c.execute(
            "UPDATE memories SET tier='archive', archived=1 "
            "WHERE tier='project' AND last_used_at < ?", (cutoff,)).rowcount
        self.c.commit()
        return {"promoted": up, "archived": arch}

    def stats(self):
        rows = self.c.execute(
            "SELECT tier, COUNT(*) FROM memories GROUP BY tier").fetchall()
        return {t: n for t, n in rows}


def _selftest():
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "mem_test.db")
    m = Memory(db)
    t0 = 1_000_000.0

    a = m.record("chose Lloyd-Max quantization because recall stays >0.97 at 2-bit",
                 now=t0)
    b = m.record("merge_intervals bug: < should be <= for touching intervals", now=t0)
    m.add_edge("Bench", "BenchOptions", "USES", now=t0)
    assert m.stats().get("working") == 2

    # 'a' keeps getting recalled -> should promote to project.
    for _ in range(PROMOTE_USES):
        m.recall("quantization", now=t0 + 10)
    print("promote result:", m.promote(now=t0 + 20))
    assert m.stats().get("project") == 1, m.stats()

    # 'b' never touched again; far in the future it goes cold -> archive (not gone).
    m.recall("merge_intervals", now=t0)           # one touch to make it project-eligible
    for _ in range(PROMOTE_USES):
        m.recall("merge_intervals", now=t0)
    m.promote(now=t0)                              # b -> project
    future = t0 + (ARCHIVE_AFTER_DAYS + 5) * _DAY
    res = m.promote(now=future)
    print("archive result:", res)
    assert res["archived"] >= 1
    # Archived row still EXISTS (recoverable), just hidden from recall.
    total = m.c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert total == 2, "nothing was deleted"
    assert m.recall("merge_intervals", now=future) == [], "archived hidden from recall"

    print("stats:", m.stats())
    print("edges:", m.c.execute("SELECT src,dst,rel FROM edges").fetchall())
    print("OK: write-back, promotion, and archive-not-delete all verified.")
    m.close()


if __name__ == "__main__":
    _selftest()
