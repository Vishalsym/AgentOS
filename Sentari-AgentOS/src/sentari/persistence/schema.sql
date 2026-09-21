CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES agents(agent_id),
    state TEXT NOT NULL CHECK (state IN ('NEW','READY','RUNNING','BLOCKED','TERMINATED','KILLED')),
    priority INTEGER NOT NULL,
    quota_total INTEGER NOT NULL,
    quota_used INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS syscall_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    syscall_type TEXT NOT NULL CHECK (syscall_type IN ('tool_call','memory_read','memory_write','spawn','yield')),
    arguments TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('OK','WAIT','DENY','ERROR')),
    timestamp DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_allocation (
    allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_key TEXT NOT NULL,
    holder_agent_id TEXT REFERENCES agents(agent_id),
    waiter_agent_id TEXT REFERENCES agents(agent_id),
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    kb_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    last_writer_agent_id TEXT REFERENCES agents(agent_id),
    updated_at DATETIME NOT NULL,
    -- Extension beyond the original 4-table Design Doc schema (novel
    -- mechanism #3: mediated inter-agent memory reads with hallucination
    -- interception). Optional, caller-supplied justification for the
    -- written value -- e.g. the actual tool output it was derived from --
    -- so a *reading* agent's syscall can optionally verify the claim is
    -- actually supported before it's allowed to "commit" into that
    -- agent's own context, instead of blindly trusting shared memory.
    evidence TEXT
);

CREATE INDEX IF NOT EXISTS idx_syscall_log_agent ON syscall_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_resource_alloc_key ON resource_allocation(resource_key);

-- Extension beyond the original 4-table Design Doc schema: reputation-driven
-- adaptive admission control (novel mechanism #5). Keyed by `agent_type`
-- (a caller-declared, *reusable* label -- unlike agent_id, which is retired
-- forever once used, per DuplicateAgentError), so a recurring class of agent
-- (e.g. every worker spawned from the same prompt/template) can be tracked
-- across many individual admissions, including across kernel restarts when
-- given a real db_path instead of ":memory:".
CREATE TABLE IF NOT EXISTS agent_reputation (
    agent_type TEXT PRIMARY KEY,
    admissions INTEGER NOT NULL DEFAULT 0,
    kills INTEGER NOT NULL DEFAULT 0,
    quota_exhaustions INTEGER NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL
);
