ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_reason_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_reason_check
    CHECK (reason IN ('processor_unavailable', 'offline_only', 'extraction_failed'));
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_check2;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_terminal_check CHECK ((
    (stage = 'raw_ready' AND candidate_count IS NULL AND selected_count IS NULL
        AND selected_at IS NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
    OR (stage = 'selected' AND candidate_count IS NOT NULL AND selected_count IS NOT NULL
        AND selected_at IS NOT NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
    OR (stage = 'complete' AND finished_at IS NOT NULL AND owner IS NULL
        AND ((outcome = 'raw_legacy' AND reason IS NULL AND candidate_count IS NULL
                AND selected_count IS NULL AND selected_at IS NULL)
            OR (outcome = 'raw_not_selected' AND reason IS NULL AND candidate_count = 0
                AND selected_count = 0 AND selected_at IS NOT NULL)
            OR (outcome = 'raw_degraded' AND reason IS NOT NULL
                AND candidate_count > 0 AND selected_count > 0 AND selected_at IS NOT NULL)))
) IS TRUE);

ALTER TABLE processing_windows ADD UNIQUE (id, run_id, batch_id);

CREATE TABLE extraction_attempts (
    window_id uuid NOT NULL,
    run_id uuid NOT NULL,
    batch_id uuid NOT NULL,
    number integer NOT NULL CHECK (number BETWEEN 1 AND 2),
    input_hash text NOT NULL,
    fence bigint NOT NULL,
    status text NOT NULL CHECK (status IN ('reserved', 'accepted', 'empty', 'failed')),
    error_code text CHECK (error_code ~ '^[a-z_]{1,64}$'),
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY (window_id, number),
    FOREIGN KEY (window_id, run_id, batch_id)
        REFERENCES processing_windows(id, run_id, batch_id) ON DELETE CASCADE,
    CHECK ((status = 'reserved' AND error_code IS NULL AND finished_at IS NULL)
        OR (status IN ('accepted', 'empty') AND error_code IS NULL AND finished_at IS NOT NULL)
        OR (status = 'failed' AND error_code IS NOT NULL AND finished_at IS NOT NULL))
);

CREATE TABLE episode_candidates (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL,
    batch_id uuid NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = 'episode-candidates-v1'),
    fingerprint text NOT NULL,
    evidence_key text NOT NULL,
    body jsonb NOT NULL CHECK (jsonb_typeof(body) = 'object'),
    semantic_status text NOT NULL DEFAULT 'unreviewed' CHECK (semantic_status = 'unreviewed'),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (run_id, batch_id) REFERENCES processing_runs(id, batch_id) ON DELETE CASCADE,
    UNIQUE (run_id, fingerprint),
    UNIQUE (id, run_id, batch_id)
);
CREATE INDEX candidate_evidence_idx ON episode_candidates(run_id, evidence_key);

CREATE TABLE candidate_windows (
    candidate_id uuid NOT NULL,
    window_id uuid NOT NULL,
    run_id uuid NOT NULL,
    batch_id uuid NOT NULL,
    PRIMARY KEY (candidate_id, window_id),
    FOREIGN KEY (candidate_id, run_id, batch_id)
        REFERENCES episode_candidates(id, run_id, batch_id) ON DELETE CASCADE,
    FOREIGN KEY (window_id, run_id, batch_id)
        REFERENCES processing_windows(id, run_id, batch_id) ON DELETE CASCADE
);
