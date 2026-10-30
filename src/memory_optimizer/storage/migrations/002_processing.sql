CREATE TABLE processing_runs (
    id uuid PRIMARY KEY,
    batch_id uuid NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    revision integer NOT NULL CHECK (revision > 0),
    spec jsonb NOT NULL,
    stage text NOT NULL CHECK (stage IN ('raw_ready', 'selected', 'complete')),
    outcome text CHECK (outcome IN ('raw_legacy', 'raw_not_selected', 'raw_degraded')),
    reason text CHECK (reason = 'processor_unavailable'),
    candidate_count integer CHECK (candidate_count >= 0),
    selected_count integer CHECK (selected_count >= 0 AND selected_count <= candidate_count),
    model_calls integer NOT NULL DEFAULT 0 CHECK (model_calls = 0),
    owner uuid,
    fence bigint NOT NULL DEFAULT 0 CHECK (fence >= 0),
    lease_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    selected_at timestamptz,
    finished_at timestamptz,
    UNIQUE (batch_id, revision),
    UNIQUE (id, batch_id),
    CHECK ((owner IS NULL) = (lease_until IS NULL)),
    CHECK ((
        (stage = 'raw_ready' AND candidate_count IS NULL AND selected_count IS NULL
            AND selected_at IS NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
        OR (stage = 'selected' AND candidate_count IS NOT NULL AND selected_count IS NOT NULL
            AND selected_at IS NOT NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
        OR (stage = 'complete' AND finished_at IS NOT NULL AND owner IS NULL
            AND ((outcome = 'raw_legacy' AND reason IS NULL AND candidate_count IS NULL
                    AND selected_count IS NULL AND selected_at IS NULL)
                OR (outcome = 'raw_not_selected' AND reason IS NULL AND candidate_count = 0
                    AND selected_count = 0 AND selected_at IS NOT NULL)
                OR (outcome = 'raw_degraded' AND reason = 'processor_unavailable'
                    AND candidate_count > 0 AND selected_count > 0 AND selected_at IS NOT NULL)))
    ) IS TRUE)
);

CREATE TABLE processing_windows (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL,
    batch_id uuid NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    message_ordinal integer NOT NULL,
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    context_spans jsonb NOT NULL CHECK (jsonb_typeof(context_spans) = 'array'),
    reasons jsonb NOT NULL CHECK (jsonb_typeof(reasons) = 'array'),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'deferred')),
    FOREIGN KEY (run_id, batch_id) REFERENCES processing_runs(id, batch_id) ON DELETE CASCADE,
    FOREIGN KEY (batch_id, message_ordinal) REFERENCES messages(batch_id, ordinal) ON DELETE CASCADE,
    UNIQUE (run_id, ordinal),
    UNIQUE (run_id, message_ordinal, start_offset, end_offset)
);

-- Legacy batches already returned successful receipts. Do not silently reprocess them.
INSERT INTO processing_runs (id, batch_id, revision, spec, stage, outcome, finished_at)
SELECT id, id, 1, '{"flow_version":"legacy-raw-v1"}'::jsonb, 'complete', 'raw_legacy', clock_timestamp()
FROM batches;
