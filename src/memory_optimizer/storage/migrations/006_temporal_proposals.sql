-- Existing v1 branches retain their constraints. Only the explicit new flow
-- may terminate before candidate binding or with zero selected windows and gaps.
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_reason_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_reason_check CHECK (
    reason IS NULL OR reason IN ('processor_unavailable','offline_only','extraction_failed')
    OR ((spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1')
        AND reason IN ('time_budget_exhausted','planning_attempts_exhausted','selection_budget')) IS TRUE)
);
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_terminal_check;
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
                AND candidate_count > 0 AND selected_count > 0 AND selected_at IS NOT NULL)
            OR (spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1') AND outcome = 'raw_degraded'
                AND reason IN ('time_budget_exhausted','planning_attempts_exhausted','selection_budget')
                AND ((candidate_count IS NULL AND selected_count IS NULL AND selected_at IS NULL)
                    OR (candidate_count IS NOT NULL AND selected_count IS NOT NULL AND selected_at IS NOT NULL)))))
) IS TRUE);

CREATE OR REPLACE FUNCTION protect_coverage_runtime() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1') AND (
       ROW(NEW.id,NEW.batch_id,NEW.revision,NEW.spec,NEW.created_at)
           IS DISTINCT FROM ROW(OLD.id,OLD.batch_id,OLD.revision,OLD.spec,OLD.created_at)
       OR (OLD.stage = 'complete' AND NEW IS DISTINCT FROM OLD)
       OR (OLD.stage = 'selected' AND ROW(NEW.candidate_count,NEW.selected_count,NEW.selected_at)
           IS DISTINCT FROM ROW(OLD.candidate_count,OLD.selected_count,OLD.selected_at))
       OR NEW.fence < OLD.fence
       OR (NEW.stage <> OLD.stage AND NOT
           ((OLD.stage = 'raw_ready' AND NEW.stage IN ('selected','complete'))
            OR (OLD.stage = 'selected' AND NEW.stage = 'complete')))
    ) THEN RAISE EXCEPTION 'immutable coverage runtime'; END IF;
    RETURN NEW;
END $$;

ALTER TABLE candidate_windows ADD CONSTRAINT candidate_windows_temporal_link_key UNIQUE(candidate_id,window_id,run_id,batch_id);
CREATE TABLE candidate_temporal (
    candidate_id uuid NOT NULL,
    window_id uuid NOT NULL,
    run_id uuid NOT NULL,
    batch_id uuid NOT NULL,
    schema_version text NOT NULL CHECK (schema_version = 'time-proposals-v1'),
    input_hash text NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    body jsonb NOT NULL CHECK (jsonb_typeof(body) = 'object'),
    body_hash text NOT NULL CHECK (body_hash ~ '^[0-9a-f]{64}$'),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(candidate_id,window_id),
    FOREIGN KEY(candidate_id,window_id,run_id,batch_id)
        REFERENCES candidate_windows(candidate_id,window_id,run_id,batch_id) ON DELETE CASCADE
);
CREATE FUNCTION guard_candidate_temporal() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'immutable temporal record'; END IF;
    ELSE
        IF NOT EXISTS (SELECT 1 FROM processing_runs r JOIN extraction_attempts a ON a.run_id=r.id
            WHERE r.id=NEW.run_id AND r.spec->>'flow_version'='coverage-temporal-v1'
            AND r.stage='selected' AND r.lease_until>clock_timestamp()
            AND a.window_id=NEW.window_id AND a.status='accepted' AND a.input_hash=NEW.input_hash)
        THEN RAISE EXCEPTION 'temporal record requires matching live new-version candidate'; END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER guard_candidate_temporal BEFORE INSERT OR UPDATE ON candidate_temporal
    FOR EACH ROW EXECUTE FUNCTION guard_candidate_temporal();
