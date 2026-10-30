-- Existing v1 branches retain their constraints. Only the explicit new flow
-- may terminate before candidate binding or with zero selected windows and gaps.
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_reason_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_reason_check CHECK (
    reason IS NULL OR reason IN ('processor_unavailable','offline_only','extraction_failed')
    OR ((spec->>'flow_version' = 'coverage-offline-v1'
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
            OR (spec->>'flow_version' = 'coverage-offline-v1' AND outcome = 'raw_degraded'
                AND reason IN ('time_budget_exhausted','planning_attempts_exhausted','selection_budget')
                AND ((candidate_count IS NULL AND selected_count IS NULL AND selected_at IS NULL)
                    OR (candidate_count IS NOT NULL AND selected_count IS NOT NULL AND selected_at IS NOT NULL)))))
) IS TRUE);

CREATE TABLE coverage_bindings (
    run_id uuid PRIMARY KEY,
    batch_id uuid NOT NULL,
    planning_id uuid NOT NULL UNIQUE REFERENCES planning_runs(id) ON DELETE CASCADE,
    selection_hash text NOT NULL CHECK (selection_hash ~ '^[0-9a-f]{64}$'),
    adapter_version text NOT NULL CHECK (adapter_version = 'final-windows-v1'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (run_id,batch_id) REFERENCES processing_runs(id,batch_id) ON DELETE CASCADE
);
CREATE FUNCTION protect_coverage_binding() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'immutable coverage binding'; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_coverage_binding BEFORE UPDATE ON coverage_bindings
    FOR EACH ROW EXECUTE FUNCTION protect_coverage_binding();

CREATE FUNCTION protect_coverage_runtime() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.spec->>'flow_version' = 'coverage-offline-v1' AND (
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
CREATE TRIGGER protect_coverage_runtime BEFORE UPDATE ON processing_runs
    FOR EACH ROW EXECUTE FUNCTION protect_coverage_runtime();
