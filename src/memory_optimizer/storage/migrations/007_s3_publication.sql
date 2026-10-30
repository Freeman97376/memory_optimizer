-- Existing v1 branches retain their constraints. Only the explicit new flow
-- may terminate before candidate binding or with zero selected windows and gaps.
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_reason_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_reason_check CHECK (
    reason IS NULL OR reason IN ('processor_unavailable','offline_only','extraction_failed')
    OR ((spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1','s3-offline-v1')
        AND reason IN ('time_budget_exhausted','planning_attempts_exhausted','selection_budget')) IS TRUE)
);
ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_terminal_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_terminal_check CHECK ((
    (stage = 'raw_ready' AND candidate_count IS NULL AND selected_count IS NULL
        AND selected_at IS NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
    OR (stage = 'selected' AND candidate_count IS NOT NULL AND selected_count IS NOT NULL
        AND selected_at IS NOT NULL AND outcome IS NULL AND reason IS NULL AND finished_at IS NULL)
    OR (stage = 'complete' AND finished_at IS NOT NULL AND owner IS NULL
        AND ((spec->>'flow_version'='s3-offline-v1' AND outcome IN ('s3_published','s3_withheld') AND reason IS NULL)
            OR (outcome = 'raw_legacy' AND reason IS NULL AND candidate_count IS NULL
                AND selected_count IS NULL AND selected_at IS NULL)
            OR (outcome = 'raw_not_selected' AND reason IS NULL AND candidate_count = 0
                AND selected_count = 0 AND selected_at IS NOT NULL)
            OR (outcome = 'raw_degraded' AND reason IS NOT NULL
                AND candidate_count > 0 AND selected_count > 0 AND selected_at IS NOT NULL)
            OR (spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1','s3-offline-v1') AND outcome = 'raw_degraded'
                AND reason IN ('time_budget_exhausted','planning_attempts_exhausted','selection_budget')
                AND ((candidate_count IS NULL AND selected_count IS NULL AND selected_at IS NULL)
                    OR (candidate_count IS NOT NULL AND selected_count IS NOT NULL AND selected_at IS NOT NULL)))))
) IS TRUE);

CREATE OR REPLACE FUNCTION protect_coverage_runtime() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.spec->>'flow_version' IN ('coverage-offline-v1','coverage-temporal-v1','s3-offline-v1') AND (
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

CREATE TABLE relation_snapshots (
    run_id uuid NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    revision integer NOT NULL CHECK(revision BETWEEN 1 AND 2),
    body jsonb NOT NULL,
    input_hash text NOT NULL CHECK(input_hash ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(run_id,revision)
);
CREATE TABLE relation_attempts (
    run_id uuid NOT NULL,
    revision integer NOT NULL,
    number integer NOT NULL CHECK(number BETWEEN 1 AND 2),
    fence bigint NOT NULL,
    status text NOT NULL CHECK(status IN ('reserved','accepted','failed')),
    input_hash text NOT NULL,
    reply text,
    decision jsonb,
    decision_hash text,
    error_code text,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY(run_id,revision,number),
    FOREIGN KEY(run_id,revision) REFERENCES relation_snapshots(run_id,revision) ON DELETE CASCADE,
    CHECK((status='reserved' AND reply IS NULL AND decision IS NULL AND finished_at IS NULL)
       OR (status='accepted' AND reply IS NOT NULL AND decision IS NOT NULL AND decision_hash IS NOT NULL AND finished_at IS NOT NULL AND error_code IS NULL)
       OR (status='failed' AND decision IS NULL AND error_code IS NOT NULL AND finished_at IS NOT NULL))
);
CREATE TABLE episode_versions (
    id uuid PRIMARY KEY,
    episode_id uuid NOT NULL,
    user_id text NOT NULL,
    revision integer NOT NULL CHECK(revision>0),
    created_by_run uuid NOT NULL REFERENCES processing_runs(id) ON DELETE CASCADE,
    body jsonb NOT NULL,
    body_hash text NOT NULL,
    content text NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple',content)) STORED,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(episode_id,revision),
    UNIQUE(id,episode_id,user_id)
);
CREATE INDEX episode_search_idx ON episode_versions USING gin(search_vector);
CREATE TABLE episode_heads (
    episode_id uuid PRIMARY KEY,
    user_id text NOT NULL,
    version_id uuid NOT NULL,
    FOREIGN KEY(version_id,episode_id,user_id) REFERENCES episode_versions(id,episode_id,user_id)
);
CREATE INDEX episode_scope_idx ON episode_heads(user_id);
CREATE TABLE s3_publications (
    run_id uuid PRIMARY KEY REFERENCES processing_runs(id) ON DELETE CASCADE,
    body jsonb NOT NULL,
    body_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TRIGGER immutable_relation_snapshot BEFORE UPDATE ON relation_snapshots
    FOR EACH ROW EXECUTE FUNCTION protect_coverage_binding();
CREATE TRIGGER immutable_episode_version BEFORE UPDATE ON episode_versions
    FOR EACH ROW EXECUTE FUNCTION protect_coverage_binding();
CREATE TRIGGER immutable_s3_publication BEFORE UPDATE ON s3_publications
    FOR EACH ROW EXECUTE FUNCTION protect_coverage_binding();

CREATE FUNCTION s3_attempt_budget() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE runtime processing_runs; total bigint;
BEGIN
    IF TG_TABLE_NAME='planning_attempts' THEN
        SELECT r.* INTO runtime FROM processing_runs r JOIN planning_runs p ON p.batch_id=r.batch_id
        WHERE p.id=NEW.run_id AND r.revision=1 FOR UPDATE OF r;
    ELSE
        SELECT * INTO runtime FROM processing_runs WHERE id=NEW.run_id FOR UPDATE;
    END IF;
    IF runtime.spec->>'flow_version'='s3-offline-v1' THEN
        SELECT (SELECT count(*) FROM planning_attempts a JOIN planning_runs p ON p.id=a.run_id WHERE p.batch_id=runtime.batch_id)
             + (SELECT count(*) FROM extraction_attempts WHERE run_id=runtime.id)
             + (SELECT count(*) FROM relation_attempts WHERE run_id=runtime.id) INTO total;
        IF total >= (runtime.spec->'budget'->>'max_attempts')::integer THEN
            RAISE EXCEPTION 'S3 shared attempt budget exhausted';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER s3_planning_budget BEFORE INSERT ON planning_attempts FOR EACH ROW EXECUTE FUNCTION s3_attempt_budget();
CREATE TRIGGER s3_extraction_budget BEFORE INSERT ON extraction_attempts FOR EACH ROW EXECUTE FUNCTION s3_attempt_budget();
CREATE TRIGGER s3_relation_budget BEFORE INSERT ON relation_attempts FOR EACH ROW EXECUTE FUNCTION s3_attempt_budget();

CREATE FUNCTION guard_relation_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='UPDATE' AND (OLD.status<>'reserved' OR
        ROW(NEW.run_id,NEW.revision,NEW.number,NEW.input_hash,NEW.fence,NEW.started_at)
        IS DISTINCT FROM ROW(OLD.run_id,OLD.revision,OLD.number,OLD.input_hash,OLD.fence,OLD.started_at))
    THEN RAISE EXCEPTION 'immutable relation attempt'; END IF;
    IF NOT EXISTS(SELECT 1 FROM processing_runs r JOIN relation_snapshots s ON s.run_id=r.id
        WHERE r.id=NEW.run_id AND r.spec->>'flow_version'='s3-offline-v1'
        AND r.stage='selected' AND r.lease_until>clock_timestamp()
        AND s.revision=NEW.revision AND s.input_hash=NEW.input_hash)
    THEN RAISE EXCEPTION 'relation attempt requires live matching runtime'; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER guard_relation_attempt BEFORE INSERT OR UPDATE ON relation_attempts
    FOR EACH ROW EXECUTE FUNCTION guard_relation_attempt();


ALTER TABLE processing_runs DROP CONSTRAINT processing_runs_outcome_check;
ALTER TABLE processing_runs ADD CONSTRAINT processing_runs_outcome_check CHECK(outcome IN ('raw_legacy','raw_not_selected','raw_degraded','s3_published','s3_withheld'));

CREATE OR REPLACE FUNCTION guard_candidate_temporal() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'immutable temporal record'; END IF;
    ELSE
        IF NOT EXISTS (SELECT 1 FROM processing_runs r JOIN extraction_attempts a ON a.run_id=r.id
            WHERE r.id=NEW.run_id AND r.spec->>'flow_version' IN ('coverage-temporal-v1','s3-offline-v1')
            AND r.stage='selected' AND r.lease_until>clock_timestamp()
            AND a.window_id=NEW.window_id AND a.status='accepted' AND a.input_hash=NEW.input_hash)
        THEN RAISE EXCEPTION 'temporal record requires matching live new-version candidate'; END IF;
    END IF;
    RETURN NEW;
END $$;
