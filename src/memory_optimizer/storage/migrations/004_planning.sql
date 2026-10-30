-- Independent planning checkpoints; legacy stages, sources and receipts are untouched.
CREATE TABLE planning_runs (
    id uuid PRIMARY KEY,
    batch_id uuid NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    revision integer NOT NULL DEFAULT 1 CHECK (revision = 1),
    source_hash text NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    spec jsonb NOT NULL CHECK (jsonb_typeof(spec) = 'object'),
    total_seconds integer NOT NULL CHECK (total_seconds BETWEEN 1 AND 86400),
    max_attempts integer NOT NULL CHECK (max_attempts BETWEEN 1 AND 32),
    created_at timestamptz NOT NULL,
    deadline_at timestamptz NOT NULL,
    stage text NOT NULL DEFAULT 'raw_ready' CHECK (stage IN ('raw_ready','plan_ready','selection_ready')),
    plan_body jsonb CHECK (jsonb_typeof(plan_body) = 'object'),
    plan_hash text CHECK (plan_hash ~ '^[0-9a-f]{64}$'),
    planned_at timestamptz,
    selection_body jsonb CHECK (jsonb_typeof(selection_body) = 'object'),
    selected_at timestamptz,
    closed_reason text CHECK (closed_reason IN ('deadline_exhausted','attempts_exhausted')),
    owner uuid,
    fence bigint NOT NULL DEFAULT 0 CHECK (fence >= 0),
    lease_until timestamptz,
    UNIQUE (batch_id, revision),
    CHECK (deadline_at = created_at + make_interval(secs => total_seconds)),
    CHECK ((owner IS NULL) = (lease_until IS NULL)),
    CHECK ((
      (stage = 'raw_ready' AND plan_body IS NULL AND plan_hash IS NULL AND planned_at IS NULL
         AND selection_body IS NULL AND selected_at IS NULL)
      OR (stage = 'plan_ready' AND plan_body IS NOT NULL AND plan_hash IS NOT NULL AND planned_at IS NOT NULL
         AND selection_body IS NULL AND selected_at IS NULL)
      OR (stage = 'selection_ready' AND plan_body IS NOT NULL AND plan_hash IS NOT NULL AND planned_at IS NOT NULL
         AND selection_body IS NOT NULL AND selected_at IS NOT NULL AND closed_reason IS NULL)
    ) IS TRUE)
);

CREATE TABLE planning_attempts (
    run_id uuid NOT NULL REFERENCES planning_runs(id) ON DELETE CASCADE,
    number integer NOT NULL CHECK (number BETWEEN 1 AND 32),
    reservation_id uuid NOT NULL,
    phase text NOT NULL CHECK (phase IN ('planning','selection')),
    input_hash text NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    fence bigint NOT NULL CHECK (fence > 0),
    status text NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved','completed','failed','interrupted','deadline_exhausted')),
    diagnostic jsonb CHECK (jsonb_typeof(diagnostic) = 'object'),
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY (run_id, number),
    UNIQUE (run_id, reservation_id),
    CHECK ((status = 'reserved' AND diagnostic IS NULL AND finished_at IS NULL)
        OR (status <> 'reserved' AND diagnostic IS NOT NULL AND finished_at IS NOT NULL))
);
CREATE UNIQUE INDEX planning_one_reservation ON planning_attempts(run_id) WHERE status = 'reserved';

CREATE FUNCTION protect_planning_run() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.id,NEW.batch_id,NEW.revision,NEW.source_hash,NEW.spec,NEW.total_seconds,NEW.max_attempts,
           NEW.created_at,NEW.deadline_at)
       IS DISTINCT FROM ROW(OLD.id,OLD.batch_id,OLD.revision,OLD.source_hash,OLD.spec,OLD.total_seconds,
                            OLD.max_attempts,OLD.created_at,OLD.deadline_at)
       OR (OLD.plan_body IS NOT NULL AND ROW(NEW.plan_body,NEW.plan_hash,NEW.planned_at)
           IS DISTINCT FROM ROW(OLD.plan_body,OLD.plan_hash,OLD.planned_at))
       OR (OLD.selection_body IS NOT NULL AND ROW(NEW.selection_body,NEW.selected_at)
           IS DISTINCT FROM ROW(OLD.selection_body,OLD.selected_at))
       OR (OLD.closed_reason IS NOT NULL AND NEW.closed_reason IS DISTINCT FROM OLD.closed_reason)
       OR NEW.fence < OLD.fence
       OR (NEW.stage <> OLD.stage AND NOT
          ((OLD.stage = 'raw_ready' AND NEW.stage = 'plan_ready')
           OR (OLD.stage = 'plan_ready' AND NEW.stage = 'selection_ready')))
    THEN RAISE EXCEPTION 'immutable planning identity or checkpoint'; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_planning_run BEFORE UPDATE ON planning_runs
    FOR EACH ROW EXECUTE FUNCTION protect_planning_run();

CREATE FUNCTION protect_planning_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.run_id,NEW.number,NEW.reservation_id,NEW.phase,NEW.input_hash,NEW.fence,NEW.started_at)
       IS DISTINCT FROM ROW(OLD.run_id,OLD.number,OLD.reservation_id,OLD.phase,OLD.input_hash,OLD.fence,OLD.started_at)
       OR (OLD.status <> 'reserved' AND NEW IS DISTINCT FROM OLD)
    THEN RAISE EXCEPTION 'immutable planning attempt'; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_planning_attempt BEFORE UPDATE ON planning_attempts
    FOR EACH ROW EXECUTE FUNCTION protect_planning_attempt();
