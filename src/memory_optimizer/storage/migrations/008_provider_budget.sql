-- Paid validation ledger. No changes to the seven existing migration files.
CREATE TABLE llm_campaigns (
    id text PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 100),
    max_calls integer NOT NULL CHECK (max_calls BETWEEN 1 AND 200),
    max_nusd bigint NOT NULL CHECK (max_nusd BETWEEN 1 AND 10000000000),
    model text NOT NULL CHECK (model='gpt-4o-mini-2024-07-18'),
    input_nusd integer NOT NULL CHECK (input_nusd=150),
    output_nusd integer NOT NULL CHECK (output_nusd=600),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE llm_calls (
    id uuid PRIMARY KEY,
    campaign_id text NOT NULL REFERENCES llm_campaigns(id),
    scope_hash text NOT NULL CHECK (length(scope_hash)=64),
    cache_key text NOT NULL CHECK (length(cache_key)=64),
    state text NOT NULL CHECK (state IN ('reserved','complete','error','unknown')),
    reserved_nusd bigint NOT NULL CHECK (reserved_nusd>0),
    charged_nusd bigint NOT NULL CHECK (charged_nusd>=0 AND charged_nusd<=reserved_nusd),
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    result jsonb,
    result_hash text,
    CHECK ((state='reserved' AND result IS NULL AND result_hash IS NULL)
        OR (state IN ('complete','error') AND result IS NOT NULL AND length(result_hash)=64)
        OR (state='unknown' AND result IS NULL AND result_hash IS NULL))
);
CREATE UNIQUE INDEX llm_single_flight ON llm_calls(campaign_id,scope_hash,cache_key)
    WHERE state='reserved';
CREATE INDEX llm_cache_lookup ON llm_calls(campaign_id,scope_hash,cache_key)
    WHERE state='complete';
CREATE FUNCTION protect_llm_campaign() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'provider campaign limits are immutable'; END $$;
CREATE TRIGGER llm_campaign_immutable BEFORE UPDATE OR DELETE ON llm_campaigns
    FOR EACH ROW EXECUTE FUNCTION protect_llm_campaign();
CREATE FUNCTION protect_llm_call() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='DELETE' OR OLD.state<>'reserved' THEN
        RAISE EXCEPTION 'provider accounting is append-only after settlement';
    END IF;
    IF ROW(NEW.id,NEW.campaign_id,NEW.scope_hash,NEW.cache_key,NEW.reserved_nusd,
           NEW.expires_at,NEW.created_at) IS DISTINCT FROM
       ROW(OLD.id,OLD.campaign_id,OLD.scope_hash,OLD.cache_key,OLD.reserved_nusd,
           OLD.expires_at,OLD.created_at) OR NEW.state='reserved' THEN
        RAISE EXCEPTION 'provider reservation identity is immutable';
    END IF;
    IF NEW.state='unknown' AND NEW.charged_nusd<>OLD.reserved_nusd THEN
        RAISE EXCEPTION 'unknown call retains worst-case charge';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER llm_call_immutable BEFORE UPDATE OR DELETE ON llm_calls
    FOR EACH ROW EXECUTE FUNCTION protect_llm_call();
