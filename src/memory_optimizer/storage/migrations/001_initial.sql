CREATE TABLE batches (
    id uuid PRIMARY KEY,
    scope_hash text NOT NULL,
    request_hash text NOT NULL,
    user_id text NOT NULL,
    request_id text NOT NULL,
    session_id text NOT NULL,
    payload_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_hash, request_hash)
);

CREATE TABLE messages (
    batch_id uuid NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL CHECK (length(content) > 0),
    source_timestamp bigint,
    PRIMARY KEY (batch_id, ordinal)
);

CREATE TABLE chunks (
    id uuid PRIMARY KEY,
    batch_id uuid NOT NULL,
    message_ordinal integer NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    content text NOT NULL CHECK (length(content) = end_offset - start_offset),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    FOREIGN KEY (batch_id, message_ordinal) REFERENCES messages(batch_id, ordinal)
        ON DELETE CASCADE,
    UNIQUE (batch_id, message_ordinal, ordinal)
);

CREATE INDEX chunks_search_idx ON chunks USING gin(search_vector);
