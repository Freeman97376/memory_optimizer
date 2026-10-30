"""Shared SQL lease guards, used only by storage adapters."""

from memory_optimizer.domain import ProcessingBusy
from memory_optimizer.processing.types import RunLease


def lock_run(conn, lease: RunLease, stage: str):
    row = conn.execute(
        """SELECT r.* FROM processing_runs r JOIN batches b ON b.id = r.batch_id
           WHERE r.id = %s AND r.batch_id = %s AND b.user_id = %s
             AND r.owner = %s AND r.fence = %s AND r.stage = %s
             AND r.lease_until > clock_timestamp() FOR UPDATE OF r""",
        (lease.run_id, lease.batch_id, lease.user_id, lease.owner, lease.fence, stage),
    ).fetchone()
    if row is None:
        raise ProcessingBusy("processing lease lost")
    return row


def renew(conn, lease: RunLease):
    row = conn.execute(
        """UPDATE processing_runs SET lease_until = clock_timestamp() + interval '30 seconds'
           WHERE id = %s AND owner = %s AND fence = %s AND lease_until > clock_timestamp()
           RETURNING id""",
        (lease.run_id, lease.owner, lease.fence),
    ).fetchone()
    if row is None:
        raise ProcessingBusy("processing lease expired during candidate transaction")
