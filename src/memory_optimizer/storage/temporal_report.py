from memory_optimizer.storage.coverage_report import PostgresCoverageReportRepository
from memory_optimizer.temporal.proposals import FLOW_VERSION
from memory_optimizer.temporal.report import temporal_report


class PostgresTemporalReportRepository(PostgresCoverageReportRepository):
    FLOW_VERSION = FLOW_VERSION

    def read_extra(self, conn, run):
        return tuple(
            conn.execute(
                "SELECT * FROM candidate_temporal WHERE run_id=%s ORDER BY candidate_id,window_id",
                (run["id"],),
            ).fetchall()
        )

    def project(self, data, extra):
        return temporal_report(data, extra)
