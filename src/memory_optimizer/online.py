"""Explicit network-enabled S3 composition, with persisted time and spend limits."""

from dataclasses import dataclass
from datetime import timedelta

from memory_optimizer.admission import SingleAdd
from memory_optimizer.application import AddService, SearchService, Services
from memory_optimizer.chunking import chunks
from memory_optimizer.llm.adapters import OpenAIExtractor, OpenAIRelations
from memory_optimizer.llm.contracts import VERSION
from memory_optimizer.llm.openai import OpenAITransport
from memory_optimizer.llm.runtime import BudgetedModel
from memory_optimizer.processing.checkpoints import RunBudget
from memory_optimizer.processing.coverage_types import CoverageSpec
from memory_optimizer.processing.plan_types import CAPABILITY_SELECTOR, PlanLimits, PlanningSpec
from memory_optimizer.relations.workflow import S3Flow
from memory_optimizer.storage.coverage import CoverageIngestRepository
from memory_optimizer.storage.evidence_search import EvidenceSearchIndex
from memory_optimizer.storage.llm import PostgresCallLedger
from memory_optimizer.storage.postgres import PostgresStore
from memory_optimizer.storage.relations import PostgresRelationsRepository, S3Candidates

FLOW = "s3-openai-v1"


@dataclass(frozen=True)
class OpenAIS3Flow(S3Flow):
    FLOW_VERSION = FLOW
    lease_seconds: int = 120

    def work_deadline(self, lease):
        return self.repository.load_plan(lease).deadline_at - timedelta(
            seconds=lease.spec.cleanup_seconds
        )

    def extract_reply(self, lease, request, attempt):
        return self.extractor.extract(request, attempt, deadline=self.work_deadline(lease))

    def relation_reply(self, lease, request, attempt):
        return self.proposer.propose(request, attempt, deadline=self.work_deadline(lease))


class OpenAICandidates(S3Candidates):
    FLOW_VERSION = FLOW


class OpenAIRepository(PostgresRelationsRepository):
    candidate_type = OpenAICandidates


def online_spec():
    # Initial validation profile; omissions stay visible in reports and raw Search.
    return CoverageSpec(
        PlanningSpec(
            PlanLimits(
                focus_chars=3000,
                context_chars=1000,
                max_windows=4,
                max_scan_chars=120000,
                max_context_chars=24000,
                max_pool_windows=40,
            ),
            selector_version=CAPABILITY_SELECTOR,
        ),
        RunBudget(total_seconds=600, max_attempts=16),
        VERSION,
        cleanup_seconds=15,
        flow_version=FLOW,
    )


def build_online_services(settings, *, api_key, transport=None, limits=None, spec=None):
    from functools import partial

    spec = spec or online_spec()
    if spec.flow_version != FLOW or spec.processor_version != VERSION:
        raise ValueError("online S3 requires the exact frozen provider contract")
    store = PostgresStore(settings.database)
    model = BudgetedModel(
        PostgresCallLedger(store.transaction), transport or OpenAITransport(api_key), limits
    )
    flow = OpenAIS3Flow(
        OpenAIRepository(store.transaction),
        spec,
        OpenAIExtractor(model),
        proposer=OpenAIRelations(model),
    )
    return Services(
        add=SingleAdd(
            AddService(
                CoverageIngestRepository(store),
                partial(chunks, size=settings.chunk_chars, overlap=settings.chunk_overlap),
                flow,
            )
        ),
        search=SearchService(EvidenceSearchIndex(store)),
        lifecycle=store,
    )
