"""The only service assembly point that chooses concrete adapters and strategies."""

from functools import partial

from memory_optimizer.application import AddService, SearchService, Services
from memory_optimizer.chunking import chunks
from memory_optimizer.config import Settings
from memory_optimizer.extraction.types import OfflineExtractor
from memory_optimizer.extraction.workflow import OfflineCandidateFlow
from memory_optimizer.processing.types import ProcessingSpec
from memory_optimizer.processing.workflow import ProcessingFlow
from memory_optimizer.storage.candidates import PostgresCandidateRepository
from memory_optimizer.storage.postgres import PostgresStore
from memory_optimizer.storage.processing import PostgresProcessingRepository


def build_services(
    settings: Settings, *, offline_extractor: OfflineExtractor | None = None
) -> Services:
    store = PostgresStore(settings.database)
    flow = ProcessingFlow(PostgresProcessingRepository(store.transaction))
    if offline_extractor is not None:
        from dataclasses import replace

        flow = replace(
            flow,
            spec=ProcessingSpec(
                flow_version="offline-candidates-v1",
                processor_version=offline_extractor.version,
            ),
            processors={
                "offline-candidates-v1": OfflineCandidateFlow(
                    PostgresCandidateRepository(store.transaction),
                    offline_extractor,
                )
            },
        )
    return Services(
        add=AddService(
            repository=store,
            slicer=partial(chunks, size=settings.chunk_chars, overlap=settings.chunk_overlap),
            processing=flow,
        ),
        search=SearchService(index=store),
        lifecycle=store,
    )


def build_coverage_services(settings, *, spec, offline_extractor):
    """Explicit internal entry. HTTP and the existing offline CLI keep v1."""
    from memory_optimizer.processing.coverage_workflow import CoverageFlow
    from memory_optimizer.storage.coverage import (
        CoverageIngestRepository,
        PostgresCoverageRepository,
    )

    store = PostgresStore(settings.database)
    if spec.flow_version == "coverage-temporal-v1":
        from memory_optimizer.storage.temporal import PostgresTemporalRepository
        from memory_optimizer.temporal.workflow import TemporalCoverageFlow

        flow = TemporalCoverageFlow(
            PostgresTemporalRepository(store.transaction), spec, offline_extractor
        )
    else:
        flow = CoverageFlow(PostgresCoverageRepository(store.transaction), spec, offline_extractor)
    return Services(
        add=AddService(
            CoverageIngestRepository(store),
            partial(chunks, size=settings.chunk_chars, overlap=settings.chunk_overlap),
            flow,
        ),
        search=SearchService(store),
        lifecycle=store,
    )


def build_s3_services(settings, *, spec, offline_extractor, relation_proposer):
    """Explicit offline S3 assembly; never selected by default HTTP or v1 CLI."""
    from memory_optimizer.domain import ProcessingUnavailable
    from memory_optimizer.relations.replay import JointExtractor, processor_version
    from memory_optimizer.relations.workflow import S3Flow
    from memory_optimizer.storage.coverage import CoverageIngestRepository
    from memory_optimizer.storage.episode_search import EpisodeSearchIndex
    from memory_optimizer.storage.relations import PostgresRelationsRepository

    version = processor_version(offline_extractor, relation_proposer)
    if spec.processor_version != version:
        raise ProcessingUnavailable("S3 requires both frozen offline response versions")
    store = PostgresStore(settings.database)
    flow = S3Flow(
        PostgresRelationsRepository(store.transaction),
        spec,
        JointExtractor(offline_extractor, version),
        proposer=relation_proposer,
    )
    return Services(
        add=AddService(
            CoverageIngestRepository(store),
            partial(chunks, size=settings.chunk_chars, overlap=settings.chunk_overlap),
            flow,
        ),
        search=SearchService(EpisodeSearchIndex(store)),
        lifecycle=store,
    )
