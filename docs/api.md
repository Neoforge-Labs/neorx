# API reference

Generated from the docstrings in the source. The top-level `neorx`
package re-exports the pipeline entry points, so
`from neorx import run_pipeline` and
`from neorx.core.pipeline import run_pipeline` are equivalent.

```python
>>> import neorx
>>> "run_pipeline" in neorx.__all__ and "identify_causal_targets" in neorx.__all__
True

```

## neorx.core.causal.graph_semantics

Which edges may carry a causal claim. See
[Which edges earn an arrow](admissibility.md).

::: neorx.core.causal.graph_semantics
    options:
      show_root_heading: false
      members:
        - is_causal_admissible
        - causal_subgraph
        - cyclic_components
        - acyclic_core
        - AcyclicCore

### Vocabularies

`CAUSAL_EDGE_TYPES`
:   Frozenset of `edge_type` values that assert a directed mechanism.

`GENETIC_EVIDENCE_CLASSES`
:   Frozenset of `evidence_class` values that license a gene-to-disease
    arrow.

## neorx.core.causal.backdoor

Identification on the admissible subgraph. See
[Identification](identification.md).

::: neorx.core.causal.backdoor
    options:
      show_root_heading: false
      members:
        - find_adjustment_set
        - Identification
        - IdentificationReason
        - satisfies_backdoor
        - mutilated_graph
        - confounding_sensitivity

### Search bounds

`MAX_ADJUSTMENT_SET_SIZE`
:   Largest adjustment set the search will consider. Default `3`.

`MAX_CANDIDATE_POOL`
:   Largest candidate pool the search will draw from. Default `20`.

Reaching either bound sets `Identification.search_truncated`.

## neorx.core.causal.identifier

Candidate evaluation and ranking. See
[From verdict to target](targets.md).

::: neorx.core.causal.identifier
    options:
      show_root_heading: false
      members:
        - identify_causal_targets
        - evaluate_all_targets

::: neorx.core.causal.ranking
    options:
      show_root_heading: false
      members:
        - rank_causal_targets

::: neorx.core.causal.scoring
    options:
      show_root_heading: false
      members:
        - compute_causal_confidence
        - classify_target
        - assess_druggability
        - organism_disease_relevance
        - evaluate_pathogen_target

::: neorx.core.causal.evidence
    options:
      show_root_heading: false
      members:
        - collect_source_scores
        - count_evidence_streams
        - count_pathway_connections
        - count_protein_interactions
        - compute_path_strength
        - corroboration_factor

## neorx.core.graph

Graph assembly and persistence. See
[The graph and its sources](graph.md).

::: neorx.core.graph.graph_builder
    options:
      show_root_heading: false
      members:
        - build_disease_graph
        - disease_graph_to_networkx

::: neorx.core.graph.persistence
    options:
      show_root_heading: false
      members:
        - save_graph
        - load_graph
        - list_saved_graphs

## neorx.core.graph.models

The data model shared by every stage.

::: neorx.core.graph.models
    options:
      show_root_heading: false
      members:
        - NodeType
        - EdgeType
        - TargetClassification
        - JobStatus
        - GraphNode
        - GraphEdge
        - DiseaseGraph
        - NeoRxResult
        - ScoredCandidate
        - PipelineJob
        - PipelineResult

## neorx.core.pipeline

::: neorx.core.pipeline
    options:
      show_root_heading: false
      members:
        - run_pipeline
        - run_rl_pipeline

## neorx.core.scoring

Candidate ranking. See [Ranking candidates](scoring.md).

::: neorx.core.scoring.scorer
    options:
      show_root_heading: false
      members:
        - score_candidate
        - rank_candidates
        - normalise_affinity
        - normalise_sa

::: neorx.core.scoring.admet
    options:
      show_root_heading: false
      members:
        - predict_admet
        - ADMETProfile

## neorx.core.bio

The biological intelligence layer.

::: neorx.core.bio.classifier
    options:
      show_root_heading: false
      members:
        - TargetType
        - DiseaseType
        - classify_disease
        - TargetClassifier

## neorx.core.sources

One function per source. Each returns
`tuple[list[GraphNode], list[GraphEdge]]` and returns empty lists rather
than raising, so the graph builder can aggregate what answered.

::: neorx.core.sources
    options:
      show_root_heading: false
      members: []

`query_monarch`, `query_open_targets`, `query_kegg_pathways`,
`query_reactome_pathways`, `query_string_interactions`,
`query_omnipath`, `query_uniprot`, `query_pdb_structures`,
`query_chembl`.

## neorx.snapshots

The versioned data layer. See [Versioned snapshots](snapshots.md).

::: neorx.snapshots.schema
    options:
      show_root_heading: false
      members:
        - ExtractNotes
        - empty_associations
        - empty_interactions

::: neorx.snapshots.manifest
    options:
      show_root_heading: false
      members:
        - SnapshotEntry
        - read_manifest
        - write_entry
        - digest_file

::: neorx.snapshots.reader
    options:
      show_root_heading: false
      members:
        - SnapshotStore
        - SnapshotMissingError

::: neorx.snapshots.resolver
    options:
      show_root_heading: false
      members:
        - SourceResolver
        - UnpinnedSourceError

::: neorx.snapshots.releases
    options:
      show_root_heading: false
      members:
        - release_for
        - omnipath_archive_url
        - UnknownTimePointError

## neorx.experiments

The run-record system. See
[Experiments and run records](experiments.md).

::: neorx.experiments.record
    options:
      show_root_heading: false
      members:
        - RunRecord
        - ProvenanceError
        - RecordTooLargeError
        - RunIDCollisionError

::: neorx.experiments.registry
    options:
      show_root_heading: false
      members:
        - experiment
        - run_experiment
        - get_experiment
        - list_experiments
        - ExperimentDef
        - settle_or_refuse
        - UnknownExperimentError

::: neorx.experiments.replay
    options:
      show_root_heading: false
      members:
        - replay_experiment
        - prune_record
        - ReplayResult
        - RowDiff
        - NotReplayableError

::: neorx.experiments.gates
    options:
      show_root_heading: false
      members:
        - Finding
        - find_hardcoded_metrics
        - check_records_wellformed
        - check_record_integrity
        - check_committed_records_are_citable
        - check_cited_runs
        - check_no_fabricated_statistics

## neorx.genmol

Molecule generation. See [GenMol](genmol.md).

::: neorx.genmol.pretrained
    options:
      show_root_heading: false
      members:
        - load_pretrained
        - GenMolAssetError

::: neorx.genmol.generate
    options:
      show_root_heading: false
      members:
        - generate
        - generate_conditional
        - interpolate

::: neorx.genmol.models.vae
    options:
      show_root_heading: false
      members:
        - MolVAE
        - vae_loss

::: neorx.genmol.train
    options:
      show_root_heading: false
      members:
        - TrainConfig
        - train_vae
        - load_checkpoint

## neorx.molscreen

Drug-likeness screening. See [MolScreen](molscreen.md).

::: neorx.molscreen.parser
    options:
      show_root_heading: false
      members:
        - parse_smiles
        - validate_smiles
        - canonicalise
        - name_to_smiles
        - smart_parse

::: neorx.molscreen.properties
    options:
      show_root_heading: false
      members:
        - calculate_properties

::: neorx.molscreen.accessibility
    options:
      show_root_heading: false
      members:
        - qed_score
        - sa_score

::: neorx.molscreen.filters
    options:
      show_root_heading: false
      members:
        - lipinski_filter
        - veber_filter
        - ghose_filter
        - egan_filter
        - pains_filter
        - brenk_filter
        - run_all_filters
        - classify_drug_likeness

::: neorx.molscreen.similarity
    options:
      show_root_heading: false
      members:
        - tanimoto_similarity
        - find_similar_drugs
        - build_cache
        - clear_cache

::: neorx.molscreen.models
    options:
      show_root_heading: false
      members:
        - MolecularProperties
        - FilterResult
        - ScreeningReport
        - DrugLikelihoodCategory

## neorx.dockbot

Docking. See [DockBot](dockbot.md).

::: neorx.dockbot.protein_prep
    options:
      show_root_heading: false
      members:
        - prepare_protein

::: neorx.dockbot.ligand_prep
    options:
      show_root_heading: false
      members:
        - prepare_ligand
        - prepare_ligand_pdbqt

::: neorx.dockbot.docker
    options:
      show_root_heading: false
      members:
        - dock

::: neorx.dockbot.models
    options:
      show_root_heading: false
      members:
        - ProteinInfo
        - BindingSite
        - BindingSiteMethod
        - DockingPose
        - DockingResult

## neorx.causalbiorl

Reinforcement learning. See [CausalBioRL](causalbiorl.md).

::: neorx.causalbiorl.causal.scm
    options:
      show_root_heading: false
      members:
        - StructuralCausalModel

::: neorx.causalbiorl.causal.graph_encoder
    options:
      show_root_heading: false
      members:
        - DiseaseGraphEncoder
        - RGCNLayer

::: neorx.causalbiorl.causal.surrogate_docker
    options:
      show_root_heading: false
      members:
        - SurrogateDockingModel
        - smiles_to_fingerprint

::: neorx.causalbiorl.causal.reward_learner
    options:
      show_root_heading: false
      members:
        - AdaptiveRewardLearner

::: neorx.causalbiorl.envs.drug_discovery
    options:
      show_root_heading: false
      members:
        - DrugDiscoveryEnv

::: neorx.causalbiorl.agents.causal_agent
    options:
      show_root_heading: false
      members:
        - CausalAgent

## neorx.mirrorfold

Mirror-image proteins. See [MirrorFold](mirrorfold.md).

::: neorx.mirrorfold.mirror
    options:
      show_root_heading: false
      members:
        - mirror_sequence
        - is_valid_sequence
        - reflect_coordinates
        - get_l_smiles
        - get_d_smiles
        - verify_mirror_smiles

::: neorx.mirrorfold.predictor
    options:
      show_root_heading: false
      members:
        - predict_structure
        - predict_pair

::: neorx.mirrorfold.compare
    options:
      show_root_heading: false
      members:
        - compare_structures
        - superimpose
        - calculate_tm_score

::: neorx.mirrorfold.therapeutic
    options:
      show_root_heading: false
      members:
        - assess_therapeutic_potential
        - assess_protease_resistance
        - assess_immunogenicity
