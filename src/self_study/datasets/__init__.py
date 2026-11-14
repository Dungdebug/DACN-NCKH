"""
Datasets and pattern recognition modules for Self-Study Knowledge Bank (Wave 5 & Wave 5.5).
Pure-Python, strict offline.
"""

from src.self_study.datasets.cryptography_and_security_patterns import (
    expand_domain_query_terms,
    CRYPTO_SYNONYM_CLUSTERS,
    CRYPTO_FEW_SHOT_QA_EXAMPLES,
)
from src.self_study.datasets.definition_and_taxonomy_patterns import (
    extract_taxonomy_and_types,
    TAXONOMY_INSTRUCTION_DATASET,
)
from src.self_study.datasets.formula_and_example_patterns import (
    extract_examples_and_formulas,
    FORMULA_EXAMPLE_DATASET,
    is_formula_sentence,
    is_example_sentence,
    is_definition_sentence,
)
from src.self_study.datasets.question_paraphrase_normalizer import (
    normalize_student_question,
    QuestionParaphraseNormalizer,
    PARAPHRASE_DATASET,
)
from src.self_study.datasets.autonomous_qa_generator import (
    SelfCurriculumGenerator,
    CurriculumLevel,
)
from src.self_study.datasets.extractive_text_stitcher import (
    ExtractiveTextStitcher,
)

# Wave 5.5 Multi-Domain Academic Datasets
from src.self_study.datasets.networking_and_protocols_patterns import (
    get_networking_synonyms,
    expand_networking_query_terms,
    NETWORKING_FEW_SHOT_EXAMPLES,
)
from src.self_study.datasets.database_and_sql_patterns import (
    get_database_synonyms,
    expand_database_query_terms,
    is_sql_or_relational_query,
    DATABASE_FEW_SHOT_EXAMPLES,
)
from src.self_study.datasets.operating_systems_patterns import (
    get_os_synonyms,
    expand_os_query_terms,
    OS_FEW_SHOT_EXAMPLES,
)
from src.self_study.datasets.ai_and_machine_learning_patterns import (
    get_ai_ml_synonyms,
    expand_ai_ml_query_terms,
    AI_ML_FEW_SHOT_EXAMPLES,
)
from src.self_study.datasets.data_structures_algorithms_patterns import (
    get_dsa_synonyms,
    expand_dsa_query_terms,
    DSA_FEW_SHOT_EXAMPLES,
    DSA_COMPLEXITY_REFERENCE,
)
from src.self_study.datasets.software_engineering_patterns import (
    get_se_synonyms,
    expand_se_query,
    SE_FEW_SHOT_EXAMPLES,
)
from src.self_study.datasets.vietnamese_academic_formatting import (
    parse_vietnamese_academic_blocks,
    extract_definitions,
    extract_theorems_and_properties,
    extract_examples,
    extract_academic_hierarchy,
    search_academic_blocks,
    format_block_as_markdown,
    generate_qa_seeds_from_blocks,
    ACADEMIC_BLOCK_BENCHMARK,
)
from src.self_study.datasets.comparison_and_pros_cons_stitcher import (
    synthesize_pros_cons_and_comparison,
)
from src.self_study.datasets.step_by_step_process_stitcher import (
    synthesize_step_by_step_process,
)
from src.self_study.datasets.domain_router_registry import (
    AcademicDomain,
    AcademicDomainRegistry,
    expand_all_academic_synonyms,
)

__all__ = [
    # Wave 5 Core
    "expand_domain_query_terms",
    "CRYPTO_SYNONYM_CLUSTERS",
    "CRYPTO_FEW_SHOT_QA_EXAMPLES",
    "extract_taxonomy_and_types",
    "TAXONOMY_INSTRUCTION_DATASET",
    "extract_examples_and_formulas",
    "FORMULA_EXAMPLE_DATASET",
    "is_formula_sentence",
    "is_example_sentence",
    "is_definition_sentence",
    "normalize_student_question",
    "QuestionParaphraseNormalizer",
    "PARAPHRASE_DATASET",
    "SelfCurriculumGenerator",
    "CurriculumLevel",
    "ExtractiveTextStitcher",
    # Wave 5.5 Multi-Domain
    "get_networking_synonyms",
    "expand_networking_query_terms",
    "NETWORKING_FEW_SHOT_EXAMPLES",
    "get_database_synonyms",
    "expand_database_query_terms",
    "is_sql_or_relational_query",
    "DATABASE_FEW_SHOT_EXAMPLES",
    "get_os_synonyms",
    "expand_os_query_terms",
    "OS_FEW_SHOT_EXAMPLES",
    "get_ai_ml_synonyms",
    "expand_ai_ml_query_terms",
    "AI_ML_FEW_SHOT_EXAMPLES",
    "get_dsa_synonyms",
    "expand_dsa_query_terms",
    "DSA_FEW_SHOT_EXAMPLES",
    "DSA_COMPLEXITY_REFERENCE",
    "get_se_synonyms",
    "expand_se_query",
    "SE_FEW_SHOT_EXAMPLES",
    "parse_vietnamese_academic_blocks",
    "extract_definitions",
    "extract_theorems_and_properties",
    "extract_examples",
    "extract_academic_hierarchy",
    "search_academic_blocks",
    "format_block_as_markdown",
    "generate_qa_seeds_from_blocks",
    "ACADEMIC_BLOCK_BENCHMARK",
    "synthesize_pros_cons_and_comparison",
    "synthesize_step_by_step_process",
    "AcademicDomain",
    "AcademicDomainRegistry",
    "expand_all_academic_synonyms",
]
