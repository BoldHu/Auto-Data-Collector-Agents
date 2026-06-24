"""Text cleaning prompt templates v2.0 — improved for Phase 2.7 recleaning.

Key improvements over v1.0:
- Strict JSON output schema with provenance fields
- Domain filtering: detect and handle non-carbon-fiber content
- Systematic OCR repair rules (not vague "fix OCR errors")
- Boilerplate removal: publisher names, ISBNs, copyright notices, etc.
- 6 prompt types: body, formula, table, mixed, header_footer, empty
- Separate source-faithful cleaned_text from model-generated enriched_notes
- cleaned_text must NOT contain any model-generated content
- enriched_notes can contain domain annotations and explanations
- keep_for_corpus flag for corpus inclusion decisions
- drop_reason when content should not enter corpus

CRITICAL RULE: cleaned_text must be 100% source-faithful.
No hallucination, no paraphrase, no model-generated additions.
Only: fix OCR, remove noise, merge broken lines, restore formatting.
All model commentary/annotation goes into enriched_notes only.
"""

from __future__ import annotations

PROMPT_VERSION = "v2.0"

# ── Common JSON output schema ──────────────────────────────────────

CLEANING_OUTPUT_SCHEMA = """输出格式（严格JSON）：
{
  "cleaned_text": "清洗后的文本（必须100%忠实于原文，不得添加任何模型生成的内容）",
  "enriched_notes": "模型注释（可以包含领域背景解释、术语说明、技术上下文等，但不得混入cleaned_text）",
  "keep_for_corpus": true或false（该文本是否应进入预训练语料库）,
  "drop_reason": "如果keep_for_corpus为false，说明原因（如：非碳纤维内容/纯页眉页脚/过度损坏无法恢复/空白内容等）",
  "removed_noise_types": ["去除的噪声类型列表，如：page_header, page_footer, isbn_line, copyright_notice, publisher_name, garbled_text, broken_line_marker, repeated_header, watermark_text, blank_line_cluster"],
  "ocr_repairs": [
    {"original": "原文中的OCR错误片段", "repaired": "修复后的片段", "type": "错字/乱码/漏字/符号错误/断行/格式错误"}
  ],
  "technical_content_types": ["该文本包含的技术内容类型，如：formula, table, data_values, process_description, material_property, experimental_method, comparison, definition, citation_reference"],
  "uncertainty_notes": "对清洗结果的不确定性说明（如：某公式可能修复不完整、某数值OCR可能错误等）",
  "cleaning_actions": [
    {"action": "具体清洗动作", "original": "原文片段", "cleaned": "清洗后片段"}
  ],
  "confidence": 0.0-1.0
}"""

DOMAIN_FILTER_INSTRUCTION = """领域过滤规则：
1. 碳纤维领域相关内容 → keep_for_corpus = true
   包括：碳纤维材料、复合材料、PAN纤维、预氧化、碳化、石墨化、表面处理、上浆剂、
   CFRP、力学性能、热性能、电性能、制造工艺、测试方法、应用领域、缺陷分析、
   纤维结构、微观组织、界面性能、树脂基体、预浸料、成型工艺、连接技术、
   标准规范、质量控制、失效分析、无损检测、环境效应等
2. 明确非碳纤维内容 → keep_for_corpus = false, drop_reason = "非碳纤维内容"
   包括：纯数学推导（无碳纤维背景）、纯化学知识（无碳纤维应用）、
   通用材料科学（无碳纤维具体内容）、出版信息、版权声明、空白页等
3. 可疑内容 → keep_for_corpus = true, 在uncertainty_notes中标注领域相关性存疑"""

OCR_REPAIR_RULES = """OCR修复规则（按优先级）：
1. 断行修复：将因OCR换行导致的不完整句子合并为完整句子
   例："碳纤维复/\n合材料" → "碳纤维复合材料"
2. 错字修复：根据上下文推断并修复明显OCR错误字
   例："碳纤维复合材判" → "碳纤维复合材料"
   例："预氧丝的拉伸强廈" → "预氧丝的拉伸强度"
3. 乱码清理：删除无意义的乱码字符序列
   例："CFRP@@#%^" → "CFRP"
4. 符号修复：修复OCR错误的数学符号
   例："σ≥3.5GPa" 中若OCR为 "o≥3.5GPa" → "σ≥3.5GPa"
5. 数值修复：核对数值是否合理，修复明显的OCR数值错误
   例："3500GPq" → "3500 GPa" (单位错误)
6. 格式修复：修复表格错位、列表编号错误等
7. 不可修复标注：对无法可靠修复的内容，保留原样并标注 [uncertain_ocr]"""

BOILERPLATE_REMOVAL_RULES = """页眉页脚/出版信息去除规则：
1. 页眉模式：如"第X章 碳纤维XXX"、"碳纤维丛书"、"第X页"
2. 页脚模式：页码、版权声明、出版社名称
3. 出版信息：ISBN号、版权页、出版日期、出版社地址、定价等
4. 目录条目：纯目录页（只含章节标题和页码，无正文内容）
5. 水印文字：如"样本"、"内部资料"等
6. 重复标记：同一页眉在不同页面重复出现

去除这些内容时，在removed_noise_types中记录具体类型。
但如果页眉中包含有用的技术信息（如章节标题与内容相关），保留该信息部分。"""

# ── A. Chinese body text cleaning (v2.0) ──────────────────────────

ZH_BODY_CLEANING_PROMPT_V2 = """你是一个专业的中文技术文档OCR清洗专家。以下文本来自碳纤维领域的中文书籍或论文的OCR扫描。

【核心原则】
- cleaned_text 必须100%忠实于原文信息，不得添加任何模型推断或生成的信息
- 所有模型注释、领域背景解释、术语说明必须放在 enriched_notes 中，不得混入 cleaned_text
- 严格遵守领域过滤规则，正确标注 keep_for_corpus

{domain_filter}

{ocr_rules}

{boilerplate_rules}

{output_schema}

原文：
{raw_text}"""


# ── B. English body text cleaning (v2.0) ──────────────────────────

EN_BODY_CLEANING_PROMPT_V2 = """You are a professional English technical document OCR cleaning specialist. The following text comes from OCR scanning of carbon fiber domain English books or papers.

【CORE PRINCIPLES】
- cleaned_text must be 100% faithful to the original source information. No model-inferred or model-generated content may be added.
- All model annotations, domain context explanations, terminology notes must go into enriched_notes, NEVER into cleaned_text.
- Strictly follow domain filtering rules. Correctly mark keep_for_corpus.

Domain filtering rules:
1. Carbon fiber domain content → keep_for_corpus = true
   Includes: carbon fiber materials, composites, PAN fiber, preoxidation, carbonization, graphitization,
   surface treatment, sizing agents, CFRP, mechanical/thermal/electrical properties, manufacturing processes,
   testing methods, applications, defect analysis, fiber structure, microstructure, interface properties,
   resin matrix, prepreg, molding processes, joining techniques, standards, quality control, failure analysis,
   NDT, environmental effects, etc.
2. Clearly non-carbon-fiber content → keep_for_corpus = false, drop_reason = "non_carbon_fiber_content"
   Includes: pure math derivation (no CF context), pure chemistry (no CF application),
   general materials science (no specific CF content), publication info, copyright, blank pages, etc.
3. Suspicious content → keep_for_corpus = true, note domain relevance concerns in uncertainty_notes

OCR repair rules (by priority):
1. Broken line repair: merge incomplete sentences split by OCR line breaks
2. Typo repair: infer and fix obvious OCR character errors from context
3. Garbled text cleanup: remove meaningless garbled character sequences
4. Symbol repair: fix OCR errors in mathematical symbols
5. Value repair: check numerical values for plausibility, fix obvious OCR errors
6. Format repair: fix table misalignment, list numbering errors, etc.
7. Unrepairable: preserve as-is and mark with [uncertain_ocr]

Boilerplate removal rules:
1. Headers: chapter titles, book series names, page numbers used as headers
2. Footers: page numbers, copyright notices, publisher names
3. Publication info: ISBN, copyright page, publication dates, publisher addresses, prices
4. Table of contents entries: pure TOC pages (only chapter titles + page numbers, no content)
5. Watermark text: "sample", "internal document", etc.
6. Repeated markers: same header appearing across multiple pages

{output_schema}

Original text:
{raw_text}"""


# ── C. Formula-preserving cleaning (Chinese v2.0) ──────────────────

ZH_FORMULA_CLEANING_PROMPT_V2 = """你是一个专业的中文技术文档OCR清洗专家，专注于公式密集的碳纤维领域文本。

【核心原则】
- cleaned_text 必须100%忠实于原文信息，不得添加任何模型推断或生成的信息
- 公式部分必须严格保留，不得简化、改写、或删除
- 所有模型注释放在 enriched_notes 中
- 严格遵守领域过滤规则

公式处理规则：
1. 完整公式 → 原样保留，修复OCR符号错误但不改公式结构
2. OCR损坏但可推断的公式 → 尽力修复，在uncertainty_notes中标注
3. OCR损坏严重无法恢复的公式 → 标注 [formula_uncertain] 并保留原OCR文本块
4. 上下文暗示有公式但OCR完全丢失的 → 标注 [formula_missing]
5. 绝不编造公式中未出现的符号或数值

{domain_filter}

{ocr_rules}

{boilerplate_rules}

{output_schema}

原文：
{raw_text}"""


# ── D. Table-preserving cleaning (Chinese v2.0) ──────────────────

ZH_TABLE_CLEANING_PROMPT_V2 = """你是一个专业的中文技术文档OCR清洗专家，专注于表格密集的碳纤维领域文本。

【核心原则】
- cleaned_text 必须100%忠实于原文信息，不得添加任何模型推断或生成的信息
- 表格数据必须严格保留，不得删除任何数据行或列
- 绝不编造表格中没有的数据
- 所有模型注释放在 enriched_notes 中

表格处理规则：
1. 尽力将OCR损坏的表格重构为清晰的行列格式
2. 保留表格标题、注释、单位信息
3. 修复表格中的OCR错字，但不改变数据值
4. 损坏严重无法重构的表格 → 标注 [table_uncertain] 并保留原文本块
5. 表格中数值若OCR明显错误 → 保留原值，标注 [value_uncertain] 在uncertainty_notes中说明

{domain_filter}

{ocr_rules}

{boilerplate_rules}

{output_schema}

原文：
{raw_text}"""


# ── E. Mixed content cleaning (Chinese v2.0) ──────────────────

ZH_MIXED_CLEANING_PROMPT_V2 = """你是一个专业的中文技术文档OCR清洗专家。以下文本包含公式、表格和普通文本的混合内容，请保守处理。

【核心原则】
- cleaned_text 必须100%忠实于原文信息，不得添加任何模型推断或生成的信息
- 对公式部分：保守处理，不简化不修改；损坏严重的标注 [formula_uncertain]
- 对表格部分：尽量保留原始格式；损坏严重的标注 [table_uncertain]
- 对普通文本部分：正常清洗（修复OCR、去除噪声、合并断行）
- 所有模型注释放在 enriched_notes 中

{domain_filter}

{ocr_rules}

{boilerplate_rules}

{output_schema}

原文：
{raw_text}"""


# ── F. Header/footer handling (Chinese v2.0) ──────────────────

ZH_HEADER_FOOTER_CLEANING_PROMPT_V2 = """你是一个专业的中文技术文档OCR清洗专家。以下文本主要为页眉、页脚、版权声明、出版信息等非正文内容。

【处理规则】
1. 纯页眉页脚 → keep_for_corpus = false, drop_reason = "纯页眉页脚"
2. 版权声明/ISBN → keep_for_corpus = false, drop_reason = "出版信息"
3. 目录页（仅含章节标题和页码）→ keep_for_corpus = false, drop_reason = "纯目录"
4. 但如果页眉包含有用的章节标题（与碳纤维技术内容相关），提取该标题到enriched_notes中
5. 如果版权页包含碳纤维书籍的完整标题和作者信息，保留到enriched_notes中作为元数据

cleaned_text 仅保留任何可能有用的技术信息片段（如有）。
如果没有有用信息，cleaned_text 为空字符串。

{output_schema}

原文：
{raw_text}"""


# ── G. Empty/blank handling ──────────────────────────────────────

EMPTY_CHUNK_HANDLING = """该文本为空白或近乎空白（仅有少量无意义字符），无需清洗。
自动设置：keep_for_corpus = false, drop_reason = "空白内容", cleaned_text = """""


# ── Prompt selector (v2.0) ──────────────────────────────────────

def get_cleaning_prompt(language: str, raw_text: str, chunk_type: str = "body") -> str:
    """Select and format the appropriate v2.0 cleaning prompt.

    Routing:
    - body → body text cleaning prompt
    - formula → formula-preserving cleaning prompt
    - table / table_uncertain → table-preserving cleaning prompt
    - mixed → mixed content conservative cleaning prompt
    - header_footer → header/footer handling prompt
    - empty → returns empty chunk handling instruction (no LLM call needed)
    """
    if chunk_type == "empty":
        return EMPTY_CHUNK_HANDLING

    if chunk_type == "header_footer":
        template = ZH_HEADER_FOOTER_CLEANING_PROMPT_V2
    elif chunk_type == "formula":
        template = ZH_FORMULA_CLEANING_PROMPT_V2 if language == "zh" else EN_BODY_CLEANING_PROMPT_V2
    elif chunk_type in ("table", "table_uncertain"):
        template = ZH_TABLE_CLEANING_PROMPT_V2 if language == "zh" else EN_BODY_CLEANING_PROMPT_V2
    elif chunk_type == "mixed":
        template = ZH_MIXED_CLEANING_PROMPT_V2 if language == "zh" else EN_BODY_CLEANING_PROMPT_V2
    else:  # body
        template = ZH_BODY_CLEANING_PROMPT_V2 if language == "zh" else EN_BODY_CLEANING_PROMPT_V2

    # Inject sub-prompts
    result = template
    result = result.replace("{domain_filter}", DOMAIN_FILTER_INSTRUCTION)
    result = result.replace("{ocr_rules}", OCR_REPAIR_RULES)
    result = result.replace("{boilerplate_rules}", BOILERPLATE_REMOVAL_RULES)
    result = result.replace("{output_schema}", CLEANING_OUTPUT_SCHEMA)
    result = result.replace("{raw_text}", raw_text)

    return result


def get_knowledge_extraction_prompt(cleaned_text: str) -> str:
    """Format the knowledge extraction prompt (unchanged from v1.0)."""
    return KNOWLEDGE_EXTRACTION_PROMPT.replace("{cleaned_text}", cleaned_text)


def get_sft_generation_prompt(cleaned_text: str) -> str:
    """Format the SFT candidate generation prompt (unchanged from v1.0)."""
    return SFT_GENERATION_PROMPT.replace("{cleaned_text}", cleaned_text)


def get_quality_verification_prompt(cleaned_text: str, original_text: str) -> str:
    """Format the quality verification prompt (enhanced for v2.0)."""
    return QUALITY_VERIFICATION_PROMPT_V2.replace(
        "{cleaned_text}", cleaned_text
    ).replace("{original_text}", original_text[:800])


# ── Enhanced quality verification prompt (v2.0) ──────────────────

QUALITY_VERIFICATION_PROMPT_V2 = """你是一个独立的文本质量验证专家。请独立评估以下清洗后的碳纤维领域文本质量。

【验证原则】
1. 你必须独立客观，不得因为清洗模型声称质量好就给出高分
2. 重点关注：cleaned_text是否100%忠实于原文（无幻觉、无添加、无改写）
3. 重点关注：领域过滤是否正确（keep_for_corpus标注是否合理）
4. 重点关注：OCR修复是否准确（修复后的值是否合理）
5. 重点关注：enriched_notes是否合理分离（不应混入cleaned_text）

评估维度：
1. clarity (清晰度): 文本是否可读、表达是否清晰？(0-1)
2. completeness (完整性): 是否保留了源文本的重要信息？(0-1)
3. consistency (一致性): 内容是否前后一致、无矛盾？(0-1)
4. feasibility (可行性): 是否可用于预训练或微调？(0-1)
5. complexity (复杂度): 内容的技术深度如何？(0-1)
6. domain_relevance (领域相关性): 是否属于碳纤维领域？(0-1)

同时检测以下问题：
- hallucination_in_corpus: cleaned_text中是否包含模型编造的内容？
- source_faithfulness_violation: cleaned_text是否偏离了原文？
- formula_loss: 重要公式是否被删除或损坏？
- table_data_loss: 表格数据是否丢失？
- over_cleaning: 是否删除了不该删除的内容？
- domain_filter_error: keep_for_corpus标注是否错误？
- enrichment_leakage: enriched_notes中的内容是否混入了cleaned_text？

评估结论：
- passed: 质量合格
- needs_revision: 需要修改
- failed: 应丢弃

输出格式（JSON）：
{
  "clarity": 0.0-1.0,
  "completeness": 0.0-1.0,
  "consistency": 0.0-1.0,
  "feasibility": 0.0-1.0,
  "complexity": 0.0-1.0,
  "domain_relevance": 0.0-1.0,
  "verdict": "passed|needs_revision|failed",
  "issues": ["问题描述列表"],
  "specific_problems": {
    "hallucination_in_corpus": true/false,
    "source_faithfulness_violation": true/false,
    "formula_loss": true/false,
    "table_data_loss": true/false,
    "over_cleaning": true/false,
    "domain_filter_error": true/false,
    "enrichment_leakage": true/false
  }
}

清洗后文本：
{cleaned_text}

原文参考（截取前800字）：
{original_text}"""


# ── Keep v1.0 prompts for backward compatibility ──────────────────

ZH_CLEANING_PROMPT = ZH_BODY_CLEANING_PROMPT_V2  # alias
EN_CLEANING_PROMPT = EN_BODY_CLEANING_PROMPT_V2  # alias
ZH_FORMULA_CLEANING_PROMPT = ZH_FORMULA_CLEANING_PROMPT_V2  # alias
EN_FORMULA_CLEANING_PROMPT = EN_BODY_CLEANING_PROMPT_V2  # alias
ZH_TABLE_CLEANING_PROMPT = ZH_TABLE_CLEANING_PROMPT_V2  # alias
EN_TABLE_CLEANING_PROMPT = EN_BODY_CLEANING_PROMPT_V2  # alias
ZH_CONSERVATIVE_CLEANING_PROMPT = ZH_MIXED_CLEANING_PROMPT_V2  # alias
EN_CONSERVATIVE_CLEANING_PROMPT = EN_BODY_CLEANING_PROMPT_V2  # alias

# v1.0 prompts kept as-is (knowledge extraction and SFT generation unchanged)
KNOWLEDGE_EXTRACTION_PROMPT = """你是一个碳纤维领域知识提取专家。

任务：从以下清洗后的碳纤维文本中提取原子知识单元。

提取要求：
1. 每个知识单元必须是独立的、可验证的单一知识点
2. 知识单元必须完全基于源文本，禁止编造任何信息
3. 保留数值、单位、条件、范围等精确信息
4. 标注知识类型：定义、性质、工艺、机理、应用、测量、缺陷、对比、公式、表格
5. 提取涉及的实体和关系
6. 提取数值信息（含单位）

输出格式（JSON数组）：
[
  {
    "topic": "主题分类",
    "subtopic": "子主题",
    "knowledge_type": "definition|property|process|mechanism|application|measurement|defect|comparison|equation|table|other",
    "claim": "知识点的核心陈述",
    "evidence_text": "支持该知识点的原文片段（必须准确引用）",
    "entities": ["涉及的关键实体"],
    "relations": ["实体间关系描述"],
    "conditions": ["适用条件或前提"],
    "numeric_values": [{"value": 数值, "unit": 单位, "context": 上下文}]
  }
]

源文本：
{cleaned_text}"""

SFT_GENERATION_PROMPT = """你是一个碳纤维领域教学样本生成专家。

任务：从以下清洗后的碳纤维文本生成候选监督微调(SFT)样本。

生成要求：
1. 每个样本必须完全基于源文本中的信息，禁止编造
2. 只有当源文本包含足够信息时才生成样本
3. 生成不同任务类型的样本：问答、解释、提取、分类、对比、过程推理
4. 标注难度等级
5. 保留源文本引用

输出格式（JSON数组）：
[
  {
    "task_type": "qa|explanation|extraction|classification|comparison|process_reasoning",
    "instruction": "任务指令",
    "input": "输入内容（可为空）",
    "output": "期望输出回答",
    "evidence_text": "支持该回答的原文片段",
    "difficulty": "easy|medium|hard"
  }
]

源文本：
{cleaned_text}"""

QUALITY_VERIFICATION_PROMPT = QUALITY_VERIFICATION_PROMPT_V2  # alias