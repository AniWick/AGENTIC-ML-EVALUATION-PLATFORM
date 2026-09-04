import json
from typing import Dict, Any
from datetime import datetime

from app.core.storage import storage_manager
from app.core.database import db
from app.llm.qwen_client import qwen_client
from app.llm.prompts import REPORT_GENERATION_PROMPT


class ReportGenerator:
    """Generates a single comprehensive Markdown evaluation report."""

    @staticmethod
    def generate_and_save(state: Any) -> Dict[str, Any]:
        project_id = state.project_id
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{project_id}_{timestamp}"

        # ---------------------------------------------------------
        # 1. Build deterministic Markdown report
        # ---------------------------------------------------------

        md_lines = [
            "# ML Model & Dataset Evaluation Report",
            (
                f"**Project:** `{project_id}` | "
                f"**Generated:** "
                f"`{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`"
            ),
            f"**User Objective:** *{state.user_goal}*",
            "",
            "---",
            "## 1. Model Baseline Performance",
        ]

        if state.baseline_metrics:
            b = state.baseline_metrics

            md_lines.extend([
                f"- **Accuracy:** `{b.get('accuracy', 'N/A')}`",
                f"- **Macro F1:** `{b.get('f1_macro', 'N/A')}`",
                f"- **Minority Class F1:** `{b.get('f1_minority', 'N/A')}`",
                f"- **Precision:** `{b.get('precision_macro', 'N/A')}`",
                f"- **Recall:** `{b.get('recall_macro', 'N/A')}`",
            ])
        else:
            md_lines.append(
                "*No baseline performance metrics recorded.*"
            )

        # ---------------------------------------------------------
        # 2. Data Quality
        # ---------------------------------------------------------

        md_lines.extend([
            "",
            "## 2. Data Quality Audit",
        ])

        if state.data_quality_metrics:
            dq = state.data_quality_metrics

            imbalance_detected = dq.get(
                "class_imbalance_detected",
                False
            )

            imbalance_ratio = dq.get(
                "imbalance_ratio",
                0
            )

            if imbalance_detected:
                imbalance_text = (
                    f"Yes (Ratio: "
                    f"{round(imbalance_ratio, 2)}:1)"
                )
            else:
                imbalance_text = "No"

            md_lines.extend([
                (
                    f"- **Missing Values:** "
                    f"`{dq.get('missing_values_count', 0)}` cells"
                ),
                (
                    f"- **Duplicate Rows:** "
                    f"`{dq.get('duplicate_rows_count', 0)}` rows"
                ),
                (
                    f"- **Class Imbalance Detected:** "
                    f"`{imbalance_text}`"
                ),
                (
                    f"- **Outliers Count:** "
                    f"`{dq.get('outliers_count', 0)}` values"
                ),
                (
                    f"- **Constant Columns:** "
                    f"`{dq.get('constant_columns', [])}`"
                ),
            ])
        else:
            md_lines.append(
                "*No data quality metrics recorded.*"
            )

        # ---------------------------------------------------------
        # 3. Robustness
        # ---------------------------------------------------------

        md_lines.extend([
            "",
            "## 3. Robustness Benchmark (10% Gaussian Noise)",
        ])

        if state.robustness_metrics:
            rob = state.robustness_metrics

            drop_percentage = rob.get(
                "drop_percentage",
                0
            )

            md_lines.extend([
                (
                    f"- **Original Score:** "
                    f"`{rob.get('original_score', 'N/A')}`"
                ),
                (
                    f"- **Perturbed Score:** "
                    f"`{rob.get('perturbed_score', 'N/A')}`"
                ),
                (
                    f"- **Performance Drop:** "
                    f"`{round(drop_percentage, 2)}%`"
                ),
                (
                    f"- **Sensitivity Assessment:** "
                    f"`{'Sensitive to Noise' if rob.get('is_sensitive') else 'Robust'}`"
                ),
            ])
        else:
            md_lines.append(
                "*No robustness benchmarks recorded.*"
            )

        # ---------------------------------------------------------
        # 4. Transformations and experiments
        # ---------------------------------------------------------

        md_lines.extend([
            "",
            "## 4. Experimental Transformations & Impact",
        ])

        if state.experiments:
            for exp in state.experiments:

                comp = exp.get(
                    "comparison",
                    {}
                )

                outcome = (
                    "Improved Performance"
                    if comp.get("is_improved")
                    else "Neutral / Degraded"
                )

                md_lines.extend([
                    (
                        f"### Experiment: "
                        f"`{exp.get('operation', 'Transformation')}`"
                    ),
                    (
                        f"- **Dataset Version:** "
                        f"`{exp.get('input_dataset_version')} "
                        f"-> "
                        f"{exp.get('output_dataset_version')}`"
                    ),
                    f"- **Outcome:** `{outcome}`",
                    "- **Summary of Changes:**",
                ])

                for change in comp.get(
                    "summary_of_changes",
                    []
                ):
                    md_lines.append(
                        f"  - {change}"
                    )

                if exp.get("agent_reasoning"):
                    md_lines.append(
                        f"- **Agent Reasoning:** "
                        f"*{exp.get('agent_reasoning')}*"
                    )

        else:
            md_lines.append(
                "*No data transformations were required.*"
            )

        # ---------------------------------------------------------
        # 5. Qwen Diagnostic Synthesis
        # ---------------------------------------------------------

        context_for_llm = json.dumps(
            {
                "baseline": state.baseline_metrics,
                "data_quality": state.data_quality_metrics,
                "robustness": state.robustness_metrics,
                "experiments": state.experiments,
                "final": state.latest_metrics,
            },
            indent=2,
        )

        try:
            llm_analysis = qwen_client.generate(
                prompt=REPORT_GENERATION_PROMPT.format(
                    context=context_for_llm
                )
            )
        except Exception:
            llm_analysis = ""

        if (
            not llm_analysis
            or len(llm_analysis.strip()) < 20
        ):
            llm_analysis = (
                "### Conclusion & Recommendation\n"
                "The model was successfully evaluated against "
                "the provided dataset. Baseline performance, "
                "data quality, and robustness metrics were "
                "calculated deterministically. Where data "
                "quality issues were detected, automated "
                "versioned transformations were applied and "
                "re-evaluated."
            )

        md_lines.extend([
            "",
            "## 5. Diagnostic Synthesis & Recommendations",
            llm_analysis,
        ])

        final_markdown = "\n".join(md_lines)

        # ---------------------------------------------------------
        # 6. Save exactly ONE Markdown report
        # ---------------------------------------------------------

        report_file = (
            storage_manager.reports_dir
            / f"{report_id}.md"
        )

        with open(
            report_file,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(final_markdown)

        # ---------------------------------------------------------
        # 7. Save report metadata/content to SQLite
        # ---------------------------------------------------------

        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reports
                (
                    id,
                    project_id,
                    title,
                    summary,
                    content_markdown,
                    content_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    project_id,
                    f"Report for {project_id}",
                    "Evaluation Report (Markdown)",
                    final_markdown,
                    "{}",
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()

        return {
            "report_id": report_id,
            "file_path": str(report_file),
            "markdown": final_markdown,
        }


report_generator = ReportGenerator()
