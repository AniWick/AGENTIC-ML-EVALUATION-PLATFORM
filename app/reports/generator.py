import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from app.core.storage import storage_manager
from app.core.database import db
from app.llm.ollama_client import ollama_client
from app.llm.prompts import REPORT_GENERATION_PROMPT

class ReportGenerator:
    """Generates a single comprehensive Markdown evaluation report."""

    @staticmethod
    def generate_and_save(state: Any) -> Dict[str, Any]:
        project_id = state.project_id
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{project_id}_{timestamp}"

        # 1. Build Markdown content
        md_lines = [
            f"# ML Model & Dataset Evaluation Report",
            f"**Project:** `{project_id}` | **Generated:** `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`",
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
                f"- **Recall:** `{b.get('recall_macro', 'N/A')}`"
            ])
        else:
            md_lines.append("*No baseline performance metrics recorded.*")

        md_lines.extend([
            "",
            "## 2. Data Quality Audit",
        ])
        if state.data_quality_metrics:
            dq = state.data_quality_metrics
            md_lines.extend([
                f"- **Missing Values:** `{dq.get('missing_values_count', 0)}` cells",
                f"- **Duplicate Rows:** `{dq.get('duplicate_rows_count', 0)}` rows",
                f"- **Class Imbalance Detected:** `{'Yes (Ratio: ' + str(round(dq.get('imbalance_ratio', 0), 2)) + ':1)' if dq.get('class_imbalance_detected') else 'No'}`",
                f"- **Outliers Count:** `{dq.get('outliers_count', 0)}` values",
                f"- **Constant Columns:** `{dq.get('constant_columns', [])}`"
            ])
        else:
            md_lines.append("*No data quality metrics recorded.*")

        md_lines.extend([
            "",
            "## 3. Robustness Benchmark (10% Gaussian Noise)",
        ])
        if state.robustness_metrics:
            rob = state.robustness_metrics
            md_lines.extend([
                f"- **Original Score:** `{rob.get('original_score')}`",
                f"- **Perturbed Score:** `{rob.get('perturbed_score')}`",
                f"- **Performance Drop:** `{round(rob.get('drop_percentage', 0), 2)}%`",
                f"- **Sensitivity Assessment:** `{'Sensitive to Noise' if rob.get('is_sensitive') else 'Robust'}`"
            ])
        else:
            md_lines.append("*No robustness benchmarks recorded.*")

        md_lines.extend([
            "",
            "## 4. Experimental Transformations & Impact",
        ])
        if state.experiments:
            for exp in state.experiments:
                comp = exp.get("comparison", {})
                md_lines.extend([
                    f"### Experiment: `{exp.get('operation', 'Transformation')}`",
                    f"- **Dataset Version:** `{exp.get('input_dataset_version')} -> {exp.get('output_dataset_version')}`",
                    f"- **Outcome:** `{'Improved Performance' if comp.get('is_improved') else 'Neutral / Degraded'}`",
                    f"- **Summary of Changes:**"
                ])
                for chg in comp.get("summary_of_changes", []):
                    md_lines.append(f"  - {chg}")
                if exp.get("agent_reasoning"):
                    md_lines.append(f"- **Agent Reasoning:** *{exp.get('agent_reasoning')}*")
        else:
            md_lines.append("*No data transformations were required.*")

        # 2. LLM Interpretation Synthesis
        context_for_llm = json.dumps({
            "baseline": state.baseline_metrics,
            "data_quality": state.data_quality_metrics,
            "robustness": state.robustness_metrics,
            "experiments": state.experiments,
            "final": state.latest_metrics
        }, indent=2)

        llm_analysis = ollama_client.generate(
            prompt=REPORT_GENERATION_PROMPT.format(context=context_for_llm)
        )
        if not llm_analysis or len(llm_analysis.strip()) < 20:
            llm_analysis = (
                "### Conclusion & Recommendation\n"
                "The model was successfully evaluated against the provided dataset. "
                "Baseline quality and performance metrics have been established. "
                "Where class imbalance or missing values were present, automated versioned transformations were benchmarked."
            )

        md_lines.extend([
            "",
            "## 5. Diagnostic Synthesis & Recommendations",
            llm_analysis
        ])

        final_markdown = "\n".join(md_lines)

        # 3. Save ONLY 1 Markdown file
        report_file = storage_manager.reports_dir / f"{report_id}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        # Record in SQLite database
        with db._get_connection() as conn:
            conn.execute("""
                INSERT INTO reports (id, project_id, title, summary, content_markdown, content_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id, project_id, f"Report for {project_id}", "Evaluation Report (Markdown)",
                final_markdown, "{}", datetime.utcnow().isoformat()
            ))
            conn.commit()

        return {
            "report_id": report_id,
            "file_path": str(report_file),
            "markdown": final_markdown
        }

report_generator = ReportGenerator()
