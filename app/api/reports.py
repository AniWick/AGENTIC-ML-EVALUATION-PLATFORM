from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from pathlib import Path
from app.core.database import db
from app.core.storage import storage_manager

router = APIRouter(tags=["reports"])

@router.get("/projects/{project_id}/reports")
def list_project_reports(project_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, project_id, title, summary, created_at FROM reports WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

@router.get("/reports/{report_id}")
def get_report(report_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Report not found")
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "title": row["title"],
            "summary": row["summary"],
            "content_markdown": row["content_markdown"],
            "created_at": row["created_at"]
        }

@router.get("/reports/{report_id}/download")
def download_markdown_report(report_id: str):
    md_file = storage_manager.reports_dir / f"{report_id}.md"
    if md_file.exists():
        return FileResponse(
            path=str(md_file),
            filename=f"{report_id}.md",
            media_type="text/markdown"
        )
    
    # Fallback from DB
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content_markdown FROM reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Report not found")
        return PlainTextResponse(row["content_markdown"], media_type="text/markdown")
