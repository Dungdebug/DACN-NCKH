#!/usr/bin/env python3
"""
Export all Office deliverables (Excel .xlsx and PowerPoint .pptx) directly from
the live comparison dashboard data (`src.web.app.get_comparison_dashboard`) and
`src.web.office_exporter`, then verify each generated file by reopening it.
"""

import asyncio
import sys
from pathlib import Path

from openpyxl import load_workbook
from pptx import Presentation

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.web.app import get_comparison_dashboard
from src.web.office_exporter import export_comparison_to_pptx, export_comparison_to_xlsx


DOCS_DIR = ROOT_DIR / "docs"
XLSX_PATH = DOCS_DIR / "Thong_Ke_So_Sanh_NCKH.xlsx"
PPTX_PATH = DOCS_DIR / "Slide_Bao_Ve_NCKH_SA_CMS_WaveCMS.pptx"


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_data = asyncio.run(get_comparison_dashboard())

    xlsx_buf = export_comparison_to_xlsx(dashboard_data)
    XLSX_PATH.write_bytes(xlsx_buf.getvalue())
    wb = load_workbook(str(XLSX_PATH))
    print(f"Exported Excel: {XLSX_PATH} | sheets={wb.sheetnames}")

    pptx_buf = export_comparison_to_pptx(dashboard_data, figures_dir=str(ROOT_DIR / "results" / "figures"))
    PPTX_PATH.write_bytes(pptx_buf.getvalue())
    prs = Presentation(str(PPTX_PATH))
    print(f"Exported PowerPoint: {PPTX_PATH} | slides={len(prs.slides)}")


if __name__ == "__main__":
    main()
