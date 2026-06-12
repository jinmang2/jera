"""Regenerate the tiny HWPX (OWPML) test fixture deterministically.

    uv run python python/jera/tests/fixtures/hwpx/make_fixture.py

Writes `sample.hwpx` next to this file. The parser only needs `Contents/section*.xml`; we
include a `mimetype` entry for realism. Table is nested inside a paragraph (as real HWPX does)
so the parser's table-vs-paragraph separation is exercised.
"""

from __future__ import annotations

import pathlib
import zipfile

HERE = pathlib.Path(__file__).parent

SECTION0 = """<?xml version="1.0" encoding="UTF-8"?>
<hs:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"
        xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section">
  <hp:p><hp:run><hp:t>개요</hp:t></hp:run></hp:p>
  <hp:p><hp:run><hp:t>이 문서는 HWPX 파서 픽스처입니다.</hp:t></hp:run></hp:p>
  <hp:p><hp:run><hp:t>결과 요약</hp:t></hp:run></hp:p>
  <hp:p><hp:run><hp:ctrl><hp:tbl>
    <hp:tr>
      <hp:tc><hp:subList><hp:p><hp:run><hp:t>항목</hp:t></hp:run></hp:p></hp:subList></hp:tc>
      <hp:tc><hp:subList><hp:p><hp:run><hp:t>값</hp:t></hp:run></hp:p></hp:subList></hp:tc>
    </hp:tr>
    <hp:tr>
      <hp:tc><hp:subList><hp:p><hp:run><hp:t>매출</hp:t></hp:run></hp:p></hp:subList></hp:tc>
      <hp:tc><hp:subList><hp:p><hp:run><hp:t>1000</hp:t></hp:run></hp:p></hp:subList></hp:tc>
    </hp:tr>
  </hp:tbl></hp:ctrl></hp:run></hp:p>
</hs:sec>
"""


def build(path: pathlib.Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/hwp+zip")
        zf.writestr("Contents/section0.xml", SECTION0)


if __name__ == "__main__":
    out = HERE / "sample.hwpx"
    build(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
