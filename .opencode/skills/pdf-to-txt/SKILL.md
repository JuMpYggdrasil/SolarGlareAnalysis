---
name: pdf-to-txt
description: Convert reference PDFs and standards into AI-parseable plain-text extracts. Use when a task mentions extracting text from a PDF, "PDF to txt", "ทำ pdf เป็น txt", "normalize"ing an extracted document, or preparing a reference/standard (`references/_*.txt`) for reading by an LLM. Produces `=== PAGE N ===` page markers, UTF-8 no-BOM (LF) files, and the project house style documented in `references/_FORMAT_62471.txt` (units like `W/(m^2*sr)`, `^` exponents, `*` multiply, ×-free equations, decimal commas→dots, Greek preserved).
---

# PDF → TXT (extract + normalize to house style)  (เป็นภาษาไทยหลัก)

เปลี่ยน `references/*.pdf` → `references/_*.txt` ให้ AI/คนอ่านสมการ/หน่วย/ตัวเลขตรง 1:1
กับที่โค้ดใช้ (`hazard.py`, `iec62471.py`).

## ภาพรวม (ต่อ PDF หนึ่งไฟล์)
1. Extract กับ PyMuPDF → `=== PAGE N ===` คั่นหน้า, UTF-8 ไม่มี BOM, LF
2. Normalize ให้ตรง house style (หน่วย/`^`/ทศนิยม/สมการกอง/ตาราง)
3. ตรวจ-ทานตาม checklist ห้ามข้าม

สคริปต์อยู่ในโฟลเดอร์นี้: `.opencode/skills/pdf-to-txt/scripts/`
(ชื่อ power user: `pdf_extract.py`, `normalize_pdf_text.py`)

## ข้อบังคับ (ทำทุกครั้ง)
- รันด้วย interpreter ของโปรเจกต์เท่านั้น: `.venv\Scripts\python.exe` (แก้ path ให้ถูก)
- Console เครื่องนี้เป็น cp874 → ตั้งก่อนรัน Python ที่ print อักขระพิเศษ:
  `$env:PYTHONIOENCODING='utf-8'`
- **ห้าม**แตะไฟล์ที่ได้ด้วย PowerShell `Get-Content`/`Set-Content` (PS 5.1 อ่าน UTF-8
  ไร้ BOM ผิดเป็น cp874 → mojibake ทุกอักขระพิเศษ + เติม BOM ให้เอง) — ใช้ Read tool
  หรือ Python ที่ระบุ `encoding="utf-8"` เสมอ
- ให้อ่าน/ไปต่อกับปัญหา encoding ที่สงสัยด้วย Python/short Read เท่านั้น

## ขั้นตอน

### 1) Extract
```
.venv\Scripts\python.exe .opencode\skills\pdf-to-txt\scripts\pdf_extract.py "references\<file>.pdf"
```
- default output = `references\<file>.txt`; ระบุ output path เป็น arg 2 ได้
- ยืนยันในผลว่า `pages: N` ตรงกับจำนวนหน้าจริง; ไฟล์ต้องขึ้นต้น `=== PAGE 1 ===`
- ถ้า PDF เป็นสแกน (ต้องการ OCR) หรือตารางหลัก ได้ข้อความปนกัน — จดและแทรก
  `--- <manual> ---` แล้วเติมข้อความที่อ่านถูก (house style: ดูข้อ 3) ตรงที่หาย

### 2) Normalize (ผ่านสคริปต์ ครั้งเดียว)
```
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe .opencode\skills\pdf-to-txt\scripts\normalize_pdf_text.py "references\<file>.txt"
```
- default = loose (report count เท่านั้น, อ่านซ้ำได้ = idempotent)
- เอกสารที่ยังไม่ normalize ให้รันพร้อม `--strict` (assert count ตามที่กำหนดใน config)
- `--dry-run` = ดู report โดยไม่เขียนไฟล์
- config ฝังในตัวสคริปต์ (บล็อก RULES) — กลุ่ม A–E, H = กติกาทั่วไป (หน่วย/ตัวห้อย/
  เลขยกกำลัง/ทศนิยม/จุดคูณ); กลุ่ม F (EQ stacks) + G (figure-legend) = เฉพาะเอกสาร
  (ค่า default เป็นของ BS EN 62471) → ถ้าเป็น PDF อื่น: เอา F/G ออกหรือแก้ literal
  ให้ตรง glyph ของ PDF นั้น

### 3) ปรับข้อที่สคริปต์จับไม่ได้ (ใช้ Edit tool)
- สมการ/หน่วย/เลขที่เหลือยังอ่านไม่ง่าย → แก้เป็น 1 บรรทัด ASCII ตาม house style
- เปิดเอกสารหมวด D–G: `references/_FORMAT_62471.txt` (คู่มือ notation เต็ม) และ
  `references/_HANDWRITING_CHECKLIST.txt` (เช็กลิสต์พิมพ์มือ) เพื่อยึดสไตล์เดียวกัน
- ตัวอย่างที่ต้องได้ (จากมาตรฐานจริง):
  `Σ L_λ(λ) * R(λ) * ∆λ ≤ 50000 / (α * t^0.25) W/(m^2*sr)`
  `t_max = 10^6 / L_B [s]`
  `E_λ = dΦ(λ) / (dA * dλ)`

### 4) ตรวจ-ทาน (ห้ามข้าม)
รันด้วย Python (ไม่ใช่ PowerShell):
1. byte แรก 3 ตัว ≠ `EF BB BF` (ไม่มี BOM)
2. grep หา `⋅`, `·`, `m-2`, `nm-1`, `sr-1`, `106 nm`, `104 s`, `t 0,` → ทั้งหมดต้อง 0
3. regex `(?<=\d),(?=\d)` → 0
4. `=== PAGE N ===` ครบทุกหน้า; tag สมการ `(n.m)` ที่พบในเนื้อความครบ
5. λ/α/θ/Ω/Ω/Φ/µ, `≤ ≥ ∫ ∑ ∑∑ ∆ ½` ยังมีอยู่ (กรีกไม่หาย)
6. เปิด Read 2–3 จุดที่เคยเป็นสมการกอง → อ่านเป็นสูตร 1 บรรทัด

## ได้ผลแล้ว
- ปกติเขียนลง `references/_<stem>.txt` (เครื่องหมาย `_` นำหน้า = เป็น extract)
- รายงานสั้นๆ ให้ user: path, หน้า-X, เจอกี่ transformed, เหลือ junk กี่จุด (ถ้ามี)

## อ้างอิงสไตล์ (repo)
- `references/_FORMAT_62471.txt` — notation guide (encoding/BOM, หน่วย, สมการ, tag, tables)
- `references/_HANDWRITING_CHECKLIST.txt` — ถ้าจะพิมพ์/เขียนไฟล์ด้วยมือ
- `AGENTS.md` — โน้ต "2026-09-16" บรรทัด references