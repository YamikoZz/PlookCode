# PlookCode — สถานะล่าสุด 2026-11-08 (เย็น): รุ่น 200M เทรน+SFT เสร็จ

- 2026-11-08 รุ่น 193M (G4) เทรนหลักครบ 7,600 steps, val_loss 10→**1.229** (exit 0, ~6.9 ชม., 3.62 วิ/step, ไม่หลุดเลย)
- 2026-11-08 SFT 200M: สร้างชุดรวม 160,056 ข้อ (chat-v4 116,951 + hf-code ตัด Python เหลือ ~38k เก็บ Lua/JS/TS/C/C++ ครบ + ai-gen 436)
  -> `prepare-instructions` context 2048 reuse tokenizer -> train 119,712 / val 6,646 / test 6,604 (ไฟล์ SFT ต้อง <128 MiB)
  SFT `configs/code-sft-200m.json` 1,500 steps --init จาก run-200m -> val_loss **0.886** (~8 นาที). run บน Drive `run-200m-sft`
  สคริปต์ `scripts/colab_200m_sft.sh` (resumable บน Drive)
- 2026-11-08 chat-probes รุ่น 200M-SFT: Python is_even ✅, JS sum_list ✅, Lua factorial ✗ (คืน table), อธิบาย for loop วกวน
  => เขียนโค้ด Py/JS ได้จริง (ก้าวจาก 20M ที่โจทย์ใหม่ 0%); Lua + สนทนาทั่วไปยังอ่อน
- 2026-11-08 ติด: repo.zip บน Colab เป็น snapshot เก่า ไม่มี `--suite`/exam.py -> อัป repo-cur.zip ใหม่ unzip /content/lbcur แล้วสอบใหม่
- 2026-11-08 ผล code-eval รุ่น 200M-SFT (step 1500, model-only, samples 8):
  core: seen/python pass@1 **0.78** (pass@8 0.829), seen/js pass@1 **0.866** (0.89);
        unseen/python pass@1 **0.125** (pass@8 0.25), unseen/js pass@1 **0.0**
  exam (90 ข้อ Lua/TS/JS ไม่เคยเห็น): **0/90 ทุกภาษา** (pass@1 0.0)
  ตรวจคำตอบจริง: เขียนไวยากรณ์ถูก (JS/TS valid) แต่ตอบโจทย์ผิด เช่น count_peaks -> `filter(x>0)`; Lua ยัง error/มั่ว
  => สรุปตรงไปตรงมา: **จำรูปแบบที่เคยเห็นได้ดี (78-87%) แต่ยังไม่ generalize โจทย์ใหม่** (เหมือน 20M; unseen/py ขยับ 0->12.5%, และเขียนภาษาถูกแล้ว)
  ไฟล์ผลเก็บที่ `workspace/eval-200m/` (code-eval.json, exam-eval.json, chat-probes.json)
- 2026-11-08 ปิด G4 (`colab stop`), ไม่มี session ค้าง, units 106.22, rate 0/hr. โมเดลอยู่ที่ Drive `run-200m-sft/`
- 2026-11-08 โหลดโมเดลลงเครื่องผู้ใช้: `workspace/run-200m-sft/` (best.pt 1.6GB + tokenizer.json ผ่าน colab download ตรง 1.6GB ได้)
  `Try-200M.cmd` เปิดแชท CPU (`chat --model-only --device cpu`); โหลด 2 วิ ตอบ 1 วิ/คำถาม
- 2026-11-08 **A5 เครื่องคิดเลขในแชท** (บางส่วน): `tools.auto_math` แปลงคำถามเลขเป็น expression แล้วคิดด้วย `calculate()` (AST ปลอดภัย)
  wire ใน `chat.py` โหมด model-only -> ถามเลขได้คำตอบถูก 100% (25*10=250, 48+76=124), คำถามโค้ดไม่โดนจับ
  tests/test_automath.py (3 เทสต์) ผ่าน; ชุดทดสอบรวม 91 ผ่าน (skip 2 lua)
  ยังไม่ทำ: โจทย์เลขแบบประโยค/หลายขั้น (ต้องให้โมเดลเขียนโค้ดแล้วรันในแซนด์บ็อกซ์ — เครื่องบ้านไม่มี Docker)
- 2026-11-08 ทดสอบเลข 16 ข้อ (ทศนิยม/หาร/ยกกำลัง/เศษ/เลขใหญ่): เครื่องคิดเลขในแชท 16/16, โมเดลเดาเอง 0/16
- 2026-11-08 **Distillation เฟส 1 (pilot)**: ครู Qwen/Qwen2.5-Coder-32B-Instruct (license apache-2.0 เช็กจาก HF API)
  บน G4 ด้วย vLLM 0.33 (venv แยกผ่าน uv; ต้องลง `ninja` + PATH) -> 6,000 โจทย์ใน 31 นาที (~1,100 tok/s)
  ตรวจด้วย run_tests: เก็บ **1,486/6,000 (25%)** (py 387 / js 405 / ts 353 / lua 341)
  ทิ้ง: JSON พัง 1,189, ค่าเทสต์ผิดชนิด 1,166, เฉลยครูผิด 1,561, ซ้ำ 480
  เทรนต่อจาก run-200m-sft 400 steps (distill 1,486 + replay SFT เดิม 12,000) -> val 0.907->0.913
  **ผล (step 400)**: core 137/196 (เท่าเดิม; unseen/py 2->4/16, seen/js 71->69), **exam 0/90 เท่าเดิม**
  ดูคำตอบ: ยังไม่เข้าใจโจทย์ (rotate_right -> `map(x => x + k)`), Lua เขียนได้แต่ตรรกะผิด
  บทเรียน: best.pt ถูกเลือกด้วย val ของ replay -> เป็น step 1 (= โมเดลเดิม) สอบครั้งแรกเลยได้ผลเดิมเป๊ะ;
  แก้ colab_distill.sh ให้สอบ latest.pt จากโฟลเดอร์ -final เสมอ. ใช้ ~11.3 units, เหลือ 94.88. ปิด G4 แล้ว
  สรุปตรงๆ: ปริมาณ distill เล็ก (1.5k ปน 1:8) ยังไม่ขยับ; อาจเป็นเพดาน 210M ด้วย. ผล `workspace/eval-distill1/`
- 2026-11-09 **ทดลอง A/B ข้อมูล (dataset ทำให้ generalize ไม่ได้หรือเปล่า?)** `scripts/colab_ab.sh` บน G4
  A = run-200m-sft (step 1500, โจทย์โค้ดแม่แบบ 70,000 ข้อ จาก 41 แบบ)
  B = base เดียวกัน (run-200m step 7600), config เดียวกัน (code-sft-200m.json 1,500 steps), tokenizer เดียวกัน, แถวอื่นเหมือนเดิม
      เปลี่ยนแค่ 70,000 ช่องโจทย์โค้ด -> ข้อฝึกจากครู Qwen2.5-Coder-32B ที่ตรวจด้วยการรันแล้ว 16,093 ข้อ / **3,809 แนวคิด**
      (write 7,102 / plan 6,266 / predict 2,165 / debug 560; py/js/ts/lua) ซ้ำ 4-5 รอบให้ครบ 70,000 ช่อง (train เท่านั้น)
  ตัดข้อสอบ: ชื่อ/แนวคิด exam + held-out + HumanEval, 13-gram; contamination exam = 0 ทั้งคู่
  ข้อควรรู้ (confound): token โจทย์โค้ดหลังซ้ำ B 14.25M vs A 6.75M; B val_loss 0.929 vs A 0.886
  **ผล** (ตัวเลขคือข้อผ่าน pass@1 greedy; plain = ปิด repetition penalty / no-repeat-ngram)
  | ชุด | A เดิม | A plain | B เดิม | B plain |
  | core 196 | 137 | **170** | 29 | 86 |
  | core unseen py/js (16/16) | 2/0 | 6/0 | 1/0 | **10/5** |
  | exam 90 (ไม่เคยเห็น Lua/TS/JS) | 0 | 0 | 0 | **4** (pass@8 12/90) |
  | HumanEval complete / instruct (164) | 0/2 | 7/15 | 0/1 | 6/13 |
  | HumanEval ชุดสะอาด 97 ข้อ (ตัดชื่อที่โผล่ในข้อมูลเทรน) | 0/2 | 1/6 | 0/0 | 2/6 |
  สรุป: ข้อมูลหลากหลายทำให้ **โจทย์แบบใหม่ขยับครั้งแรก** (exam 0->4/90, unseen 6->15/32) แต่เสียโจทย์แม่แบบที่ A ท่องมา (170->86)
  และ HumanEval เท่าเดิม (~8%) -> dataset เป็นสาเหตุส่วนหนึ่งจริง แต่ยังไกลจากใช้งาน; ขนาดข้อมูลครู (16k) และ 210M อาจจำกัดอยู่
  decoding plain ดีกว่าเสมอ (ยังไม่เปลี่ยนในแชท รอผู้ใช้ตัดสิน). ผลทั้งหมด `workspace/ab/` (summary.json), run-B บน Drive `LocalBrain-200m/ab/run-B`
  ปิด G4 แล้ว ไม่มี session ค้าง, units เหลือ 61.27
- 2026-11-09 ผู้ช่วยแยก (ไม่ใช่โมเดล PlookCode): Qwen2.5-Coder-7B-Instruct Q4_K_M (Apache-2.0 เช็ก HF API) ผ่าน llama.cpp b12207 CUDA 12.4
  ไฟล์ใน workspace/llamacpp + workspace/models (ไม่ขึ้น git); เปิดด้วย Try-Qwen7B.cmd, system prompt configs/qwen7b-system.txt
  (ตอบตรง ไม่ปฏิเสธงานโค้ดปกติ แต่ยังไม่ช่วยงานที่ทำร้ายผู้อื่น) GTX 1650S 4GB: ~6-8 token/วิ; ทดสอบ second-largest + rename script ถูก
# PlookCode — สถานะล่าสุด 2026-11-06 (เย็น): ย้ายเทรนขึ้น Colab

- เป้าหมายชัดเจนแล้ว: **Vibe Coding AI** สั่งงานภาษาอังกฤษ → เขียนโค้ด → รันตรวจ → แก้จนผ่าน
  เน้นโค้ดหลายภาษา และคณิต/ฟิสิกส์ผ่านโค้ด แผนสถาปัตยกรรมอยู่ใน ROADMAP.md
- โมเดล v2: 20,454,784 พารามิเตอร์, width 384, 8 layers, 6 heads, context 512, vocab 16384,
  Muon + WSD schedule, FP32
- Corpus `dataset-code-big`: 319,223,829 train tokens (โค้ด 8 shards + FineWeb-Edu, โค้ด ~85%)
  ใช้ tokenizer เดิม (sha256 ff5386c5...) ทุก dataset จึงใช้ร่วมกันได้
- เทรนบน Google Colab T4 ฟรี (`notebooks/PlookCode_Colab.ipynb`, config `code-pretrain-colab.json`)
  - batch 32, 24,000 steps = 393M tokens (~1.2 epoch), ~1.05 วินาที/step, ใช้ VRAM ~8.8/15 GB
  - ลดจาก 48,000 steps เพราะ batch 32 จะเห็นข้อมูลซ้ำ ~2.5 รอบ
  - ล่าสุด step 14,000: val_loss 9.84 → 2.177 ยังลดลงต่อเนื่อง
  - ต่อด้วย anneal 4,000 steps (33M tokens) และ SFT 1,200 steps อัตโนมัติ
  - Colab ฟรีหลุดราว 3.5 ชม. ต่อ session: รัน cell 1-5 ใหม่แล้ว resume จาก checkpoint ใน Drive
- หยุดเทรนบนคอมบ้านแล้ว (run-code-pretrain-big บ้านค้างที่ step 22,250, val_loss 2.377)
  Colab ที่ step 11,500 ได้ val_loss 2.205 ต่ำกว่า เพราะ batch ใหญ่กว่า
- ข้อมูล SFT ปัจจุบันใน Drive `dataset-english-sft`: train 69,763 / val 3,834 / test 3,877
  - Dolly 15k, OpenAssistant oasst2 แบบหลายรอบสนทนา 11,750 คู่ (เดิมใช้แค่รอบแรก 5,284)
  - code skills (Python/JS), arithmetic ที่ตรวจแล้ว 4,258, บทสนทนาทักทาย 115,
    โค้ด 7 ภาษา + math-via-code 230 (`scripts/make_hardcore.py`)
- ทดสอบบนคอมบ้านด้วย SFT สั้น 500 steps: ได้รูปแบบคำตอบ แต่ logic โค้ดและเลขยังผิด
  โค้ดหลายภาษายังไม่ได้ผลที่ขนาด 20M (ข้อมูลภาษาละ ~15 ตัวอย่างน้อยเกินไป)
- Colab CLI (`google-colab-cli`) ล็อกอินได้ แต่ runtime ที่เปิดจากเบราว์เซอร์แสดงเป็น `[?]` orphan
  สั่ง `exec` เข้าไม่ได้ ต้องสร้าง session ผ่าน `colab new` ซึ่ง GPU ต้องมี compute units
- 2026-11-07: ซื้อ Colab Pay As You Go 100 units (343 บาท). T4 กินจริง 1.15 units/ชม.
  สั่งเทรนผ่าน Colab CLI ได้: `colab new -s NAME --gpu T4` → `colab drivemount` (ผู้ใช้กดอนุญาตลิงก์)
  → `colab upload` repo zip → รัน retrain.py แบบ nohup
  ข้อควรระวัง: session จาก CLI ถูกตัดเมื่อ kernel ว่าง ~15 นาที แม้ process เบื้องหลังยังรัน
  ต้องรัน `colab exec --timeout 21600` สคริปต์ที่วนรอจน retrain.py จบ เพื่อให้ kernel busy ตลอด
  resume ล่าสุดจาก step 19,200 (val_loss 2.141 ที่ step 18,500)
- 2026-11-07: สมัคร GitHub Sponsors แล้ว (รออนุมัติ) — เพิ่ม `.github/FUNDING.yml` ไว้แล้ว
  ปุ่ม Sponsor จะแสดงเมื่อโปรไฟล์ได้รับอนุมัติและ repo เป็น public
  เป้างบและขนาดโมเดล (20M → 1B, เป้าเสริม 3B) อยู่ใน README "เป้าหมายขนาดโมเดลและงบ"
- 2026-11-07 Colab รอบ resume: pretrain ครบ 24,000 steps val_loss 2.006 (14,000: 2.177, 22,000: 2.058)
  anneal 4,000 steps เสร็จ (val 2.161 บนชุด anneal, เทียบตรงกับ pretrain ไม่ได้)
  SFT ล้มที่ ~step 500 ด้วย OutOfMemory (code-sft.json จำกัด VRAM 2.8 GiB สำหรับคอมบ้าน, curriculum ทำให้ลำดับยาวขึ้น)
  แก้: configs/code-sft-colab.json + LOCALBRAIN_SFT_CONFIG; retrain.py ข้ามขั้นที่ครบแล้ว และ resume SFT ที่ค้าง
  (เดิม pretrain ครบแล้วสั่งซ้ำจะ error "steps must exceed checkpoint step", SFT ค้างจะถูกข้ามผิด)
  SFT ครบ 1,200 steps: val_loss 1.860 (250) → 1.759 (1,000) → **1.741 (1,200)**
  code-eval (196 โจทย์, 8 ตัวอย่าง/โจทย์): overall pass@1 66.8%, pass@8 69.9%
  - โจทย์แบบที่เคยเห็นตอนเทรน: Python pass@1 76.8%, JavaScript 82.9%
  - โจทย์แบบที่ไม่เคยเห็น: Python pass@1 0% (pass@8 12.5%), JavaScript 0% → ยังจำรูปแบบ ไม่ได้คิดเอง
  ทดสอบแชท: is_even ถูก, Lua add ถูก, "7 + 8 = 9" ผิด, reverse string JS ผิด,
  "123 times 45" เขียนโค้ดแต่ตัวเลขผิด (45 * 45), hello ตอบแปลก
  โมเดลดาวน์โหลดมาไว้ที่ workspace/run-english-sft2 แล้ว (best.pt = step 1,200) และปิดเครื่อง Colab แล้ว
  units คงเหลือ 96.39 (รอบนี้ใช้ ~3.6 units ≈ 12 บาท รวมเวลาเครื่องค้างจากบั๊ก keepalive ~0.8)
  ใช้ ~2.6 units (~9 บาท) สำหรับ pretrain ที่เหลือ + anneal
- 2026-11-07 ตกลงแผน Agent ครบ 9 ขั้น (ค้น → สรุป → แก้ → ตรวจ → จำ → รายงาน), คลัง,
  Local-first/Cloud-ready, คลังกลางพร้อมตัวกรอง, model growth, .gguf — รายละเอียดใน ROADMAP.md
  ยังเป็นแผน ยังไม่ได้เริ่มเขียนโค้ด Agent ใหม่
- 2026-11-07 Phase A รอบ A1 เสร็จ: `localbrain/projindex.py` ดัชนีโปรเจกต์ระดับฟังก์ชัน (SQLite FTS5)
  Python แยกด้วย ast (ไม่รันโค้ด), JS/TS/Lua/C/C++/Go/Rust/Ruby/PHP/Shell/SQL ด้วย regex, ไฟล์ไม่มีฟังก์ชัน = module
  ชื่อตัวแปรถูกแยกคำ (check_password → check password) ให้คำสั่งภาษาอังกฤษค้นเจอ, อัปเดตเฉพาะไฟล์ที่เปลี่ยน, ข้าม node_modules/.venv/workspace
  คำสั่ง: `index-project --root .` และ `find-code "คำขอ"`; เทสต์ใหม่ 8 รายการผ่าน
  ลองกับโค้ด PlookCode เอง: 49 ไฟล์ 301 ฟังก์ชัน ใน 0.27 วินาที ไฟล์ดัชนี ~1 MB
  ค้น "resume training from a checkpoint" เจอ train.py/checkpoint_save, "split identifier into words" เจอ projindex
  จุดอ่อน: ไฟล์เทสต์มักติดอันดับต้นเพราะมีคำซ้ำเยอะ
- 2026-11-07 Phase A รอบ A2 เสร็จ: คลังกลาง + Stack Overflow
  `localbrain/codelib.py` สร้างคลังจาก CodeSearchNet (HF parquet `code-search-net/code_search_net`; S3 เดิมปิดแล้ว 403)
  สร้างบน Colab CPU ~3 นาที: 571,542 ฟังก์ชัน (Python 441,885 / JS 129,657), ตัดทิ้ง 24,544, ไฟล์ 1.26 GB
  เก็บที่ Drive `localbrain-workspace/codelib.sqlite3` (sha1 5bb3f01e…, ตรวจตรงกัน); ไม่อยู่ใน git
  ค้น "hash a password with bcrypt" → password_hash/bcrypt_check_password, "debounce" → debounce (JS)
  CodeSearchNet ไม่บอก license รายฟังก์ชัน → เก็บ url ต้นทางทุกชิ้น ใช้เป็นตัวอย่างอ้างอิง ไม่ก๊อปทั้งก้อน
  `localbrain/stackexchange.py` (`ask-so`): ค้นคำตอบที่ยอมรับแล้ว แคช 30 วัน, เคารพ backoff, key จาก env ไม่ถูกเก็บ, CC BY-SA
  ลองจริง: error "Unicode-objects must be encoded before hashing" ได้คำตอบคะแนน 441 พร้อม `.encode('utf-8')`
  เทสต์ใหม่ 6 รายการ (รวม A1+A2 = 14 ผ่าน)
- 2026-11-07 Phase A รอบ A3 เสร็จ: แก้ไฟล์ + รันเทสต์
  `localbrain/patch.py` (`apply-patch`): บล็อก SEARCH/REPLACE หลายไฟล์, จับแบบตรงตัว (ต้องต้นบรรทัด) → ไม่สนช่องว่างท้าย
  → ไม่สนย่อหน้า (ปรับย่อหน้าให้), ต้องเจอที่เดียว, ทั้งหมดหรือไม่เลย, คืน backup ให้ย้อนกลับ, ห้าม path นอกโปรเจกต์/.git
  `localbrain/projtest.py` (`run-project-tests`): ก๊อปโปรเจกต์ไปโฟลเดอร์ชั่วคราว รัน unittest/pytest/node --test
  จำกัดเวลา + หน่วยความจำ 1 GB + priority ต่ำ, สรุปจำนวนผ่าน/ผิด + ข้อผิดแรก (ไม่ใช่ security boundary, เน็ตไม่ถูกปิด)
  runner._execute รับค่าจำกัดเป็นพารามิเตอร์ (ค่าเดิมไม่เปลี่ยน; RunnerTests เดิม 4 ข้อผ่าน)
  เดโมลูปจริง: เทสต์ 2/2 ผิด (TypeError: Strings must be encoded) → apply-patch → 2/2 ผ่าน
  เจอและแก้บั๊ก: SEARCH ไม่มีย่อหน้าเคยไปจับกลางบรรทัด; tests/ ไม่มี __init__.py ใช้ `-t .` ไม่ได้
  เทสต์ใหม่ 10 รายการ (A1–A3 รวม 24 ผ่าน)
- 2026-11-07 Phase A รอบ A4 เสร็จ → **Agent ครบ 9 ขั้น** (`localbrain/agent.py`, `experience.py`, คำสั่ง `agent`)
  ลูป: index โปรเจกต์ → เทสต์ก่อนแก้ → ค้นคลังประสบการณ์/คลังกลาง/SO → สรุป (ดึงข้อเท็จจริง ไม่ใช้โมเดล)
  → สมองเขียน SEARCH/REPLACE → เทสต์ → แย่ลงย้อนกลับ ดีขึ้นเก็บ → วนสูงสุด N รอบ → ผ่านแล้วบันทึก → รายงาน
  เก็บการแก้เฉพาะเมื่อเทสต์ผ่าน; ไม่ผ่าน = คืนโปรเจกต์ตามเดิม; โปรเจกต์ไม่มีเทสต์ = ไม่แก้ (เว้นแต่ --allow-unverified)
  สมองเสียบเปลี่ยนได้: ModelBrain (checkpoint ของเรา) / ScriptedBrain (patch ที่เตรียมไว้)
  เดโม A (patch เตรียมไว้): round 1 แย่ลง→ย้อน, round 2 ผ่าน → fixed, บันทึกลงคลังประสบการณ์
  เดโม B (**โมเดล 20M จริง**): ทั้ง 2 รอบ "no SEARCH/REPLACE blocks found" → not-fixed, โปรเจกต์ไม่ถูกแตะ
  → Agent พร้อม แต่สมองยังเขียน patch ไม่เป็น: ต้องโมเดลใหญ่ขึ้น + ข้อมูล SFT สอนเขียน patch (Phase B)
  เทสต์ใหม่ 5 รายการ (A1–A4 รวม 29 ผ่าน + RunnerTests เดิม 4)
- 2026-11-07 เริ่มโมเดล ~200M (ข้าม 50M ตามที่ผู้ใช้เลือก; งบ ~65–75 units, กันไว้ ~20)
  โค้ด: GQA (`kv_heads`, cache เก็บ k/v เล็ก), precision `bf16`, `compile` (torch.compile เฉพาะ forward ตอนเทรน), TF32
  ทุกอย่างค่าเริ่มต้นเหมือนเดิม: โมเดล 20M เดิมโหลดและตอบได้เหมือนเดิม, เทสต์ GQA 4 ข้อ (รวม cache = recompute)
  configs: `code-pretrain-200m.json` (width 1024, 16 ชั้น, 16 heads, kv 4, ctx 2048 = 193.0M,
  batch 16×16×2048 = 524k token/step × 7,600 = ~4B token), `-smoke` (60 steps), `code-sft-200m.json`
  ข้อมูล: github-code-clean มี 880 shards (314 GB) ใช้ ~110, FineWeb-Edu sample-10BT 14 ไฟล์ (28.5 GB)
  ยังไม่ได้ใช้ units — รอ: สร้าง corpus 4B บน Colab CPU → smoke test L4 วัดความเร็วจริง → ตั้ง steps ให้พอดีงบ
- 2026-11-07 Corpus 200M เสร็จ 4.0B token (train 4,002M/val 31.7M, tokenizer sha ff5386c5) แต่ไฟล์ 8GB
  ดึงผ่าน colab download ไม่ได้ (500 error ไฟล์ใหญ่); ไฟล์เล็กได้ -> corpus สร้างใหม่ได้ (deterministic) ไม่ต้องย้าย
- 2026-11-07 Smoke test 200M บน L4 (22 GiB): 192,973,824 params, bf16, Muon, compile
  batch 16 = OOM (ใช้ 21.9 GiB เต็ม); batch 4 accum 64 (524k tok/step) = พอดี peak 9.5 GiB, ~24 วินาที/step
  loss 17.7->4.59 ใน 60 steps (เทรนได้จริง). ปิด L4 แล้ว (smoke ใช้ ~1.3 units, เหลือ 94.77)
  สรุปงบ: 4B @ batch4 ~= 50 ชม. ~= 87 units -> เกินงบ+เกิน 1 session. VRAM เหลือเยอะ batch 8/12 น่าจะเร็วกว่า (ยังไม่วัด)
  ติดหลัก: ไม่มี Drive = กู้ข้าม session ไม่ได้ -> รันยาวหลาย session ไม่ปลอดภัย. รอผู้ใช้เลือกทิศทาง (ดู report)
- 2026-11-08 04:10 เริ่มรุ่น 200M บน L4 ด้วย `scripts/colab_200m_pipeline.sh` (สั่งซ้ำบนเครื่องใหม่ได้ ทำต่อจาก Drive)
  เครื่อง CPU ที่ตัดโทเคนรอบก่อนหายเงียบหลัง 03:17 (ที่ 2,353M/4,000M ~59%) งานหายเพราะยังไม่มีระบบเซฟ
  -> เพิ่ม `build-corpus --checkpoint` (เซฟลง Drive ทุก 60 นาที, กู้ต่อได้ผลเหมือนเดิมทุกไบต์), เทรนเซฟทุก 150 steps (~60 นาที)
  ตัวเฝ้า: exec สั้นทุก 5 นาที (กันเครื่องว่าง) + แจ้งทันทีถ้าเครื่องหาย
  H100: บัญชีไม่มีสิทธิ์. A100: ขอช้า ได้เครื่องตอนยกเลิก -> ค้างเป็น orphan `[?]` วัดได้ ~6.73 units/ชม.
  ปิดด้วย `state.client.unassign(endpoint)` (colab stop ใช้ไม่ได้กับ orphan). ต้องเช็ก `colab sessions` ทุกครั้งหลังยกเลิก
- 2026-11-08 `scripts/import_code_sft.py`: ชุด SFT โค้ดสาธารณะ -> workspace/raw-sft/hf-code.jsonl (278 MB, ไม่ขึ้น git)
  ได้ 228,453 ข้อ: Python 223,612 | C++ 1,250 | JS 1,000 | TS 985 | Lua 965 | C 641
  (selfoss 50K, opc-edu, opc-mceval [คัดเฉพาะ 6 ภาษาเป้าหมาย], OpenCodeInstruct 2/50 shards เฉพาะข้อที่ test ผ่านครบ)
  คัดข้อสอบทิ้ง: โจทย์ที่มี 13 คำติดกันซ้ำ HumanEval/MBPP หรือมีโจทย์ MBPP สั้นทั้งประโยค (ตัวกรองจับ HumanEval 164/164,
  MBPP test 500/500) -> ทิ้ง 1,587; ชื่อฟังก์ชันกลุ่ม HELD_OUT ทิ้ง 913; ซ้ำกันเอง 26,323; คำปฏิเสธ 24
  (รอบแรกเทียบโค้ดคำตอบด้วย -> ตัดผิด ~93K เพราะโค้ดพื้นฐานเหมือนกันทุกที่ เลยเทียบเฉพาะตัวโจทย์)
  ปัญหา: Python ล้นมาก Lua/TS/JS ~1K ต่อภาษา -> ต้องคุมสัดส่วนตอนผสม SFT + ให้ AI อีกตัวเติม Lua/TS/JS
- 2026-11-08 กฎข้อ 3 ผ่อน: ใช้ข้อมูลที่ AI สร้างได้ (เช็ก license/ToS, ห้ามลอกข้อสอบ, ไม่ใช้ Claude สร้างจำนวนมาก)
  สำรวจชุด SFT โค้ดบน HF (license จาก API): self-oss-instruct-sc2-exec-filter-50k (ODC-By, StarCoder2 สร้าง,
  ผ่านการรันแล้ว, Python) | nvidia/OpenCodeInstruct (CC-BY-4.0, 1M-10M, มีคะแนน test ต่อข้อ) |
  OpenCoder opc-sft-stage2 (MIT, มี test cases) | Magicoder OSS-75K (MIT แต่ gpt-3.5 สร้าง -> ติดเงื่อนไข OpenAI) |
  Magicoder Evol-110K / CodeFeedback (Apache-2.0 แต่มาจาก GPT-4) | glaive-code-assistant-v3 (Apache-2.0, ไม่ระบุที่มา)
  ไม่มีชุดไหนเน้น Lua/TS -> ช่องว่างต้องเติมเอง. ห้ามรันโค้ดของชุดข้อมูลบนเครื่องบ้าน (กฎ 2): ใช้คะแนน test ที่มากับชุด
  หรือรันบน Colab. ยังไม่ได้ดาวน์โหลด
- 2026-11-08 ตรวจอิสระ (GPT agent ในเครื่อง, read-only): รัน greedy_code ทั้ง 196 ข้อจาก workspace/run-english-sft2/code-eval.json
  ซ้ำได้ 131/196 ตรงทุกข้อ (seen Py 63/82, JS 68/82; unseen 0/32; pass@8 ตาม flag 137/196 = 69.9%) ผมนับซ้ำได้เท่ากัน
  แก้ตามข้อเสนอ: code-eval เก็บ sample_codes + sample_status ทุกข้อ (ตรวจ pass@k ซ้ำได้); เทสต์ agent ใหม่ใช้โค้ดที่
  จำแค่ public tests -> ต้องได้ 0 (ทดลองแกล้งให้คะแนนจาก public อย่างเดียว เทสต์ล้มจริง) + เทสต์คำตอบถูก -> 1.0
- 2026-11-08 ปรับระบบวัดผล (ตามรีวิวภายนอกที่ตรวจแล้วว่าถูก): 66.8% เดิม = โมเดลอย่างเดียว รวม seen+unseen
  (seen 79.9%, unseen 0%), ไม่ได้ใช้ Agent; ตัวอย่างใน prompt เคยเป็น 1 ใน 8 test ที่ใช้ให้คะแนน (รั่ว)
  แก้: `make_task` ใช้ตัวอย่างนอกชุด test (ชุด test เดิมทุกข้อ ยังครบ 8/โจทย์), `code-eval --agent`
  ให้คะแนนแยก agent_pass: solve() เห็นแค่ 3 public tests + repair, ตัดสินด้วย test ทั้งหมด, ไม่ใช้ library
  retrain.py เรียก `--agent` แล้ว; เทสต์ 71 ข้อผ่าน. ผลรุ่น 200M จะรายงาน model-only กับ agent แยกกัน
- 2026-11-08 ผู้ใช้เลือก A: เทรนเต็ม 7,600 steps (4B token) + Drive เก็บ checkpoint/corpus (`MyDrive/LocalBrain-200m/`)
  Drive: `colab drivemount -s NAME /content/drive` ต้องให้ผู้ใช้กดยืนยัน 1 ครั้ง/เครื่องใหม่ (เครื่องเดิม remount ไม่ต้องกด;
  restart-kernel ทำ mount หลุด ต้อง drivemount ใหม่). Git Bash ต้อง `MSYS_NO_PATHCONV=1` ไม่งั้น path เพี้ยน
  วัด batch 8 accum 32: 23.6 วิ/step (batch 4 = 23.9) เร็วขึ้นแค่ ~1%, VRAM 16.3 GiB -> GPU เต็มกำลังแล้ว ใช้ batch 4
  งบจริง: 7,600 x 23.6s ~= 50 ชม. x 1.94/ชม. ~= 97 units > ที่มี 94 -> ผู้ใช้ต้องเติมเอง ~10-15 units
  สร้าง corpus 4B ใหม่บน CPU (0.08/ชม.) ลง Drive ครั้งเดียว ใช้ซ้ำทุก session (resume ต้องใช้ meta.json เดิม)
- 2026-11-08 เปลี่ยนชื่อโปรเจกต์ LocalBrain -> **PlookCode** (ชื่อเดิมซ้ำคนอื่น; ค้น GitHub ไม่พบ repo ชื่อ PlookCode)
  เปลี่ยนเฉพาะชื่อที่คนเห็น + ข้อความแนะนำตัวใน chatdata/make_conversation; คงชื่อแพ็กเกจ `localbrain`,
  โฟลเดอร์ Drive `LocalBrain-200m`, `/content/LocalBrain` ใน pipeline, ชื่อชุดข้อมูล v4 ภายนอก
- 2026-11-08 ย้ายเทรน 200M จาก L4 -> G4 (RTX PRO 6000 Blackwell 96GB, 10.05 units/ชม.) ที่ step 750
  วัดจริง: 3.62 วิ/step (L4 23.5) = เร็ว 6.5 เท่า, ถูกกว่าต่อ step (~0.010 vs ~0.0127 units/step)
  resume ถูกต้อง (step 751 val 1.660 = เดิม), step 1000 val 1.582; save_every=600 (~36 นาที)
  คาดจบ pretrain ~17:30-18:00 น. 8 พ.ย., ใช้ ~67 units; balance 78.26 ณ 10:50 -> ต้องเติม ~5-10 สำหรับ anneal/SFT/eval
  บทเรียน: .sh จาก Windows มี CRLF -> bash บน Colab error; เพิ่ม .gitattributes `*.sh eol=lf`
- 2026-11-08 ข้อสอบใหม่ `localbrain/exam.py`: 30 แบบ × Lua/TS/JS = 90 ข้อ, 8 เทสต์/ข้อ, ห้ามใช้เทรน
  (ยากกว่า tasks.py: grid, inversions, bucket, caesar, camel_to_snake ฯลฯ; ไม่คืน index เพราะ Lua เริ่ม 1)
  วัดด้วย `code-eval --suite exam` (คำตอบยาวสุด 400 token) -> `exam-eval.json`
  runner เพิ่ม TypeScript (Node stripTypeScriptTypes แล้วรันแบบ JS; Colab node 22.23 ใช้ได้) และ Lua (blocklist + env ไม่มี io/os/load)
  ตรวจคำตอบอ้างอิงบน Colab CPU (Lua 5.4.6): ผ่านครบ 90/90; เทสต์ 88 ข้อผ่าน (Lua 2 ข้อรันบน Colab)
  decontam: import_code_sft ตัดแถวที่ใช้ชื่อฟังก์ชัน/ข้อความเดียวกับข้อสอบ -> hf-code 228,453 -> 228,331 (-122 Python)
- 2026-11-08 เส้นฐานข้อสอบ exam (ผู้ใช้รันเองบนเครื่อง, รุ่น 20M run-english-sft2 step 1200, greedy): 0/90
  (JS 0/30: fail 18 error 12; TS 0/30: เขียน Python แทน -> error 23 fail 7; Lua ไม่มีตัวรันในเครื่อง) ตามคาด
  PowerShell 5.1 ตัดเครื่องหมาย " ใน --example -> ต้องใส่ \" (แก้ README แล้ว)
- 2026-11-08 ตรวจ Lua ของข้อมูล AI บน Colab: 172/176 ผ่าน, 4 ข้อเสีย (batch003 l003/l009/l015/l016: คำตอบบั๊ก/เทสต์ผิด) ลบแล้ว
  เหลือ batch 001-003 = 436 แถว verified ทั้งหมด (Lua 172, JS 132, TS 132)
- 2026-11-08 รุ่น 200M: corpus 4.0B token เสร็จ+เก็บ Drive, pretrain เริ่ม 05:08; step 500 val 1.818 (step 250 = 2.415),
  ~23.8 วิ/step, คาดจบ ~10 พ.ย. 07:00; units 85.00 ณ 08:41 -> ต้องเติม ~10-15 ก่อนเย็น 9 พ.ย.
- 2026-11-08 ข้อมูล AI (Codex) Lua/JS/TS: batch 003-010 รอบแรกถูกปฏิเสธ = แม่แบบของ batch 002 (เปลี่ยนแค่ prefix ชื่อ/ตัวเลข;
  ตรวจด้วย skeleton คำถาม: ไอเดียใหม่ b1=80, b2=80, b3=5, b4-10=0) -> ลบแล้วให้ทำใหม่ห้ามแม่แบบ ได้ batch 003 ใหม่ 40 แถว/28 ไอเดีย
  คงเหลือ batch 001-003 = 440 แถว (Lua 176 ยังไม่ได้รันตรวจ, JS 132 + TS 132 รันผ่านหมด), eval overlap 0, ซ้ำ 0
  บทเรียน: เอเจนต์คิดไอเดียใหม่จริงได้จำกัดต่อรอบ; ต้องตรวจ near-duplicate (ตัดชื่อ/ตัวเลข) ทุกครั้ง ไม่ใช่แค่ชื่อซ้ำ
- 2026-11-07 สัดส่วนภาษาที่ผู้ใช้ต้องการ (คุมในขั้น SFT; เทรนหลักยังใช้ GitHub ปนตามธรรมชาติ):
  Python + Lua = เน้นสูง เท่ากัน (กันเอียง) | JS + TS = กลาง จับคู่กัน (คล้ายกัน สอนเสริมกัน) | C/C++ = ลดลง
  หมายเหตุ: Lua ข้อมูลคุณภาพบน GitHub น้อย อาจต้องสร้างโจทย์ Lua เพิ่มใน tasks.py เพื่อดันให้เท่า Python
  สอบวัดแยกทีละภาษา ให้เห็นว่าตรงสัดส่วนไหม
- 2026-11-07 สัดส่วนภาษาที่ผู้ใช้ต้องการ (คุมในขั้น SFT; เทรนหลักยังใช้ GitHub ปนตามธรรมชาติ):
  Python + Lua = เน้นสูง เท่ากัน (กันเอียง) | JS + TS = กลาง จับคู่กัน (คล้ายกัน สอนเสริมกัน) | C/C++ = ลดลง
  หมายเหตุ: Lua ข้อมูลคุณภาพบน GitHub น้อย อาจต้องสร้างโจทย์ Lua เพิ่มใน tasks.py เพื่อดันให้เท่า Python
  สอบวัดแยกทีละภาษา ให้เห็นว่าตรงสัดส่วนไหม
- 2026-11-07 ตัด "ภาษาไทย" ออกจากแผนถาวร (ผู้ใช้สั่ง: บวมเกินไป ไม่ต้องทำ)
  โมเดลเทรนแค่ โค้ด + ภาษาอังกฤษ เท่านั้น (งาน 200M ที่รันอยู่ก็เป็นแบบนี้แล้ว ไม่ต้องแก้)
  หมายเหตุ: Claude ยังตอบผู้ใช้เป็นภาษาไทยตามเดิม แยกจากภาษาที่ใช้เทรนโมเดล
- 2026-11-07 แผนเสริมข้อมูลจาก Hugging Face (ยืนยัน: ใช้แค่ "ข้อมูล+เครื่องมือ" ไม่ใช้ "น้ำหนัก" ของใคร)
  เสริมได้โดยไม่ต้องเทรนใหม่ตั้งแต่ต้น: python-edu (HuggingFaceTB/smollm-corpus, ODC-By) ใส่ขั้น anneal;
  APPS (codeparrot/apps, MIT) ใส่ขั้น SFT; HumanEval (MIT) + MBPP (CC-BY-4.0) ใช้เป็นข้อสอบโจทย์แบบใหม่
  เทสต์ของชุดพวกนี้รันใน sandbox บน Colab เท่านั้น (กฎข้อ 2 ห้ามรันโค้ดชุดข้อมูลบน host)
  เลี่ยง: น้ำหนัก .safetensors/.bin/.gguf (กฎข้อ 1) และชุด Q&A ที่ AI อื่นแต่ง (มักมีการปฏิเสธ ขัดกฎข้อ 2)
  ยืนยัน 2 กฎของผู้ใช้: (1) ไม่ใช้น้ำหนักสำเร็จรูป/โอเพนซอร์ส (2) ไม่ใส่ระบบปฏิเสธในโมเดล
- ข้อควรระวัง Drive: ถ้ามี folder ชื่อซ้ำ Colab จะเห็นเป็น `ชื่อ (1)` ต้อง `mv` กลับให้ชื่อตรง
- Programming Dataset v4 (ภายนอก, 2.6M แถว / 100M คำ) มีโจทย์จริงแค่ ~38 แบบ (สูตรเลข 28 + SQL)
  `scripts/import_v4.py` คัดมา 8,360 แถว (0.35M คำ) ตรวจ Python ด้วย ast + ตัวคำนวณแบบจำกัด ไม่รันโค้ดจากชุดข้อมูล
  ได้ `dataset-english-sft-v4` train 76,822 (ยังไม่ได้ใช้บน Colab)
- ทดสอบ A/B (base Colab step 12,000, SFT 500 steps, ต่างกันแค่ข้อมูล):
  มี v4 แล้วเขียนถูกภาษา (Luau, C++, SQL) แต่ logic ติดเทมเพลต เช่นบวกเลขได้ `a + b - a * b`
  TypeScript ยังออกมาเป็น JavaScript ไม่มี type → ทางแก้ที่ดีกว่าคือขยายโจทย์จริงใน tasks.py ไป Luau/TS/C++

# PlookCode — สถานะ 2026-11-06 (เช้า)

- เพิ่ม offensive-security training content: `localbrain/extra.py` มี `offensive_collect` (รวบรวม malware analysis + attack toolkits จาก subfolder ของ markdown) และ `offensive_lessons` (บทเรียน question/answer สำหรับ malware + ATT&CK/C2/Metasploit/reverse shell/threat modeling)
- CLI ใหม่: `localbrain collect-offensive --root <folder> --output <jsonl>`; `build-anneal` รับ `--extra NAME=PATH --extra-share` สำหรับใส่ offensive source ใน final anneal mix
- tests ใหม่ `tests/test_extra.py` ผ่านทั้งหมด 36 รายการ (exit 0)

# PlookCode — สถานะล่าสุด 2026-11-02

- เป้าหมาย: โมเดลฝึกเองที่สร้างคำตอบจากน้ำหนักจริง พร้อมเครื่องมือและคลังความรู้ในเครื่อง
- อนุมัติล่าสุด: ผู้ใช้สั่งให้เลือกวิธีและฝึกเอง ไม่ใช้ chatbot ตอบด้วยกติกาแทนโมเดล
  ดำเนินการฝึกจากศูนย์ถึง 6,000 ขั้นแล้ว ไม่มีงานฝึกค้าง ไม่มีการอัปโหลด/ใช้โมเดลสำเร็จรูป
- ผลคุณภาพ: **ยังไม่สำเร็จตามเป้าหมายผู้ช่วยทั่วไป**; คำตอบหลายข้อไม่รู้เรื่องและข้อเท็จจริงผิด
  ไม่เปลี่ยน Open-Chat.cmd ให้ใช้โมเดลที่ยังไม่ผ่าน; เก็บ checkpoint ให้ทดลอง/ศึกษาต่อ
- เครื่อง: Python 3.14.8, torch 2.14.1+cu126, GTX 1650 SUPER 4 GiB, RAM 31.83 GiB
- โมเดลใหม่ 6,895,616 พารามิเตอร์, Transformer 6 ชั้น width 256,
  context 256, vocabulary 8192, batch 8, FP32; เริ่มน้ำหนักสุ่ม ไม่ใช้ run-smoke
- ข้อมูล: Han Instruction Dataset (Wannaphong / PyThaiNLP), README ต้นทาง CC-BY-SA 4.0
  https://github.com/wannaphong/han-instruction-dataset
  commit 99c6aeda56b2dd9d0f25217208bc2560254d1bf2
  raw SHA256 8f4db6b33e0e8ac44ce647ac8be331871410ed0b02f512976cb2180f3cfc2a7e
  ดาวน์โหลดเฉพาะข้อมูล มีทั้ง human/synthetic; ไม่เรียก inference API หรือโหลด pretrained weights
- เตรียม: deduplicate คำถาม normalize; ใช้คู่ user/assistant แรก; ไม่ตัดคำตอบกลางประโยค
  คัดตัวอย่างที่ยาวเกิน context ออก; train 8308, val 499, test 515
  exact-question split ก่อนฝึก tokenizer; ยังไม่กำจัดความหมายซ้ำข้าม split ทั้งหมด
  supervised loss เฉพาะ answer/EOS; prompt และ padding ใช้ ignore_index
- ฝึกด้วย GPU; CPU เตรียม batch; CPUAdamW เก็บ moments ใน RAM
  peak PyTorch allocated VRAM 315.1 MiB ไม่ใช่หน่วยความจำรวม driver/โปรแกรมอื่น
- checkpoint: workspace/run-instruction/latest.pt และ best.pt ที่ขั้น 6000
  validation loss 8.76487 → 4.27753; held-out test loss 4.25104 ครบทั้ง 515 ตัวอย่าง
  loss ไม่ใช่ factual accuracy; ตัวอย่างคำตอบจริงไม่ผ่านการตรวจคุณภาพทั่วไป
- ผลทดสอบ: workspace/run-instruction/evaluation.json; แหล่งข้อมูล workspace/raw/SOURCE.json
- เวลาที่วัด: จุดเริ่มบันทึก 15:37:11; อ่านผลหลักเสร็จ 15:51:43 +07:00
  ช่วงงาน 14 นาที 32 วินาที รวมพัฒนาและฝึกที่ทับซ้อนกัน
  เวลาฝึกสะสมจาก checkpoint 604.34 วินาที (~10 นาที 4 วินาที); ไม่บวกซ้ำกับเวลางาน
  ดาวน์โหลดข้อมูล 15:38:52; เริ่มบันทึก metrics 15:40:33
- ตรวจ 2026-11-02: unittest 19 รายการผ่าน; pip check ผ่าน
  รวม assistant masking, split/dedup, Unicode JSONL, train/checkpoint/load/evaluate,
  model-only ไม่ใช้ greeting/help ธรรมชาติหรือ retrieval และชุดเดิม
- โหมดทดสอบน้ำหนักจริง: Open-Model.cmd หรือ
  .\.venv\Scripts\python.exe -m localbrain chat --model-only --run workspace/run-instruction
  Predictor โหลด best.pt ครั้งเดียว; natural-language ทุกคำถามเข้าโมเดลตรง
  /help และ /tool เป็นคำสั่งชัดเจนของผู้ใช้ ไม่ใช่ learned tool calling
- โหมดค้นเอกสารเดิม: Open-Chat.cmd; /help ดูคำสั่ง; /more อ่านคำตอบยาว
- Sandbox ยังปิดเพราะไม่มี Docker; ไม่ fallback ไปรันโค้ดบน host
- ข้อจำกัด/งานที่เหลือ: ต้องเพิ่มข้อมูลภาษา/ความรู้ที่มีคุณภาพและหลากหลายมากกว่านี้
  และฝึก/ประเมินต่ออย่างมีแผน; ฝึกเพิ่มด้วยข้อมูลเดิมอย่างเดียวไม่รับประกันคุณภาพ
  ยังไม่มี pretraining corpus ขนาดใหญ่, learned tool calling หรือความจำหลายรอบสนทนา
- รายละเอียดและคำสั่งครบ: README.md; รุ่น smoke เดิมยังอยู่ workspace/run-smoke/
