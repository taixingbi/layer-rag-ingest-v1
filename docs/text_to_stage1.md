```bash
python3 app/plain_text_chunks.py data_save/resume.txt data/chunks_resume_stage1.json
```

`plain_text_chunks.py` does not call the chat API; each chunk has `synthetic_questions: []`. After **`prepare_payloads.py`**, run **`app/synthetic_questions.py`** on **`points_*.json`** to fill questions (see README).

```bash
python3 app/plain_text_chunks.py data_raw/resume.txt data/chunks_resume.json
python3 app/plain_text_chunks.py data_raw/qa.txt data/chunks_qa.json
python3 app/plain_text_chunks.py data_raw/profile.txt data/chunks_profile.json
```
