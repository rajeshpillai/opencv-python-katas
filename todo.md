# OpenCV Interactive Playground — TODO

## Phase 1: Backend
- [/] Create project folder structure
- [ ] `backend/requirements.txt`
- [ ] `backend/main.py` — FastAPI app entry
- [ ] `backend/models/db.py` — SQLite init + schema
- [ ] `backend/models/schemas.py` — Pydantic schemas
- [ ] `backend/executor/sandbox.py` — subprocess-based safe execution
- [ ] `backend/executor/sandbox-runner.py` — isolated runner script
- [ ] `backend/routers/execute.py` — POST /api/execute
- [ ] `backend/routers/katas.py` — GET /api/katas, GET /api/katas/{slug}
- [ ] `backend/routers/auth.py` — POST /api/auth/register, POST /api/auth/login

## Phase 2: Kata Data
- [ ] `backend/data/katas/00-image-loading.json`
- [ ] `backend/data/katas/01-color-spaces.json`
- [ ] `backend/data/katas/02-pixel-access.json`

## Phase 3: Frontend
- [ ] Scaffold SolidJS + Vite project in `frontend/`
- [ ] `frontend/src/api/client.ts` — API client
- [ ] `frontend/src/components/kata-sidebar.tsx`
- [ ] `frontend/src/components/kata-header.tsx`
- [ ] `frontend/src/components/code-editor.tsx` — Monaco Editor
- [ ] `frontend/src/components/output-panel.tsx`
- [ ] `frontend/src/components/demo-panel.tsx`
- [ ] `frontend/src/pages/kata-page.tsx`
- [ ] `frontend/src/App.tsx` — routing + layout
- [ ] `frontend/src/index.css` — dark theme

## Phase 4: Verification
- [ ] Run backend, test /api/execute with curl
- [ ] Run frontend dev server, verify UI renders
- [ ] End-to-end: write code in editor → run → see image output
