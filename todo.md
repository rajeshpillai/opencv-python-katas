# OpenCV Interactive Playground — TODO

## Phase 1: Backend
- [x] Create project folder structure
- [x] `backend/requirements.txt`
- [x] `backend/main.py` — FastAPI app entry
- [x] `backend/models/db.py` — SQLite init + schema
- [x] `backend/models/schemas.py` — Pydantic schemas
- [x] `backend/executor/sandbox.py` — subprocess-based safe execution
- [x] `backend/executor/sandbox-runner.py` — isolated runner script
- [x] `backend/routers/execute.py` — POST /api/execute
- [x] `backend/routers/katas.py` — GET /api/katas, GET /api/katas/{slug}
- [x] `backend/routers/auth.py` — POST /api/auth/register, POST /api/auth/login

## Phase 2: Kata Data
- [x] `backend/data/katas/00-image-loading.json`
- [x] `backend/data/katas/01-color-spaces.json`
- [x] `backend/data/katas/02-pixel-access.json`

## Phase 3: Frontend
- [x] Scaffold SolidJS + Vite project in `frontend/`
- [x] `frontend/src/api/client.ts` — API client
- [x] `frontend/src/components/kata-sidebar.tsx`
- [x] `frontend/src/components/kata-header.tsx`
- [x] `frontend/src/components/code-editor.tsx` — Monaco Editor
- [x] `frontend/src/components/output-panel.tsx`
- [x] `frontend/src/components/demo-panel.tsx`
- [x] `frontend/src/pages/kata-page.tsx`
- [x] `frontend/src/App.tsx` — routing + layout
- [x] `frontend/src/index.css` + `components.css` — Tailwind v4 dark theme, class-based

## Phase 4: Verification
- [x] Run backend, test /api/execute with curl ✅ returns base64 image
- [x] GET /api/katas returns all 3 katas ✅
- [x] README.md with setup + run instructions ✅
- [x] Initial git commit + pushed to GitHub ✅
- [ ] End-to-end: write code in editor → run → see image output (manual)

## Next Steps
- [ ] Add more katas (drop JSON files in `backend/data/katas/`)
- [ ] Wire auth into frontend (login/register UI)
- [ ] Interactive demo sliders in `demo-panel.tsx`
- [ ] Start frontend dev server and verify full UI flow
