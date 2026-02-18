# OpenCV Interactive Playground

A web-based interactive platform for learning OpenCV from first principles. Write Python/OpenCV code, run it safely in a sandbox, and see the visual output instantly. Progress through structured katas from beginner to advanced.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, SQLite |
| Execution | Sandboxed subprocess (safe, isolated) |
| Frontend | SolidJS, Vite, Tailwind CSS v4 |
| Editor | Monaco Editor (VS Code engine) |

---

## Project Structure

```
python-opencv-katas/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── requirements.txt
│   ├── data/
│   │   └── katas/               # Kata JSON files (kata-as-data)
│   │       ├── 00-image-loading.json
│   │       ├── 01-color-spaces.json
│   │       └── 02-pixel-access.json
│   ├── executor/
│   │   ├── sandbox.py           # Subprocess orchestrator
│   │   └── sandbox-runner.py    # Isolated execution script
│   ├── models/
│   │   ├── db.py                # SQLite init + kata seeding
│   │   └── schemas.py           # Pydantic request/response models
│   └── routers/
│       ├── katas.py             # GET /api/katas, GET /api/katas/{slug}
│       ├── execute.py           # POST /api/execute
│       └── auth.py              # POST /api/auth/register, /login
├── frontend/
│   ├── index.html
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx              # Root layout + routing
│       ├── index.css            # Tailwind v4 theme tokens
│       ├── components.css       # All component CSS classes
│       ├── api/client.ts        # Typed API client
│       ├── components/
│       │   ├── kata-sidebar.tsx
│       │   ├── kata-header.tsx
│       │   ├── code-editor.tsx  # Monaco Editor wrapper
│       │   ├── demo-panel.tsx
│       │   └── output-panel.tsx
│       └── pages/
│           └── kata-page.tsx
├── todo.md
└── SYSTEM_PROMPT.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd python-opencv-katas
```

### 2. Backend setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Frontend setup

```bash
cd frontend
npm install
cd ..
```

---

## Running

### Start the backend

```bash
# From project root, with venv activated
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

### Start the frontend

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`.

> The Vite dev server proxies `/api` requests to `http://localhost:8000` automatically.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/katas` | List all katas (id, slug, title, level, concepts) |
| `GET` | `/api/katas/{slug}` | Full kata detail (description, tips, starter code) |
| `POST` | `/api/execute` | Run code in sandbox, returns image + logs + error |
| `POST` | `/api/auth/register` | Register with email + password, returns JWT |
| `POST` | `/api/auth/login` | Login, returns JWT |

### Execute endpoint

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import cv2\nimport numpy as np\nimg = np.zeros((200,300,3), np.uint8)\nimg[:] = (0,128,255)\ncv2.imshow(\"result\", img)"
  }'
```

Response:
```json
{
  "image_b64": "<base64-encoded PNG>",
  "logs": "",
  "error": ""
}
```

---

## Sandbox Safety

User code runs in an **isolated subprocess** with:

- ✅ Only `import cv2` and `import numpy as np` allowed
- ✅ `cv2.imshow()` intercepted — image captured as PNG, not displayed
- ✅ 10-second execution timeout
- ✅ No filesystem access beyond temp
- ✅ No network access
- ❌ All other imports blocked with a friendly error message

---

## Adding New Katas

Create a JSON file in `backend/data/katas/` following the naming convention `NN-slug-name.json`:

```json
{
  "slug": "03-drawing-primitives",
  "title": "Drawing Primitives",
  "level": "beginner",
  "concepts": ["cv2.line", "cv2.rectangle", "cv2.circle"],
  "description": "Learn to draw shapes on images...",
  "prerequisites": ["00-image-loading"],
  "tips": [
    "Coordinates are always (x, y) — column first, row second.",
    "Thickness -1 fills the shape."
  ],
  "starter_code": "import cv2\nimport numpy as np\n\nimg = np.zeros((400, 400, 3), np.uint8)\n# Draw here\ncv2.imshow('result', img)\n",
  "demo_controls": []
}
```

Restart the backend — katas are seeded from JSON files on startup.

---

## Kata Progression

| Level | Topics |
|---|---|
| **Beginner** | Image loading, color spaces, pixel access, resizing, drawing |
| **Intermediate** | Thresholding, blurring, edge detection, morphology, contours |
| **Advanced** | Feature detection, video processing, object tracking, pipelines |

---

## Development Notes

- **No login required** to use the playground (anonymous mode — state is in-browser only)
- **Login** enables progress tracking and saving code versions
- CSS: all styling via classes in `components.css` — no inline styles
- Kata content lives in JSON files, not in code — easy to add/edit without touching Python
