# Deployment Guide

Complete guide to deploying the Physical AI & Humanoid Robotics textbook with RAG chatbot.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Website)                    │
│  - Docusaurus static site                               │
│  - 6 chapters (markdown → HTML)                         │
│  - Responsive UI                                         │
│  - Deployed to: GitHub Pages / Netlify / Vercel        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ HTTP requests to /api/query
                   ▼
┌─────────────────────────────────────────────────────────┐
│                  Backend API (FastAPI)                   │
│  - RAG query endpoint                                   │
│  - Health check endpoint                                 │
│  - Rate limiting (10 req/min)                           │
│  - Deployed to: Railway / Render / Fly.io              │
└──────┬────────────────────┬─────────────────────────────┘
       │                    │
       │                    │
       ▼                    ▼
┌─────────────┐      ┌──────────────────┐
│   Qdrant    │      │  Neon PostgreSQL │
│ (Vectors)   │      │   (Metadata)     │
│ Free: 1GB   │      │ Free: 0.5GB      │
└─────────────┘      └──────────────────┘
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Git

## Part 1: Backend Setup

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd "Physical AI & Humanoid Robotics — Essentials"
```

### Step 2: Set Up Cloud Services

Follow `backend/CLOUD_SETUP.md` to create:

1. **Qdrant Community Cloud** account
   - Create collection: `textbook_chunks`
   - Vector size: 384, Distance: Cosine
   - Get cluster URL and API key

2. **Neon PostgreSQL** account
   - Create project: `physical-ai-textbook`
   - Get connection string

### Step 3: Configure Environment

```bash
cd backend
cp .env.example .env
# Edit .env with your credentials:
# - QDRANT_URL
# - QDRANT_API_KEY
# - DATABASE_URL
```

### Step 4: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 5: Initialize Database

```bash
python scripts/setup_db.py
```

Expected output:
```
✅ Table created successfully!
📊 Schema:
  - chunk_id UUID NOT NULL
  - chapter_id INTEGER NOT NULL
  ...
```

### Step 6: Download Embedding Model

```bash
python scripts/download_model.py
```

Expected output:
```
✅ Model downloaded and cached successfully!
   Embedding dimension: 384
   Model size: ~80MB
```

### Step 7: Index Chapters

```bash
python scripts/index_chapters.py
```

Expected output:
```
📚 Physical AI Textbook Indexing
✅ Embedding model loaded
✅ Qdrant collection ready

📖 Processing: chapter-1-introduction-to-physical-ai.md
   ✂️  Chunking content...
   📊 Created 15 chunks
   🧠 Generating embeddings...
   ✅ Generated 15 embeddings
   💾 Inserting into Qdrant...
   💾 Inserting into Neon...
   ✅ Indexed 15/15 chunks

[... repeat for all 6 chapters ...]

✨ Indexing Complete!
   Total chapters: 6
   Total chunks: ~90
   Avg chunks/chapter: 15.0
```

### Step 8: Start Backend

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000

**Test endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Physical AI?", "top_k": 5}'
```

Expected response:
```json
{
  "answer": "Based on the textbook content, here's what I found:\n1. From Chapter 1 - Introduction: Physical AI is artificial intelligence deployed in robotic systems...",
  "sources": [
    {
      "chunk_id": "uuid",
      "chapter_id": 1,
      "section_id": "1.1",
      "section_title": "What is Physical AI?",
      "preview_text": "Physical AI is artificial intelligence deployed in robotic systems that interact directly with the physical world...",
      "relevance_score": 0.89
    }
  ],
  "query_time_ms": 145
}
```

## Part 2: Frontend Setup

### Step 1: Install Dependencies

```bash
cd website
npm install
```

### Step 2: Configure API URL

```bash
cp .env.example .env
# Edit .env:
# DOCUSAURUS_API_URL=http://localhost:8000  # For local testing
# DOCUSAURUS_API_URL=https://your-backend.railway.app  # For production
```

### Step 3: Start Development Server

```bash
npm start
```

Visit: http://localhost:3000

**Verify:**
- Homepage displays with 6 chapter cards
- Click "Get Started" → navigates to Chapter 1
- Sidebar navigation works
- All chapters render with proper formatting

### Step 4: Build for Production

**Note**: Due to special characters in the directory name (`&` and `—`), you need to rename the project directory first:

```bash
cd ../..
mv "Physical AI & Humanoid Robotics — Essentials" "Physical-AI-Humanoid-Robotics-Essentials"
cd Physical-AI-Humanoid-Robotics-Essentials/website
```

Now build:

```bash
npm run build
```

Expected output:
```
[SUCCESS] Generated static files in "build".
Use `npm run serve` command to test your build locally.
```

Test production build:

```bash
npm run serve
```

## Part 3: Deployment

### Option A: Deploy to GitHub Pages

1. **Push to GitHub**:

```bash
git remote add origin https://github.com/maryamarif24/hackathon-04.git
git push -u origin master
```

2. **Configure GitHub Pages**:
   - Go to Settings → Pages
   - Source: GitHub Actions
   - Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - name: Install dependencies
        run: cd website && npm ci
      - name: Build website
        run: cd website && npm run build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./website/build
```

3. **Update docusaurus.config.ts**:

```typescript
url: 'https://yourusername.github.io',
baseUrl: '/physical-ai-textbook/',
organizationName: 'yourusername',
projectName: 'physical-ai-textbook',
```

### Option B: Deploy Backend to Railway

1. **Install Railway CLI**:

```bash
npm i -g @railway/cli
railway login
```

2. **Initialize Railway Project**:

```bash
cd backend
railway init
```

3. **Set Environment Variables**:

```bash
railway variables set QDRANT_URL=<your-qdrant-url>
railway variables set QDRANT_API_KEY=<your-api-key>
railway variables set DATABASE_URL=<your-neon-url>
railway variables set MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
railway variables set TRANSFORMERS_CACHE=./models_cache
railway variables set ALLOWED_ORIGINS=https://yourusername.github.io
railway variables set RATE_LIMIT_PER_MINUTE=10
railway variables set LOG_LEVEL=INFO
```

4. **Deploy**:

```bash
railway up
```

5. **Get Public URL**:

```bash
railway domain
# Example: physical-ai-api.railway.app
```

6. **Update Frontend .env**:

```
DOCUSAURUS_API_URL=https://physical-ai-api.railway.app
```

## Verification Checklist

### Backend:
- [ ] Health check returns `"status": "healthy"`
- [ ] Qdrant shows `"connected"`
- [ ] PostgreSQL shows `"connected"`
- [ ] Embedding model shows `"loaded"`
- [ ] Query endpoint returns results with citations
- [ ] Rate limiting works (11th request in 1 min returns 429)

### Frontend:
- [ ] Homepage loads with all 6 chapters
- [ ] Navigation works (sidebar, prev/next buttons)
- [ ] All chapters render with proper formatting
- [ ] Code blocks have syntax highlighting
- [ ] Responsive on mobile (320px), tablet (768px), desktop (1024px+)
- [ ] Build completes without errors

### Integration:
- [ ] Frontend can call backend API (CORS configured)
- [ ] Query returns relevant results
- [ ] Source citations link to correct chapters

## Troubleshooting

### Backend won't start

```bash
# Check Python version
python --version  # Should be 3.11+

# Check dependencies
pip list | grep sentence-transformers

# Check .env file
cat .env | grep QDRANT
```

### Indexing fails

```bash
# Verify Qdrant connection
python -c "from app.services.qdrant_service import qdrant_service; print(qdrant_service.health_check())"

# Verify Neon connection
python -c "from app.services.neon_service import neon_service; print(neon_service.health_check())"

# Check chapter files exist
ls -la ../website/docs/chapter-*.md
```

### Frontend build fails

```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Check for syntax errors
npm run typecheck
```

## Performance Optimization

### Backend:
- Use Railway's Pro plan for better performance ($5/month)
- Enable Qdrant HNSW indexing for faster search
- Add caching layer (Redis) for frequent queries

### Frontend:
- Enable Docusaurus image optimization
- Use CDN (Cloudflare) for static assets
- Compress images with TinyPNG

## Monitoring

### Backend Metrics:
- Query latency: Target <200ms p95
- Error rate: Target <1%
- Rate limit hits: Monitor for abuse

### Frontend Metrics:
- Lighthouse score: Target >90
- Core Web Vitals: LCP <2.5s, FID <100ms, CLS <0.1

## Security

- **Never commit .env files** (already in .gitignore)
- **Rotate API keys** every 90 days
- **Monitor rate limits** to prevent abuse
- **Use HTTPS** for all deployments
- **Validate input** on backend (already implemented with Pydantic)

## Cost Breakdown (Free Tier)

- **Qdrant Community Cloud**: Free (1GB, unlimited requests)
- **Neon PostgreSQL**: Free (0.5GB storage, 100h compute/month)
- **GitHub Pages**: Free (unlimited bandwidth)
- **Railway**: Free tier available (500h/month)

**Total**: $0/month (within free tiers)

## Next Steps

1. Add chatbot UI to frontend (Phase 5)
2. Implement text selection feature (Phase 5)
3. Add analytics (Google Analytics, Plausible)
4. Create admin dashboard for monitoring
5. Set up CI/CD for automated deployments

---

**Support**: For issues, open a ticket on GitHub or consult the documentation in `specs/textbook-generation/`
