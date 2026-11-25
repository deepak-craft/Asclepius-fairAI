# Asclepius FairAI -- Production Deployment Guide

**Version**: `Asclepius FairAI v1.0`  
**Target Environment**: Streamlit Community Cloud, Containerized Docker, or Cloud Linux VPS (AWS EC2 / GCP Compute Engine / Azure VM).

---

## 1. Prerequisites & System Requirements

- **Runtime**: Python 3.10, 3.11, or 3.12 (64-bit)
- **Memory**: Minimum 2 GB RAM (4 GB recommended for large ensemble training and SHAP evaluations)
- **Disk Storage**: 500 MB free space (for dependencies, report generation, and SQLite audit records)
- **Supported OS**: Ubuntu 22.04 LTS+, Debian 12+, RHEL 9+, Windows 10/11, macOS 12+

---

## 2. Canonical Local Run Command

The application must always be launched from the **project root directory**:

```bash
# Clean start canonical command
python -m streamlit run src/app.py
```

Expected local endpoint:
**`http://localhost:8501`**

---

## 3. Recommended Cloud Deployment Options

### Option A: Streamlit Community Cloud (Recommended for Demos & Peer Review)

1. **Push Repository to GitHub**:
   Ensure your repository contains `requirements.txt`, `.streamlit/config.toml`, and the `src/` directory.
2. **Connect to Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New App"**.
   - Select your GitHub repository, branch (`main`), and set the main file path to:
     ```
     src/app.py
     ```
3. **Deploy**:
   - Streamlit Cloud will automatically build from `requirements.txt` and launch the app with HTTPS support.

> *Note on Ephemeral Storage*: On Streamlit Community Cloud, local files generated in `outputs/` and `outputs/history.db` are persisted during active container sessions but will reset upon container redeployment. For production persistent audit retention across container restarts, mount a persistent volume or configure cloud database backend adapters.

---

### Option B: Docker Container Deployment

Create a `Dockerfile` in the project root:

```dockerfile
# Base image with Python 3.11 slim
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure outputs directory exists
RUN mkdir -p outputs/reports

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run the container:

```bash
# Build Docker image
docker build -t asclepius-fair-ai:v1.0 .

# Run Docker container on port 8501
docker run -d -p 8501:8501 --name asclepius-app asclepius-fair-ai:v1.0
```

---

### Option C: Linux Server (Ubuntu 22.04 LTS / Systemd Service)

1. **System Preparation**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git
   git clone https://github.com/your-org/asclepius-fair-ai.git /opt/asclepius-fair-ai
   cd /opt/asclepius-fair-ai
   ```

2. **Virtual Environment & Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Systemd Service** (`/etc/systemd/system/asclepius.service`):
   ```ini
   [Unit]
   Description=Asclepius FairAI Streamlit Service
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/opt/asclepius-fair-ai
   ExecStart=/opt/asclepius-fair-ai/venv/bin/streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start and Enable Service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start asclepius
   sudo systemctl enable asclepius
   ```

---

## 4. Runtime File Storage & Directory Creation

The application dynamically validates and initializes required directories upon first write:
- `outputs/`: Automatically created by `src/core/history.py` if absent.
- `outputs/reports/`: Automatically created by `src/core/report_generator.py` if absent.
- `outputs/history.db`: SQLite database schema initialized automatically on first startup.

---

## 5. Security & Privacy Safeguards

- **Zero-Storage of Raw Data**: Patient-level dataset rows are held in volatile execution memory during analysis and never written to database tables or server disks.
- **Strict Parameterized Queries**: All history operations use bound parameters (`?`) to neutralize SQL injection vulnerabilities.
- **Safe Path Resolution**: All asset loaders and database connectors use dynamic `os.path.abspath(os.path.join(os.path.dirname(__file__), ...))` to prevent path traversal and ensure portable execution across Linux, Windows, and macOS.
- **Zero Committed Secrets**: The repository contains no API keys, tokens, or sensitive credentials.

---

## 6. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Recommended Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | App executed from inside a subdirectory | Always launch from the root directory via `python -m streamlit run src/app.py`. |
| `Port 8501 already in use` | Another instance is active | Specify an alternate port: `python -m streamlit run src/app.py --server.port 8502`. |
| `SHAP TreeExplainer Memory Spike` | Large dataset evaluation | SHAP evaluation subsamples to 500 records by default for safety. Adjust `max_samples` parameter if needed. |
| `PDF Report missing charts` | Headless Linux display error | The application explicitly enforces `matplotlib.use("Agg")` to guarantee headless chart rendering without X11. |
