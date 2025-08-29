# Project Structure

## Core Files (Root)
- `main.py` - Unified entry point (653 lines)
- `config.yaml` - Centralized configuration
- `docker-compose.yml` - Multi-service orchestration
- `Dockerfile` - Container specification
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies

## Source Code
- `src/` - Core microservices
  - `analyzers/` - 4 specialized AI analyzers
  - `cli/` - CLI component system
  - `models/` - Pydantic data models
  - `enhancers/` - Data enrichment (legacy)
  - `scrapers/` - Data collection (legacy)
  - `utils/` - Shared utilities

## Testing & Quality
- `tests/` - Comprehensive test suite
- `scripts/` - Legacy CLI and tools
- `monitoring/` - System monitoring

## Data & Results
- `data/` - Database and raw data
- `results/` - Analysis outputs
  - `json_outputs/` - Organized JSON results
- `enhanced_data/` - Enriched datasets
- `analysis_results/` - ML analysis outputs
- `langchain_results/` - AI analysis results

## Documentation
- `docs/` - Project documentation
  - `legacy/` - Legacy documentation
- `AI_Engineer_Journal/` - Development journal

## Temporary & Archive
- `temp/` - Temporary files and tests
- `archive/` - Legacy configuration files
- `logs/` - Runtime logs

## Configuration
- `.env.example` - Environment template
- `.gitignore` - Git exclusions
- `Makefile` - Build automation
