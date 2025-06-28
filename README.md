# Startup Research Assistant

An AI-powered startup research assistant that automatically enriches company data from multiple sources and provides intelligent querying capabilities.

## Features

- **Concurrent Data Enrichment**: Automatically researches startups using multiple APIs simultaneously
- **Comprehensive Data Sources**: 
  - People Data Labs (PDL) - Company information, funding, revenue
  - NewsAPI - Recent news and mentions
  - Tavily Search - Web search for additional context (optional)
  - Firecrawl - Deep web crawling for detailed content (optional)
  - PDL Founders - Founder information extraction
- **Database Storage**: SQLite database with SQLModel/SQLAlchemy
- **REST API**: Full REST API for data retrieval and analytics
- **Chat Interface**: Natural language querying via OpenAI integration
- **Analytics**: Industry analysis, funding rankings, revenue statistics
- **Simple Logging**: Clean, informative logging throughout

## Quick Start

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- API keys for external services

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd startup_researcher
```

2. Install dependencies with Poetry:
```bash
poetry install --no-root
```

3. Set up environment variables:
```bash
cp .env.example .env
# Then edit .env to add your API keys and settings
```

4. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

**Note**: The application uses SQLite as its database. A `startups.db` file will be created automatically in the project directory.

## API Documentation

Once the server is running, visit:

```
http://localhost:8000/docs
```

to see and interact with the full OpenAPI documentation for all endpoints.

## Usage Example: Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_question": "What are the top 5 funded startups in the database?"}'
```

## Architecture

### Data Flow

1. **Input**: List of startup domains via API
2. **Enrichment**: Concurrent API calls to multiple data sources
3. **Storage**: Raw data stored in SQLite database
4. **Querying**: REST API and chat interface for data access
5. **Analytics**: Aggregated insights across all startups

### Database Schema

- `Startup`: Core startup information (domain, name, processing status)
- `StartupData`: Flexible storage for raw data from different sources

### API Integration

- **People Data Labs**: Company enrichment and founder data
- **NewsAPI**: Recent news and mentions
- **Tavily**: Web search for additional context (optional)
- **Firecrawl**: Deep web crawling (optional)
- **OpenAI**: Natural language processing for chat
