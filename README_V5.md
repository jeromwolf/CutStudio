# CutStudio v5.0 - Clean Architecture Edition

> 🎬 **AI-Powered Video Editor for Educational Content**
> 교육 콘텐츠에 특화된 차세대 AI 비디오 편집 플랫폼

## 🎯 Project Vision

CutStudio v5.0은 처음부터 다시 설계된 클린 아키텍처 기반의 비디오 편집 플랫폼입니다.
교육 영상 제작자를 위한 **단순하고 강력한** 도구를 목표로 합니다.

### Core Principles
- **Clean Architecture**: 명확한 책임 분리
- **Test-Driven Development**: 안정성 우선
- **Minimal Dependencies**: 필수 기능만
- **Performance First**: 최적화된 처리
- **User-Centric Design**: 직관적 인터페이스

## 🏗️ Architecture

```
CutStudio/
├── src/                    # Source code
│   ├── core/              # Business logic (framework-agnostic)
│   │   ├── entities/      # Domain models
│   │   ├── usecases/      # Application business rules
│   │   └── interfaces/    # Port definitions
│   ├── api/               # External interfaces
│   │   ├── video/         # Video processing APIs
│   │   ├── audio/         # Audio processing APIs
│   │   └── ai/            # AI service integrations
│   ├── ui/                # User interface
│   │   ├── web/           # Web application (Streamlit/FastAPI)
│   │   ├── components/    # Reusable UI components
│   │   └── static/        # Static assets
│   └── utils/             # Utilities and helpers
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── config/               # Configuration files
├── docs/                 # Documentation
└── scripts/              # Automation scripts
```

## 🚀 Features (MVP)

### Phase 1: Core Features (v5.0)
- [ ] Video upload/download (MP4, AVI, MOV)
- [ ] Basic editing (cut, trim, merge)
- [ ] Speaker detection (single unified module)
- [ ] Speech-to-text (Whisper)
- [ ] AI summarization (Gemini)

### Phase 2: Enhanced Features (v5.1)
- [ ] Timeline editor
- [ ] Multi-track support
- [ ] Real-time preview
- [ ] Export presets
- [ ] Batch processing

### Phase 3: Advanced Features (v5.2)
- [ ] Collaborative editing
- [ ] Cloud storage integration
- [ ] Plugin system
- [ ] Mobile app
- [ ] API service

## 🛠️ Tech Stack

### Core Technologies
- **Language**: Python 3.11
- **Web Framework**: FastAPI + Streamlit
- **Video Processing**: FFmpeg (direct)
- **Audio Processing**: Librosa
- **AI/ML**: OpenAI Whisper, Google Gemini
- **Database**: SQLite (local) / PostgreSQL (production)
- **Cache**: Redis
- **Queue**: Celery + RabbitMQ

### Development Tools
- **Testing**: Pytest + Coverage
- **Linting**: Ruff + Black
- **Type Checking**: MyPy
- **Documentation**: Sphinx + MkDocs
- **CI/CD**: GitHub Actions
- **Containerization**: Docker

## 📋 Development Roadmap

### Week 1: Foundation (Current)
- [x] Clean up legacy code
- [x] Set up new project structure
- [ ] Development environment setup
- [ ] Core entity definitions
- [ ] Basic use case implementations

### Week 2: MVP Development
- [ ] Video upload/processing module
- [ ] Speaker detection integration
- [ ] Speech recognition implementation
- [ ] Basic UI with Streamlit
- [ ] Unit tests for core modules

### Week 3: Integration & Testing
- [ ] API endpoints
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] Docker containerization

### Week 4: Release Preparation
- [ ] E2E testing
- [ ] Security audit
- [ ] Performance benchmarking
- [ ] Deployment scripts
- [ ] v5.0 release

## 🔧 Installation

### Prerequisites
```bash
# System requirements
- Python 3.11+
- FFmpeg 6.0+
- 8GB RAM minimum
- 10GB free disk space
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/cutstudio-v5.git
cd cutstudio-v5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start application
python -m src.main
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test type
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with verbose output
pytest -v

# Run in watch mode
pytest-watch
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| 10min video processing | < 3 min | TBD |
| Memory usage | < 1GB | TBD |
| Startup time | < 5s | TBD |
| API response time | < 200ms | TBD |
| Test coverage | > 80% | TBD |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Implement feature
5. Ensure all tests pass
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper team
- Google Gemini team
- FFmpeg community
- All contributors

## 📞 Contact

- **Project Lead**: Kelly
- **Email**: contact@cutstudio.ai
- **Discord**: [Join our community](https://discord.gg/cutstudio)
- **Issues**: [GitHub Issues](https://github.com/yourusername/cutstudio-v5/issues)

---

**Built with ❤️ for educators worldwide**