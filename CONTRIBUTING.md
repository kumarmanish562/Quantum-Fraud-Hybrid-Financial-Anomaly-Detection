# Contributing to Quantum Fraud Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/Quantum-Fraud-Detection.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add: your feature description"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Code Style

### Python
- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to functions and classes
- Maximum line length: 100 characters

### JavaScript/React
- Use ES6+ syntax
- Follow Airbnb style guide
- Use functional components with hooks
- Add JSDoc comments for complex functions

## Testing

Before submitting a PR, ensure all tests pass:

```bash
# Backend tests
cd backend
python test_api.py

# Frontend tests
cd frontend
npm test
```

## Commit Messages

Use clear and descriptive commit messages:

- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Refactor: what was refactored`
- `Docs: documentation changes`
- `Test: test additions or changes`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Request review from maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing! 🎉
