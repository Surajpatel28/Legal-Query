# Contributing to LegalQuery

Thank you for considering contributing to LegalQuery! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Include screenshots if applicable

### Submitting Changes

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear messages (`git commit -m "Add: feature description"`)
6. Push to your branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

### Testing

- Test your changes locally before submitting
- Ensure the app runs without errors
- Verify chat functionality works correctly

### Areas for Contribution

- Improving prompt templates
- Adding new features
- Enhancing UI/UX
- Bug fixes
- Documentation improvements
- Performance optimizations

## Development Setup

```bash
git clone https://github.com/yourusername/LegalQuery.git
cd LegalQuery
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
streamlit run app.py
```

## Questions?

Feel free to open an issue for any questions or clarifications.

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the issue, not the person
- Help create a positive environment

Thank you for contributing!
