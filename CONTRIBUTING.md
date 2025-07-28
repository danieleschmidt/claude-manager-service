# Contributing to Claude Code Manager

Thank you for your interest in contributing! Please review our guidelines below.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally: `git clone <your-fork-url>`
3. **Read** our [Development Guide](docs/DEVELOPMENT.md) for setup instructions
4. **Create** a feature branch: `git checkout -b feature/your-feature`

## Development Process

1. **Write Tests**: Follow TDD approach - write tests first
2. **Implement Feature**: Make your changes following existing patterns
3. **Run Tests**: Ensure all tests pass with `make test`
4. **Quality Checks**: Run `make quality` for linting and type checking
5. **Commit**: Use conventional commit format: `feat: add new feature`

## Pull Request Guidelines

* **Clear Title**: Use descriptive, concise titles
* **Link Issues**: Reference related GitHub issues
* **Test Coverage**: Ensure adequate test coverage
* **Documentation**: Update relevant documentation
* **Review**: Request review from maintainers

## Code Standards

* Follow [PEP 8](https://pep8.org/) Python style guidelines
* Add type hints to all public functions
* Include docstrings for modules, classes, and functions
* Run pre-commit hooks before committing

## Security

Report security vulnerabilities privately to security@terragon.ai.
See [SECURITY.md](SECURITY.md) for full security policy.

## Questions?

* Check our [GitHub Discussions](https://github.com/danieleschmidt/claude-code-manager/discussions)
* Review existing [GitHub Issues](https://github.com/danieleschmidt/claude-code-manager/issues)
* Read our [Development Guide](docs/DEVELOPMENT.md)