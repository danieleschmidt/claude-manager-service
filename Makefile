# Makefile for Claude Code Manager

# Variables
DOCKER_COMPOSE = docker-compose
DOCKER_COMPOSE_PROD = docker-compose -f docker-compose.yml -f docker-compose.prod.yml
PYTHON = python3
PIP = pip3
PROJECT_NAME = claude-code-manager
VERSION = 0.1.0

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: help install install-dev clean test test-unit test-integration test-e2e test-security test-performance lint format type-check security-scan build run run-prod stop logs shell backup restore deploy health check-deps update-deps docs serve-docs clean-cache clean-docker clean-all

# Default target
help: ## Show this help message
	@echo "$(BLUE)Claude Code Manager - Makefile Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# Cleaning
clean: ## Clean temporary files and cache
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/

clean-cache: ## Clean all cache directories
	@echo "$(BLUE)Cleaning cache directories...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	rm -rf .coverage*
	rm -rf htmlcov/

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-cache clean-docker ## Clean everything

# Testing
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest tests/e2e/ -v --slow

test-security: ## Run security tests
	@echo "$(BLUE)Running security tests...$(NC)"
	pytest tests/security/ -v

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	pytest tests/performance/ -v --slow

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov-report=html --cov-report=term-missing

# Code Quality
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 src tests
	pylint src

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black .
	isort .

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check .
	isort --check-only .

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src

security-scan: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

quality: format lint type-check security-scan ## Run all code quality checks

# Docker operations
build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

build-prod: ## Build production Docker images
	@echo "$(BLUE)Building production Docker images...$(NC)"
	$(DOCKER_COMPOSE_PROD) build

run: ## Run development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) up -d

run-prod: ## Run production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE_PROD) up -d

stop: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down

stop-prod: ## Stop production services
	@echo "$(BLUE)Stopping production services...$(NC)"
	$(DOCKER_COMPOSE_PROD) down

restart: stop run ## Restart development environment

restart-prod: stop-prod run-prod ## Restart production environment

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Show logs from application service only
	$(DOCKER_COMPOSE) logs -f claude-manager

shell: ## Open shell in application container
	$(DOCKER_COMPOSE) exec claude-manager bash

# Database operations
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(PYTHON) -c "from src.database_migration_utility import migrate; migrate()"

db-reset: ## Reset database
	@echo "$(YELLOW)Resetting database...$(NC)"
	rm -f data/tasks.db
	$(PYTHON) -c "from src.services.database_service import DatabaseService; DatabaseService().initialize_database()"

# Backup and restore
backup: ## Create backup of data
	@echo "$(BLUE)Creating backup...$(NC)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U claude_user claude_manager > backups/postgres-$(shell date +%Y%m%d_%H%M%S).sql
	tar -czf backups/data-$(shell date +%Y%m%d_%H%M%S).tar.gz data/

restore: ## Restore from backup (requires BACKUP_FILE variable)
	@echo "$(BLUE)Restoring from backup...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)Error: BACKUP_FILE variable is required$(NC)"; exit 1; fi
	$(DOCKER_COMPOSE) exec postgres psql -U claude_user -d claude_manager < $(BACKUP_FILE)

# Deployment
deploy: ## Deploy to production
	@echo "$(BLUE)Deploying to production...$(NC)"
	git pull origin main
	$(DOCKER_COMPOSE_PROD) pull
	$(DOCKER_COMPOSE_PROD) up -d --force-recreate

deploy-staging: ## Deploy to staging
	@echo "$(BLUE)Deploying to staging...$(NC)"
	$(DOCKER_COMPOSE) up -d --force-recreate

# Health checks
health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	$(DOCKER_COMPOSE) ps
	curl -f http://localhost:5000/health || echo "$(RED)Application health check failed$(NC)"
	curl -f http://localhost:9090/-/healthy || echo "$(RED)Prometheus health check failed$(NC)"
	curl -f http://localhost:3000/api/health || echo "$(RED)Grafana health check failed$(NC)"

# Monitoring
metrics: ## Show current metrics
	@echo "$(BLUE)Current system metrics:$(NC)"
	$(PYTHON) -c "from src.performance_monitor import PerformanceMonitor; pm = PerformanceMonitor(); print(pm.get_metrics())"

status: ## Show status of all components
	@echo "$(BLUE)System Status:$(NC)"
	$(DOCKER_COMPOSE) ps
	@echo "\n$(BLUE)Application Status:$(NC)"
	curl -s http://localhost:5000/status | python -m json.tool || echo "Application not responding"

# Documentation
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdir -p docs/api
	$(PYTHON) -m pydoc -w src
	mv *.html docs/api/

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs && $(PYTHON) -m http.server 8080

# Dependency management
check-deps: ## Check for dependency updates
	@echo "$(BLUE)Checking for dependency updates...$(NC)"
	pip list --outdated

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# Development helpers
dev-setup: install-dev ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	pre-commit install
	mkdir -p logs data temp backups
	cp .env.example .env
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "$(YELLOW)Don't forget to edit .env with your configuration$(NC)"

quick-test: ## Quick test run (unit tests only)
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/unit/ -x --tb=short

watch-tests: ## Watch for changes and run tests
	@echo "$(BLUE)Watching for changes and running tests...$(NC)"
	pytest-watch -- tests/unit/

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	pytest tests/performance/ --benchmark-only

# Release management
version: ## Show current version
	@echo "$(BLUE)Current version: $(VERSION)$(NC)"

tag: ## Create git tag for current version
	@echo "$(BLUE)Creating git tag v$(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

release: quality test build ## Prepare for release (quality checks, tests, build)
	@echo "$(GREEN)Release preparation complete for version $(VERSION)$(NC)"

# Utility commands
ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

top: ## Show container resource usage
	docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

network: ## Show network information
	docker network ls
	docker network inspect claude-manager_claude-manager-network 2>/dev/null || echo "Network not found"

volumes: ## Show volume information
	docker volume ls | grep claude-manager

# Emergency procedures
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP - Stopping all services immediately$(NC)"
	docker stop $$(docker ps -q) || true

emergency-backup: ## Emergency backup
	@echo "$(RED)EMERGENCY BACKUP$(NC)"
	mkdir -p emergency-backups
	cp -r data/ emergency-backups/data-emergency-$(shell date +%Y%m%d_%H%M%S)/
	docker cp claude-manager-postgres:/var/lib/postgresql/data emergency-backups/postgres-emergency-$(shell date +%Y%m%d_%H%M%S)/