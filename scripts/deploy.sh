#!/bin/bash
set -e

# Deployment script for Claude Manager Service
# Usage: ./scripts/deploy.sh [environment] [options]

# Default values
ENVIRONMENT="${1:-development}"
NAMESPACE="claude-manager"
KUBECTL_CMD="kubectl"
DRY_RUN="${DRY_RUN:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300s}"
VERSION="${VERSION:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Claude Manager Service Deployment Script

Usage: $0 [environment] [options]

Environments:
    development    Deploy to development environment (default)
    staging        Deploy to staging environment
    production     Deploy to production environment

Options:
    --namespace=NAME     Kubernetes namespace (default: claude-manager)
    --version=VERSION    Application version to deploy (default: latest)
    --dry-run           Show what would be deployed without applying
    --wait-timeout=TIME Wait timeout for deployments (default: 300s)
    --help              Show this help message

Environment Variables:
    ENVIRONMENT         Target environment
    NAMESPACE          Kubernetes namespace
    KUBECTL_CMD        kubectl command to use
    DRY_RUN           Whether to perform dry run (true/false)
    WAIT_TIMEOUT      Deployment wait timeout
    VERSION           Application version

Examples:
    $0                                    # Deploy to development
    $0 production --version=1.0.0        # Deploy specific version to production
    $0 staging --dry-run                  # Dry run deployment to staging

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace=*)
            NAMESPACE="${1#*=}"
            shift
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --wait-timeout=*)
            WAIT_TIMEOUT="${1#*=}"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option $1"
            exit 1
            ;;
        *)
            ENVIRONMENT="$1"
            shift
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|staging|production)
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        log_info "Valid environments: development, staging, production"
        exit 1
        ;;
esac

# Set kubectl options
KUBECTL_OPTS="--namespace=$NAMESPACE"
if [ "$DRY_RUN" = "true" ]; then
    KUBECTL_OPTS="$KUBECTL_OPTS --dry-run=client"
    log_warning "Running in dry-run mode - no changes will be applied"
fi

# Pre-deployment checks
log_info "Starting deployment process..."
log_info "Environment: $ENVIRONMENT"
log_info "Namespace: $NAMESPACE"
log_info "Version: $VERSION"
log_info "Dry Run: $DRY_RUN"

# Check if kubectl is available
if ! command -v $KUBECTL_CMD &> /dev/null; then
    log_error "kubectl command not found: $KUBECTL_CMD"
    exit 1
fi

# Check if we can connect to the cluster
if ! $KUBECTL_CMD cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "k8s" ]; then
    log_error "k8s directory not found. Are you in the project root?"
    exit 1
fi

# Function to apply Kubernetes manifests
apply_manifests() {
    local manifest_dir="k8s"
    local env_overlay="k8s/overlays/$ENVIRONMENT"
    
    log_info "Applying Kubernetes manifests..."
    
    # Apply base manifests
    if [ -f "$manifest_dir/namespace.yaml" ]; then
        log_info "Creating namespace..."
        $KUBECTL_CMD apply -f "$manifest_dir/namespace.yaml" $KUBECTL_OPTS
    fi
    
    # Apply secrets (these should be created separately in production)
    if [ "$ENVIRONMENT" != "production" ] && [ -f "$env_overlay/secrets.yaml" ]; then
        log_info "Applying secrets..."
        $KUBECTL_CMD apply -f "$env_overlay/secrets.yaml" $KUBECTL_OPTS
    fi
    
    # Apply configmaps
    if [ -f "$manifest_dir/configmap.yaml" ]; then
        log_info "Applying configmaps..."
        $KUBECTL_CMD apply -f "$manifest_dir/configmap.yaml" $KUBECTL_OPTS
    fi
    
    # Apply environment-specific configmaps
    if [ -f "$env_overlay/configmap.yaml" ]; then
        log_info "Applying environment-specific configmaps..."
        $KUBECTL_CMD apply -f "$env_overlay/configmap.yaml" $KUBECTL_OPTS
    fi
    
    # Apply RBAC
    if [ -f "$manifest_dir/rbac.yaml" ]; then
        log_info "Applying RBAC..."
        $KUBECTL_CMD apply -f "$manifest_dir/rbac.yaml" $KUBECTL_OPTS
    fi
    
    # Apply database manifests
    if [ -f "$manifest_dir/postgres.yaml" ]; then
        log_info "Applying PostgreSQL..."
        $KUBECTL_CMD apply -f "$manifest_dir/postgres.yaml" $KUBECTL_OPTS
    fi
    
    # Apply Redis manifests
    if [ -f "$manifest_dir/redis.yaml" ]; then
        log_info "Applying Redis..."
        $KUBECTL_CMD apply -f "$manifest_dir/redis.yaml" $KUBECTL_OPTS
    fi
    
    # Update deployment with correct image version
    if [ -f "$manifest_dir/deployment.yaml" ]; then
        log_info "Applying main deployment..."
        sed "s/:latest-production/:${VERSION}-production/g" "$manifest_dir/deployment.yaml" | \
        $KUBECTL_CMD apply -f - $KUBECTL_OPTS
    fi
    
    # Apply monitoring manifests
    if [ -f "$manifest_dir/monitoring.yaml" ]; then
        log_info "Applying monitoring..."
        $KUBECTL_CMD apply -f "$manifest_dir/monitoring.yaml" $KUBECTL_OPTS
    fi
    
    # Apply ingress
    if [ -f "$env_overlay/ingress.yaml" ]; then
        log_info "Applying ingress..."
        $KUBECTL_CMD apply -f "$env_overlay/ingress.yaml" $KUBECTL_OPTS
    fi
}

# Function to wait for deployment
wait_for_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping deployment wait (dry-run mode)"
        return 0
    fi
    
    log_info "Waiting for deployment to complete..."
    
    if $KUBECTL_CMD rollout status deployment/claude-manager $KUBECTL_OPTS --timeout=$WAIT_TIMEOUT; then
        log_success "Deployment completed successfully"
    else
        log_error "Deployment failed or timed out"
        return 1
    fi
}

# Function to perform health checks
health_check() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping health check (dry-run mode)"
        return 0
    fi
    
    log_info "Performing health check..."
    
    # Get service endpoint
    local service_url
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        service_url=$(minikube service claude-manager-service --url -n $NAMESPACE 2>/dev/null || echo "")
    else
        # For other Kubernetes environments, use port-forward
        log_info "Setting up port-forward for health check..."
        $KUBECTL_CMD port-forward service/claude-manager-service 8080:5000 $KUBECTL_OPTS &
        local pf_pid=$!
        sleep 5
        service_url="http://localhost:8080"
    fi
    
    if [ -n "$service_url" ]; then
        if curl -sf "$service_url/health" > /dev/null; then
            log_success "Health check passed"
            if [ -n "$pf_pid" ]; then
                kill $pf_pid 2>/dev/null || true
            fi
        else
            log_error "Health check failed"
            if [ -n "$pf_pid" ]; then
                kill $pf_pid 2>/dev/null || true
            fi
            return 1
        fi
    else
        log_warning "Could not determine service URL for health check"
    fi
}

# Function to show deployment status
show_status() {
    if [ "$DRY_RUN" = "true" ]; then
        return 0
    fi
    
    log_info "Deployment Status:"
    echo "===================="
    
    $KUBECTL_CMD get pods $KUBECTL_OPTS -l app=claude-manager-service
    echo
    $KUBECTL_CMD get services $KUBECTL_OPTS -l app=claude-manager-service
    echo
    $KUBECTL_CMD get ingress $KUBECTL_OPTS 2>/dev/null || true
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    
    if [ "$DRY_RUN" != "true" ]; then
        # Rollback deployment if it exists
        if $KUBECTL_CMD get deployment claude-manager $KUBECTL_OPTS &> /dev/null; then
            log_info "Rolling back deployment..."
            $KUBECTL_CMD rollout undo deployment/claude-manager $KUBECTL_OPTS || true
        fi
    fi
}

# Set up error handling
trap cleanup_on_failure ERR

# Main deployment flow
apply_manifests

if [ "$DRY_RUN" != "true" ]; then
    wait_for_deployment
    health_check
    show_status
    
    log_success "Deployment to $ENVIRONMENT completed successfully!"
    
    # Show access information
    log_info "Access Information:"
    echo "=================="
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        echo "Dashboard URL: $(minikube service claude-manager-service --url -n $NAMESPACE)"
    else
        echo "Use port-forward to access the application:"
        echo "kubectl port-forward service/claude-manager-service 8080:5000 -n $NAMESPACE"
        echo "Then visit: http://localhost:8080"
    fi
else
    log_success "Dry-run completed successfully!"
fi