#!/bin/bash
# Container Security Scanning Script for claude-code-manager
# Comprehensive security scanning for Docker containers

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-claude-code-manager}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

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

# Check if required tools are installed
check_tools() {
    log_info "Checking required security scanning tools..."
    
    local tools=("docker" "trivy" "grype" "syft")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install missing tools:"
        for tool in "${missing_tools[@]}"; do
            case $tool in
                "trivy")
                    echo "  curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
                "grype")
                    echo "  curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
                "syft")
                    echo "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
                "docker")
                    echo "  Install Docker from https://docs.docker.com/get-docker/"
                    ;;
            esac
        done
        return 1
    fi
    
    log_success "All required tools are available"
}

# Build the container image if it doesn't exist
build_image() {
    log_info "Checking if image ${FULL_IMAGE} exists..."
    
    if ! docker image inspect "${FULL_IMAGE}" &> /dev/null; then
        log_info "Building container image ${FULL_IMAGE}..."
        cd "${PROJECT_ROOT}"
        docker build -t "${FULL_IMAGE}" .
        log_success "Container image built successfully"
    else
        log_info "Image ${FULL_IMAGE} already exists"
    fi
}

# Run Trivy vulnerability scanning
run_trivy_scan() {
    log_info "Running Trivy vulnerability scan..."
    
    local output_dir="${PROJECT_ROOT}/security-reports"
    mkdir -p "${output_dir}"
    
    # Comprehensive Trivy scan with multiple output formats
    log_info "Scanning for vulnerabilities, secrets, and misconfigurations..."
    
    # JSON format for processing
    trivy image \
        --format json \
        --output "${output_dir}/trivy-report.json" \
        --scanners vuln,secret,config \
        --severity HIGH,CRITICAL \
        "${FULL_IMAGE}"
    
    # Table format for human reading
    trivy image \
        --format table \
        --output "${output_dir}/trivy-report.txt" \
        --scanners vuln,secret,config \
        --severity HIGH,CRITICAL \
        "${FULL_IMAGE}"
    
    # SARIF format for GitHub integration
    trivy image \
        --format sarif \
        --output "${output_dir}/trivy-results.sarif" \
        --scanners vuln,secret,config \
        "${FULL_IMAGE}"
    
    log_success "Trivy scan completed. Reports saved to ${output_dir}/"
}

# Run Grype vulnerability scanning
run_grype_scan() {
    log_info "Running Grype vulnerability scan..."
    
    local output_dir="${PROJECT_ROOT}/security-reports"
    mkdir -p "${output_dir}"
    
    # Grype scan with multiple output formats
    grype "${FULL_IMAGE}" \
        --output json \
        --file "${output_dir}/grype-report.json"
    
    grype "${FULL_IMAGE}" \
        --output table \
        --file "${output_dir}/grype-report.txt"
    
    log_success "Grype scan completed. Reports saved to ${output_dir}/"
}

# Generate SBOM using Syft
generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    local output_dir="${PROJECT_ROOT}/security-reports"
    mkdir -p "${output_dir}"
    
    # Generate SBOM in multiple formats
    syft "${FULL_IMAGE}" \
        --output spdx-json \
        --file "${output_dir}/sbom-spdx.json"
    
    syft "${FULL_IMAGE}" \
        --output cyclonedx-json \
        --file "${output_dir}/sbom-cyclonedx.json"
    
    syft "${FULL_IMAGE}" \
        --output table \
        --file "${output_dir}/sbom-table.txt"
    
    log_success "SBOM generated. Files saved to ${output_dir}/"
}

# Run Docker security best practices check
run_docker_security_check() {
    log_info "Running Docker security best practices check..."
    
    local output_dir="${PROJECT_ROOT}/security-reports"
    mkdir -p "${output_dir}"
    
    # Check for common Docker security issues
    {
        echo "# Docker Security Best Practices Check"
        echo "Generated on: $(date)"
        echo ""
        
        echo "## Image Information"
        docker image inspect "${FULL_IMAGE}" --format '{{json .}}' | jq '{
            Id: .Id,
            Created: .Created,
            Size: .Size,
            Architecture: .Architecture,
            Os: .Os,
            RootFS: .RootFS,
            Config: {
                User: .Config.User,
                WorkingDir: .Config.WorkingDir,
                Env: .Config.Env,
                Cmd: .Config.Cmd,
                Entrypoint: .Config.Entrypoint
            }
        }'
        
        echo ""
        echo "## Security Recommendations"
        
        # Check if running as root
        local user
        user=$(docker image inspect "${FULL_IMAGE}" --format '{{.Config.User}}')
        if [[ -z "$user" || "$user" == "root" || "$user" == "0" ]]; then
            echo "   WARNING: Container runs as root user"
            echo "   Recommendation: Use a non-root user in Dockerfile"
        else
            echo " Container runs as non-root user: $user"
        fi
        
        # Check for health check
        local healthcheck
        healthcheck=$(docker image inspect "${FULL_IMAGE}" --format '{{.Config.Healthcheck}}')
        if [[ "$healthcheck" == "<nil>" ]]; then
            echo "   WARNING: No health check configured"
            echo "   Recommendation: Add HEALTHCHECK instruction to Dockerfile"
        else
            echo " Health check configured"
        fi
        
        # Check exposed ports
        local ports
        ports=$(docker image inspect "${FULL_IMAGE}" --format '{{.Config.ExposedPorts}}')
        if [[ "$ports" != "map[]" ]]; then
            echo "9  Exposed ports: $ports"
            echo "   Recommendation: Ensure only necessary ports are exposed"
        fi
        
    } > "${output_dir}/docker-security-check.txt"
    
    log_success "Docker security check completed. Report saved to ${output_dir}/docker-security-check.txt"
}

# Generate security summary report
generate_summary_report() {
    log_info "Generating security summary report..."
    
    local output_dir="${PROJECT_ROOT}/security-reports"
    local summary_file="${output_dir}/security-summary.md"
    
    cat > "${summary_file}" << EOF
# Container Security Scan Summary

**Image:** ${FULL_IMAGE}
**Scan Date:** $(date)
**Scan Host:** $(hostname)

## Scan Results

### Vulnerability Scanning
- **Trivy Scan**: $([ -f "${output_dir}/trivy-report.json" ] && echo " Completed" || echo "L Failed")
- **Grype Scan**: $([ -f "${output_dir}/grype-report.json" ] && echo " Completed" || echo "L Failed")

### Security Analysis
- **SBOM Generation**: $([ -f "${output_dir}/sbom-spdx.json" ] && echo " Completed" || echo "L Failed")
- **Docker Security Check**: $([ -f "${output_dir}/docker-security-check.txt" ] && echo " Completed" || echo "L Failed")

## Critical Findings

EOF

    # Extract critical vulnerabilities from Trivy report if available
    if [[ -f "${output_dir}/trivy-report.json" ]]; then
        local critical_count
        critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${output_dir}/trivy-report.json" 2>/dev/null || echo "0")
        local high_count
        high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "${output_dir}/trivy-report.json" 2>/dev/null || echo "0")
        
        echo "### Trivy Vulnerability Counts" >> "${summary_file}"
        echo "- **Critical**: ${critical_count}" >> "${summary_file}"
        echo "- **High**: ${high_count}" >> "${summary_file}"
        echo "" >> "${summary_file}"
    fi
    
    cat >> "${summary_file}" << EOF
## Recommendations

1. Review all HIGH and CRITICAL vulnerabilities in the detailed reports
2. Update base images and dependencies to latest secure versions
3. Implement security best practices identified in Docker security check
4. Schedule regular security scans as part of CI/CD pipeline

## Report Files

- \`trivy-report.json\` - Trivy vulnerability scan (JSON)
- \`trivy-report.txt\` - Trivy vulnerability scan (Human readable)
- \`trivy-results.sarif\` - Trivy scan results (SARIF format for GitHub)
- \`grype-report.json\` - Grype vulnerability scan (JSON)
- \`grype-report.txt\` - Grype vulnerability scan (Human readable)
- \`sbom-spdx.json\` - Software Bill of Materials (SPDX format)
- \`sbom-cyclonedx.json\` - Software Bill of Materials (CycloneDX format)
- \`sbom-table.txt\` - Software Bill of Materials (Table format)
- \`docker-security-check.txt\` - Docker security best practices check

---
*Generated by container-scan.sh*
EOF

    log_success "Security summary report generated: ${summary_file}"
}

# Main execution
main() {
    log_info "Starting comprehensive container security scan for ${FULL_IMAGE}"
    
    # Check tools availability
    if ! check_tools; then
        exit 1
    fi
    
    # Build image if needed
    build_image
    
    # Create reports directory
    mkdir -p "${PROJECT_ROOT}/security-reports"
    
    # Run all security scans
    run_trivy_scan
    run_grype_scan
    generate_sbom
    run_docker_security_check
    
    # Generate summary
    generate_summary_report
    
    log_success "Container security scan completed successfully!"
    log_info "Review the reports in: ${PROJECT_ROOT}/security-reports/"
    
    # Display summary of critical findings
    local reports_dir="${PROJECT_ROOT}/security-reports"
    if [[ -f "${reports_dir}/trivy-report.json" ]]; then
        local critical_count
        critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${reports_dir}/trivy-report.json" 2>/dev/null || echo "0")
        
        if [[ "$critical_count" -gt 0 ]]; then
            log_warning "Found ${critical_count} CRITICAL vulnerabilities. Review immediately!"
            return 1
        else
            log_success "No CRITICAL vulnerabilities found"
        fi
    fi
}

# Handle script arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Container Security Scanning Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Environment Variables:"
        echo "  IMAGE_NAME    - Container image name (default: claude-code-manager)"
        echo "  IMAGE_TAG     - Container image tag (default: latest)"
        echo ""
        echo "Options:"
        echo "  --help, -h    - Show this help message"
        echo ""
        echo "This script performs comprehensive security scanning including:"
        echo "  - Vulnerability scanning (Trivy, Grype)"
        echo "  - SBOM generation (Syft)"
        echo "  - Docker security best practices check"
        echo "  - Summary report generation"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac