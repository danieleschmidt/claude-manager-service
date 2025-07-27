#!/bin/bash

set -e

echo "🔒 Running comprehensive security scan for Claude Code Manager..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPORT_DIR="security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCAN_REPORT="${REPORT_DIR}/security_scan_${TIMESTAMP}.json"

# Create reports directory
mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}📋 Security Scan Report will be saved to: ${SCAN_REPORT}${NC}"

# Initialize report
cat > "${SCAN_REPORT}" << EOF
{
  "scan_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scan_version": "1.0.0",
  "repository": "$(basename $(pwd))",
  "scans": {}
}
EOF

# Function to update report
update_report() {
    local scan_name="$1"
    local scan_file="$2"
    local status="$3"
    
    # Create temporary JSON for this scan
    cat > /tmp/scan_update.json << EOF
{
  "${scan_name}": {
    "status": "${status}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "report_file": "${scan_file}"
  }
}
EOF
    
    # Merge with main report using jq
    if command -v jq &> /dev/null; then
        jq --argjson update "$(cat /tmp/scan_update.json)" '.scans += $update' "${SCAN_REPORT}" > /tmp/updated_report.json
        mv /tmp/updated_report.json "${SCAN_REPORT}"
    fi
}

# 1. Bandit - Python Security Linter
echo -e "\n${BLUE}🐍 Running Bandit (Python Security Linter)...${NC}"
if command -v bandit &> /dev/null; then
    bandit_report="${REPORT_DIR}/bandit_${TIMESTAMP}.json"
    if bandit -r src/ -f json -o "${bandit_report}" -ll; then
        echo -e "${GREEN}✅ Bandit scan completed successfully${NC}"
        update_report "bandit" "${bandit_report}" "success"
        
        # Check for high/medium severity issues
        high_issues=$(jq '.results[] | select(.issue_severity == "HIGH") | length' "${bandit_report}" 2>/dev/null || echo "0")
        medium_issues=$(jq '.results[] | select(.issue_severity == "MEDIUM") | length' "${bandit_report}" 2>/dev/null || echo "0")
        
        if [[ "${high_issues}" -gt 0 ]]; then
            echo -e "${RED}⚠️  Found ${high_issues} HIGH severity issues${NC}"
        fi
        if [[ "${medium_issues}" -gt 0 ]]; then
            echo -e "${YELLOW}⚠️  Found ${medium_issues} MEDIUM severity issues${NC}"
        fi
    else
        echo -e "${RED}❌ Bandit scan failed${NC}"
        update_report "bandit" "${bandit_report}" "failed"
    fi
else
    echo -e "${YELLOW}⚠️  Bandit not installed, skipping...${NC}"
    update_report "bandit" "N/A" "skipped"
fi

# 2. Safety - Python Dependency Vulnerability Scanner
echo -e "\n${BLUE}🛡️  Running Safety (Dependency Vulnerability Scanner)...${NC}"
if command -v safety &> /dev/null; then
    safety_report="${REPORT_DIR}/safety_${TIMESTAMP}.json"
    if safety check --json --output "${safety_report}"; then
        echo -e "${GREEN}✅ Safety scan completed successfully${NC}"
        update_report "safety" "${safety_report}" "success"
    else
        echo -e "${RED}❌ Safety scan found vulnerabilities${NC}"
        update_report "safety" "${safety_report}" "vulnerabilities_found"
        
        # Show summary of vulnerabilities
        if [[ -f "${safety_report}" ]]; then
            vuln_count=$(jq '. | length' "${safety_report}" 2>/dev/null || echo "unknown")
            echo -e "${RED}📊 Found ${vuln_count} vulnerabilities in dependencies${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Safety not installed, skipping...${NC}"
    update_report "safety" "N/A" "skipped"
fi

# 3. Semgrep - Static Analysis Security Scanner
echo -e "\n${BLUE}🔍 Running Semgrep (Static Analysis Security Scanner)...${NC}"
if command -v semgrep &> /dev/null; then
    semgrep_report="${REPORT_DIR}/semgrep_${TIMESTAMP}.json"
    if semgrep --config=auto --json --output="${semgrep_report}" src/; then
        echo -e "${GREEN}✅ Semgrep scan completed successfully${NC}"
        update_report "semgrep" "${semgrep_report}" "success"
        
        # Check for findings
        if [[ -f "${semgrep_report}" ]]; then
            findings=$(jq '.results | length' "${semgrep_report}" 2>/dev/null || echo "0")
            if [[ "${findings}" -gt 0 ]]; then
                echo -e "${YELLOW}📊 Found ${findings} potential security issues${NC}"
            fi
        fi
    else
        echo -e "${RED}❌ Semgrep scan failed${NC}"
        update_report "semgrep" "${semgrep_report}" "failed"
    fi
else
    echo -e "${YELLOW}⚠️  Semgrep not installed, skipping...${NC}"
    update_report "semgrep" "N/A" "skipped"
fi

# 4. Docker Security Scan
echo -e "\n${BLUE}🐳 Running Docker Security Scan...${NC}"
if command -v docker &> /dev/null && [[ -f "Dockerfile" ]]; then
    docker_report="${REPORT_DIR}/docker_${TIMESTAMP}.txt"
    
    # Build image for scanning
    echo "Building Docker image for security scan..."
    if docker build -t claude-manager-security-scan:latest . > /dev/null 2>&1; then
        
        # Scan with Docker Scout if available
        if docker scout --help &> /dev/null; then
            echo "Running Docker Scout scan..."
            docker scout cves claude-manager-security-scan:latest > "${docker_report}" 2>&1
            echo -e "${GREEN}✅ Docker Scout scan completed${NC}"
            update_report "docker_scout" "${docker_report}" "success"
        else
            # Fallback to basic Docker history analysis
            echo "Docker Scout not available, running basic analysis..."
            docker history claude-manager-security-scan:latest > "${docker_report}"
            echo -e "${YELLOW}⚠️  Limited Docker analysis completed${NC}"
            update_report "docker_basic" "${docker_report}" "limited"
        fi
        
        # Clean up scan image
        docker rmi claude-manager-security-scan:latest > /dev/null 2>&1 || true
    else
        echo -e "${RED}❌ Failed to build Docker image for scanning${NC}"
        update_report "docker" "N/A" "build_failed"
    fi
else
    echo -e "${YELLOW}⚠️  Docker not available or no Dockerfile found${NC}"
    update_report "docker" "N/A" "skipped"
fi

# 5. Secret Detection
echo -e "\n${BLUE}🔐 Running Secret Detection Scan...${NC}"
secret_report="${REPORT_DIR}/secrets_${TIMESTAMP}.txt"

# Use git-secrets if available, otherwise use basic grep
if command -v git-secrets &> /dev/null; then
    echo "Running git-secrets scan..."
    if git secrets --scan-history > "${secret_report}" 2>&1; then
        echo -e "${GREEN}✅ No secrets detected by git-secrets${NC}"
        update_report "git_secrets" "${secret_report}" "clean"
    else
        echo -e "${RED}❌ Potential secrets detected!${NC}"
        update_report "git_secrets" "${secret_report}" "secrets_found"
    fi
else
    echo "git-secrets not available, running basic pattern detection..."
    
    # Basic secret patterns
    patterns=(
        "password.*=.*['\"][^'\"]*['\"]"
        "api[_-]?key.*=.*['\"][^'\"]*['\"]"
        "secret.*=.*['\"][^'\"]*['\"]"
        "token.*=.*['\"][^'\"]*['\"]"
        "auth.*=.*['\"][^'\"]*['\"]"
        "ghp_[a-zA-Z0-9]{36}"
        "github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"
        "sk-[a-zA-Z0-9]{48}"
    )
    
    secret_found=false
    for pattern in "${patterns[@]}"; do
        if grep -r -i -E "${pattern}" src/ --exclude-dir=__pycache__ >> "${secret_report}" 2>/dev/null; then
            secret_found=true
        fi
    done
    
    if [[ "${secret_found}" == "true" ]]; then
        echo -e "${RED}❌ Potential secrets detected in code!${NC}"
        update_report "basic_secret_scan" "${secret_report}" "secrets_found"
    else
        echo -e "${GREEN}✅ No obvious secrets detected${NC}"
        update_report "basic_secret_scan" "${secret_report}" "clean"
    fi
fi

# 6. File Permission Check
echo -e "\n${BLUE}📁 Checking File Permissions...${NC}"
permission_report="${REPORT_DIR}/permissions_${TIMESTAMP}.txt"

echo "Checking for overly permissive files..." > "${permission_report}"

# Check for world-writable files
echo "World-writable files:" >> "${permission_report}"
find . -type f -perm -002 -not -path "./.*" >> "${permission_report}" 2>/dev/null || true

# Check for executable files that shouldn't be
echo -e "\nUnexpected executable files:" >> "${permission_report}"
find . -name "*.py" -perm -111 -not -path "./.*" >> "${permission_report}" 2>/dev/null || true
find . -name "*.json" -perm -111 -not -path "./.*" >> "${permission_report}" 2>/dev/null || true
find . -name "*.md" -perm -111 -not -path "./.*" >> "${permission_report}" 2>/dev/null || true

# Check for files with no read permissions
echo -e "\nFiles with restricted read permissions:" >> "${permission_report}"
find . -type f -not -perm -004 -not -path "./.*" >> "${permission_report}" 2>/dev/null || true

permission_issues=$(grep -v "^$" "${permission_report}" | grep -v ":" | wc -l)
if [[ "${permission_issues}" -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Found ${permission_issues} permission issues${NC}"
    update_report "file_permissions" "${permission_report}" "issues_found"
else
    echo -e "${GREEN}✅ File permissions look good${NC}"
    update_report "file_permissions" "${permission_report}" "clean"
fi

# 7. Configuration Security Check
echo -e "\n${BLUE}⚙️  Checking Configuration Security...${NC}"
config_report="${REPORT_DIR}/config_security_${TIMESTAMP}.txt"

echo "Configuration Security Analysis" > "${config_report}"
echo "===============================" >> "${config_report}"

# Check for debug mode in production configs
echo -e "\nChecking for debug mode settings:" >> "${config_report}"
grep -r -i "debug.*=.*true" . --include="*.json" --include="*.yaml" --include="*.yml" --include="*.env*" >> "${config_report}" 2>/dev/null || echo "No debug mode found" >> "${config_report}"

# Check for default passwords/keys
echo -e "\nChecking for default/weak configurations:" >> "${config_report}"
grep -r -i -E "(password.*=.*(admin|password|123|default))" . --include="*.json" --include="*.yaml" --include="*.yml" >> "${config_report}" 2>/dev/null || echo "No obvious default passwords found" >> "${config_report}"

# Check for hardcoded IPs
echo -e "\nChecking for hardcoded IP addresses:" >> "${config_report}"
grep -r -E "([0-9]{1,3}\.){3}[0-9]{1,3}" . --include="*.py" --include="*.json" --include="*.yaml" | grep -v "127.0.0.1\|0.0.0.0\|localhost" >> "${config_report}" 2>/dev/null || echo "No hardcoded IPs found" >> "${config_report}"

config_issues=$(grep -c "found" "${config_report}" | grep -v "No.*found" | wc -l)
if [[ "${config_issues}" -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Found potential configuration security issues${NC}"
    update_report "config_security" "${config_report}" "issues_found"
else
    echo -e "${GREEN}✅ Configuration security looks good${NC}"
    update_report "config_security" "${config_report}" "clean"
fi

# 8. Generate Summary Report
echo -e "\n${BLUE}📊 Generating Security Summary...${NC}"
summary_report="${REPORT_DIR}/security_summary_${TIMESTAMP}.md"

cat > "${summary_report}" << EOF
# Security Scan Summary

**Scan Date:** $(date)
**Repository:** $(basename $(pwd))
**Scan ID:** ${TIMESTAMP}

## Overview

This report summarizes the security scan results for Claude Code Manager.

## Scan Results

EOF

# Add individual scan results to summary
if command -v jq &> /dev/null && [[ -f "${SCAN_REPORT}" ]]; then
    echo "### Detailed Results" >> "${summary_report}"
    echo "" >> "${summary_report}"
    
    jq -r '.scans | to_entries[] | "- **\(.key)**: \(.value.status)"' "${SCAN_REPORT}" >> "${summary_report}"
    
    echo "" >> "${summary_report}"
    echo "### Recommendations" >> "${summary_report}"
    echo "" >> "${summary_report}"
    
    # Add recommendations based on findings
    if jq -e '.scans | to_entries[] | select(.value.status == "vulnerabilities_found" or .value.status == "secrets_found" or .value.status == "issues_found")' "${SCAN_REPORT}" > /dev/null; then
        echo "⚠️ **Action Required**: Security issues were detected that require attention." >> "${summary_report}"
        echo "" >> "${summary_report}"
        echo "1. Review detailed reports in the \`${REPORT_DIR}\` directory" >> "${summary_report}"
        echo "2. Address high and medium severity vulnerabilities immediately" >> "${summary_report}"
        echo "3. Update dependencies with known vulnerabilities" >> "${summary_report}"
        echo "4. Remove or secure any detected secrets" >> "${summary_report}"
        echo "5. Fix configuration security issues" >> "${summary_report}"
    else
        echo "✅ **All Clear**: No significant security issues detected." >> "${summary_report}"
        echo "" >> "${summary_report}"
        echo "Continue with regular security practices:" >> "${summary_report}"
        echo "1. Keep dependencies updated" >> "${summary_report}"
        echo "2. Regular security scans" >> "${summary_report}"
        echo "3. Follow secure coding practices" >> "${summary_report}"
    fi
fi

echo "" >> "${summary_report}"
echo "## Report Files" >> "${summary_report}"
echo "" >> "${summary_report}"
echo "- **Main Report:** \`${SCAN_REPORT}\`" >> "${summary_report}"
echo "- **Summary:** \`${summary_report}\`" >> "${summary_report}"
echo "- **Individual Reports:** Check \`${REPORT_DIR}/\` directory" >> "${summary_report}"

# Final Summary
echo -e "\n${GREEN}🎉 Security scan completed!${NC}"
echo -e "${BLUE}📋 Reports saved to: ${REPORT_DIR}/${NC}"
echo -e "${BLUE}📄 Summary report: ${summary_report}${NC}"

# Exit with error code if critical issues found
if command -v jq &> /dev/null && [[ -f "${SCAN_REPORT}" ]]; then
    critical_issues=$(jq -r '.scans | to_entries[] | select(.value.status == "vulnerabilities_found" or .value.status == "secrets_found") | length' "${SCAN_REPORT}")
    if [[ "${critical_issues}" -gt 0 ]]; then
        echo -e "\n${RED}⚠️  Critical security issues detected. Please review and address them.${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}✅ Security scan completed successfully!${NC}"