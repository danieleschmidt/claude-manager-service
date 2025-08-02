#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) generation script.
Generates SPDX-compliant SBOM for Claude Code Manager.
"""

import json
import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def get_git_info() -> Dict[str, str]:
    """Get Git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always"], text=True
        ).strip()
        
        return {
            "commit": commit_hash,
            "tag": tag,
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip()
        }
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "tag": "unknown", "branch": "unknown"}


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except (IOError, OSError):
        return "unknown"


def parse_requirements(requirements_file: str) -> List[Dict[str, Any]]:
    """Parse requirements.txt file and return package information."""
    packages = []
    
    if not os.path.exists(requirements_file):
        return packages
    
    try:
        with open(requirements_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Simple parsing - could be enhanced for complex requirements
                    if ">=" in line:
                        name, version = line.split(">=", 1)
                        version_constraint = f">={version}"
                    elif "==" in line:
                        name, version = line.split("==", 1)
                        version_constraint = f"=={version}"
                    elif ">" in line:
                        name, version = line.split(">", 1)
                        version_constraint = f">{version}"
                    else:
                        name = line
                        version_constraint = "*"
                    
                    packages.append({
                        "SPDXID": f"SPDXRef-Package-{name.replace('[', '').replace(']', '').replace('_', '-')}",
                        "name": name.split("[")[0],  # Remove extras like [toml]
                        "downloadLocation": f"https://pypi.org/project/{name.split('[')[0]}/",
                        "filesAnalyzed": False,
                        "versionInfo": version_constraint,
                        "supplier": "NOASSERTION",
                        "licenseConcluded": "NOASSERTION",
                        "licenseDeclared": "NOASSERTION",
                        "copyrightText": "NOASSERTION",
                        "packageVerificationCode": {
                            "packageVerificationCodeValue": "NOASSERTION"
                        }
                    })
    except Exception as e:
        print(f"Error parsing {requirements_file}: {e}")
    
    return packages


def get_source_files() -> List[Dict[str, Any]]:
    """Get information about source files."""
    files = []
    source_dir = Path("src")
    
    if source_dir.exists():
        for file_path in source_dir.rglob("*.py"):
            if file_path.is_file():
                file_hash = calculate_file_hash(file_path)
                files.append({
                    "SPDXID": f"SPDXRef-File-{file_path.name.replace('.', '-')}",
                    "fileName": str(file_path),
                    "checksums": [
                        {
                            "algorithm": "SHA256",
                            "checksumValue": file_hash
                        }
                    ],
                    "licenseConcluded": "MIT",
                    "copyrightText": "Copyright (c) 2025 Terragon Labs"
                })
    
    return files


def generate_sbom() -> Dict[str, Any]:
    """Generate SPDX-compliant SBOM."""
    git_info = get_git_info()
    current_time = datetime.utcnow().isoformat() + "Z"
    
    # Parse dependencies
    python_packages = parse_requirements("requirements.txt")
    dev_packages = parse_requirements("requirements-dev.txt")
    
    # Get source files
    source_files = get_source_files()
    
    # Main package information
    main_package = {
        "SPDXID": "SPDXRef-Package-claude-code-manager",
        "name": "claude-code-manager",
        "downloadLocation": "https://github.com/danieleschmidt/claude-manager-service",
        "filesAnalyzed": True,
        "versionInfo": git_info["tag"],
        "supplier": "Organization: Terragon Labs",
        "originator": "Organization: Terragon Labs",
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "Copyright (c) 2025 Terragon Labs",
        "description": "Autonomous software development lifecycle management system",
        "homepage": "https://github.com/danieleschmidt/claude-manager-service",
        "packageVerificationCode": {
            "packageVerificationCodeValue": calculate_file_hash(Path("src"))
        },
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": "pkg:github/danieleschmidt/claude-manager-service@" + git_info["tag"]
            },
            {
                "referenceCategory": "OTHER",
                "referenceType": "vcs",
                "referenceLocator": f"git+https://github.com/danieleschmidt/claude-manager-service.git@{git_info['commit']}"
            }
        ]
    }
    
    # Build relationships
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package-claude-code-manager"
        }
    ]
    
    # Add dependency relationships
    for package in python_packages + dev_packages:
        relationships.append({
            "spdxElementId": "SPDXRef-Package-claude-code-manager",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": package["SPDXID"]
        })
    
    # Add file relationships
    for file_info in source_files:
        relationships.append({
            "spdxElementId": "SPDXRef-Package-claude-code-manager",
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": file_info["SPDXID"]
        })
    
    sbom = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "spdxVersion": "SPDX-2.3",
        "creationInfo": {
            "created": current_time,
            "creators": [
                "Tool: claude-code-manager-sbom-generator",
                "Organization: Terragon Labs"
            ],
            "licenseListVersion": "3.21"
        },
        "name": f"claude-code-manager-{git_info['tag']}",
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://terragon.ai/spdx/claude-code-manager-{git_info['tag']}-{current_time}",
        "documentDescribes": ["SPDXRef-Package-claude-code-manager"],
        "packages": [main_package] + python_packages + dev_packages,
        "files": source_files,
        "relationships": relationships,
        "annotations": [
            {
                "annotationType": "REVIEW",
                "annotator": "Tool: claude-code-manager-sbom-generator",
                "annotationDate": current_time,
                "annotationComment": f"Generated SBOM for commit {git_info['commit']}",
                "annotationSPDXRef": "SPDXRef-DOCUMENT"
            }
        ]
    }
    
    return sbom


def main():
    """Main function to generate and save SBOM."""
    print("🔍 Generating Software Bill of Materials (SBOM)...")
    
    try:
        sbom = generate_sbom()
        
        # Save SBOM to file
        sbom_file = "SBOM.json"
        with open(sbom_file, "w") as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
        
        print(f"✅ SBOM generated successfully: {sbom_file}")
        
        # Print summary
        package_count = len(sbom["packages"])
        file_count = len(sbom["files"])
        relationship_count = len(sbom["relationships"])
        
        print(f"📊 SBOM Summary:")
        print(f"   - Packages: {package_count}")
        print(f"   - Files: {file_count}")
        print(f"   - Relationships: {relationship_count}")
        print(f"   - SPDX Version: {sbom['spdxVersion']}")
        print(f"   - Generated: {sbom['creationInfo']['created']}")
        
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())