# Open Policy Agent (OPA) Policies for Claude Code Manager
# Implements policy as code for security, compliance, and governance

package claude_code_manager.security

import rego.v1

# Container Security Policies
container_security := {
    "deny_root_containers",
    "require_readonly_filesystem",
    "deny_privileged_containers",
    "require_security_context",
    "limit_capabilities"
}

# Deny containers running as root user
deny_root_containers contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.securityContext.runAsUser == 0
    msg := sprintf("Container '%s' must not run as root user (UID 0)", [container.name])
}

deny_root_containers contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.securityContext.runAsRoot == true
    msg := sprintf("Container '%s' must not run as root", [container.name])
}

# Require read-only root filesystem
require_readonly_filesystem contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.securityContext.readOnlyRootFilesystem
    msg := sprintf("Container '%s' must have read-only root filesystem", [container.name])
}

# Deny privileged containers
deny_privileged_containers contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.securityContext.privileged == true
    msg := sprintf("Container '%s' must not run in privileged mode", [container.name])
}

# Require security context to be defined
require_security_context contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.securityContext
    msg := sprintf("Container '%s' must define securityContext", [container.name])
}

# Limit Linux capabilities
allowed_capabilities := {
    "NET_BIND_SERVICE",
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "SETGID",
    "SETUID"
}

limit_capabilities contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    some capability in container.securityContext.capabilities.add
    not capability in allowed_capabilities
    msg := sprintf("Container '%s' uses disallowed capability '%s'", [container.name, capability])
}

# Network Security Policies
network_security := {
    "require_network_policies",
    "deny_host_network",
    "restrict_ingress_sources",
    "require_tls_encryption"
}

# Require NetworkPolicy for pods
require_network_policies contains msg if {
    input.kind == "Pod"
    input.metadata.namespace != "kube-system"
    not has_network_policy
    msg := "Pod must have associated NetworkPolicy for network segmentation"
}

has_network_policy if {
    some policy in data.kubernetes.networkpolicies
    policy.metadata.namespace == input.metadata.namespace
    selector_matches(policy.spec.podSelector, input.metadata.labels)
}

# Helper function to check if selector matches labels
selector_matches(selector, labels) if {
    selector == {}  # Empty selector matches all pods
}

selector_matches(selector, labels) if {
    selector.matchLabels
    every key, value in selector.matchLabels {
        labels[key] == value
    }
}

# Deny host network usage
deny_host_network contains msg if {
    input.kind == "Pod"
    input.spec.hostNetwork == true
    msg := "Pod must not use host network"
}

# Resource Management Policies
resource_management := {
    "require_resource_limits",
    "require_resource_requests",
    "validate_resource_ratios",
    "limit_ephemeral_storage"
}

# Require resource limits
require_resource_limits contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.resources.limits
    msg := sprintf("Container '%s' must define resource limits", [container.name])
}

require_resource_limits contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.resources.limits
    not container.resources.limits.memory
    msg := sprintf("Container '%s' must define memory limits", [container.name])
}

require_resource_limits contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.resources.limits
    not container.resources.limits.cpu
    msg := sprintf("Container '%s' must define CPU limits", [container.name])
}

# Require resource requests
require_resource_requests contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.resources.requests
    msg := sprintf("Container '%s' must define resource requests", [container.name])
}

# Validate resource request/limit ratios
validate_resource_ratios contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.resources.requests.memory
    container.resources.limits.memory
    memory_request := parse_memory(container.resources.requests.memory)
    memory_limit := parse_memory(container.resources.limits.memory)
    memory_limit / memory_request > 4  # Limit should not be more than 4x request
    msg := sprintf("Container '%s' memory limit/request ratio exceeds 4:1", [container.name])
}

# Helper function to parse memory values (simplified)
parse_memory(memory_str) := result if {
    endswith(memory_str, "Mi")
    result := to_number(trim_suffix(memory_str, "Mi"))
}

parse_memory(memory_str) := result if {
    endswith(memory_str, "Gi")
    result := to_number(trim_suffix(memory_str, "Gi")) * 1024
}

# Image Security Policies
image_security := {
    "require_image_tags",
    "deny_latest_tags",
    "require_trusted_registries",
    "require_image_scanning"
}

# Require specific image tags (no :latest)
require_image_tags contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    endswith(container.image, ":latest")
    msg := sprintf("Container '%s' must not use 'latest' tag", [container.name])
}

deny_latest_tags contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not contains(container.image, ":")
    msg := sprintf("Container '%s' must specify explicit image tag", [container.name])
}

# Require trusted container registries
trusted_registries := {
    "ghcr.io",
    "gcr.io",
    "registry.terragon.ai",
    "docker.io/library"  # Official Docker images
}

require_trusted_registries contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not image_from_trusted_registry(container.image)
    msg := sprintf("Container '%s' image must be from trusted registry", [container.name])
}

image_from_trusted_registry(image) if {
    some registry in trusted_registries
    startswith(image, registry)
}

# Data Protection Policies
data_protection := {
    "require_volume_encryption",
    "deny_host_path_volumes",
    "require_secret_volume_permissions",
    "validate_pvc_storage_class"
}

# Deny hostPath volumes (security risk)
deny_host_path_volumes contains msg if {
    input.kind == "Pod"
    some volume in input.spec.volumes
    volume.hostPath
    msg := sprintf("Volume '%s' must not use hostPath", [volume.name])
}

# Require appropriate permissions for secret volumes
require_secret_volume_permissions contains msg if {
    input.kind == "Pod"
    some volume in input.spec.volumes
    volume.secret
    not volume.secret.defaultMode
    msg := sprintf("Secret volume '%s' must specify defaultMode", [volume.name])
}

require_secret_volume_permissions contains msg if {
    input.kind == "Pod"
    some volume in input.spec.volumes
    volume.secret
    volume.secret.defaultMode
    volume.secret.defaultMode & 0o077 != 0  # Check for world/group permissions
    msg := sprintf("Secret volume '%s' must not have world/group permissions", [volume.name])
}

# Compliance Policies
compliance := {
    "require_labels",
    "require_annotations",
    "validate_naming_conventions",
    "require_documentation"
}

# Required labels for governance
required_labels := {
    "app.kubernetes.io/name",
    "app.kubernetes.io/version",
    "app.kubernetes.io/component",
    "app.kubernetes.io/managed-by"
}

require_labels contains msg if {
    input.kind == "Pod"
    some required_label in required_labels
    not input.metadata.labels[required_label]
    msg := sprintf("Pod must have label '%s'", [required_label])
}

# Required annotations for compliance
required_annotations := {
    "terragon.ai/owner",
    "terragon.ai/cost-center",
    "terragon.ai/environment"
}

require_annotations contains msg if {
    input.kind == "Pod"
    some required_annotation in required_annotations
    not input.metadata.annotations[required_annotation]
    msg := sprintf("Pod must have annotation '%s'", [required_annotation])
}

# Validate naming conventions
validate_naming_conventions contains msg if {
    input.kind == "Pod"
    not regex.match("^[a-z0-9-]+$", input.metadata.name)
    msg := "Pod name must contain only lowercase letters, numbers, and hyphens"
}

validate_naming_conventions contains msg if {
    input.kind == "Pod"
    count(input.metadata.name) > 63
    msg := "Pod name must not exceed 63 characters"
}

# Service Mesh Policies
service_mesh := {
    "require_istio_sidecar",
    "enforce_mtls",
    "validate_virtual_services",
    "require_destination_rules"
}

# Require Istio sidecar injection
require_istio_sidecar contains msg if {
    input.kind == "Pod"
    input.metadata.namespace != "kube-system"
    input.metadata.namespace != "istio-system"
    not input.metadata.annotations["sidecar.istio.io/inject"] == "true"
    not input.metadata.labels["sidecar.istio.io/inject"] == "true"
    msg := "Pod must have Istio sidecar injection enabled"
}

# Monitoring and Observability Policies
observability := {
    "require_health_checks",
    "require_metrics_endpoint",
    "validate_log_configuration",
    "require_tracing_headers"
}

# Require health check endpoints
require_health_checks contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.livenessProbe
    msg := sprintf("Container '%s' must define livenessProbe", [container.name])
}

require_health_checks contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    not container.readinessProbe
    msg := sprintf("Container '%s' must define readinessProbe", [container.name])
}

# Deployment Policies
deployment := {
    "require_deployment_strategy",
    "validate_replica_count",
    "require_pod_disruption_budget",
    "validate_update_strategy"
}

# Require explicit deployment strategy
require_deployment_strategy contains msg if {
    input.kind == "Deployment"
    not input.spec.strategy
    msg := "Deployment must specify update strategy"
}

# Validate replica count for production
validate_replica_count contains msg if {
    input.kind == "Deployment"
    input.metadata.annotations["terragon.ai/environment"] == "production"
    input.spec.replicas < 2
    msg := "Production deployments must have at least 2 replicas"
}

# CI/CD Pipeline Policies
cicd := {
    "require_signed_commits",
    "validate_pipeline_security",
    "require_approval_process",
    "validate_artifact_integrity"
}

# Require signed commits for security
require_signed_commits contains msg if {
    input.kind == "GitCommit"
    not input.signature.verified
    msg := "Commits must be cryptographically signed"
}

# Validate pipeline security configurations
validate_pipeline_security contains msg if {
    input.kind == "Pipeline"
    some step in input.spec.steps
    step.image
    not image_from_trusted_registry(step.image)
    msg := sprintf("Pipeline step '%s' must use image from trusted registry", [step.name])
}

# Cost Management Policies
cost_management := {
    "require_cost_labels",
    "validate_resource_efficiency",
    "limit_expensive_resources",
    "require_cost_center"
}

# Require cost-related labels
cost_labels := {
    "cost-center",
    "project",
    "team",
    "environment"
}

require_cost_labels contains msg if {
    input.kind in {"Pod", "Deployment", "Service", "PersistentVolumeClaim"}
    some cost_label in cost_labels
    not input.metadata.labels[cost_label]
    msg := sprintf("Resource must have cost label '%s'", [cost_label])
}

# Limit expensive GPU resources
limit_expensive_resources contains msg if {
    input.kind == "Pod"
    some container in input.spec.containers
    container.resources.requests["nvidia.com/gpu"]
    to_number(container.resources.requests["nvidia.com/gpu"]) > 2
    msg := sprintf("Container '%s' cannot request more than 2 GPUs", [container.name])
}

# Environment-specific Policies
environment_policies := {
    "production": {
        "require_resource_limits",
        "require_health_checks",
        "deny_debug_containers",
        "require_multiple_replicas"
    },
    "staging": {
        "require_resource_requests",
        "require_health_checks",
        "allow_debug_containers"
    },
    "development": {
        "allow_relaxed_security",
        "allow_debug_containers",
        "allow_latest_tags"
    }
}

# Production-specific policies
deny_debug_containers contains msg if {
    input.kind == "Pod"
    input.metadata.annotations["terragon.ai/environment"] == "production"
    some container in input.spec.containers
    container.name == "debug"
    msg := "Debug containers not allowed in production"
}

# Custom Business Logic Policies
business_logic := {
    "validate_service_dependencies",
    "require_backup_strategy",
    "validate_data_retention",
    "require_disaster_recovery"
}

# Validate service dependencies are documented
validate_service_dependencies contains msg if {
    input.kind == "Service"
    not input.metadata.annotations["terragon.ai/dependencies"]
    msg := "Service must document its dependencies"
}

# Main policy evaluation
violations := 
    container_security |
    network_security |
    resource_management |
    image_security |
    data_protection |
    compliance |
    service_mesh |
    observability |
    deployment |
    cicd |
    cost_management |
    business_logic

# Helper functions
has_annotation(obj, key) if {
    obj.metadata.annotations[key]
}

has_label(obj, key) if {
    obj.metadata.labels[key]
}

get_environment(obj) := env if {
    env := obj.metadata.annotations["terragon.ai/environment"]
}

get_environment(obj) := "development" if {
    not obj.metadata.annotations["terragon.ai/environment"]
}