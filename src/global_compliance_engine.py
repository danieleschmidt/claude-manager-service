#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - GLOBAL COMPLIANCE ENGINE
Comprehensive global compliance framework with multi-region support, 
internationalization, and regulatory compliance management
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import structlog

logger = structlog.get_logger("GlobalComplianceEngine")

class ComplianceRegion(Enum):
    """Supported compliance regions"""
    EU = "eu"  # European Union (GDPR)
    US = "us"  # United States (CCPA, SOX, HIPAA)
    APAC = "apac"  # Asia-Pacific (PDPA, etc.)
    UK = "uk"  # United Kingdom (UK GDPR)
    CANADA = "canada"  # Canada (PIPEDA)
    GLOBAL = "global"  # Global standards (ISO, SOC)

class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    ISO27001 = "iso27001"  # Information Security Management
    SOC2 = "soc2"  # Service Organization Control 2
    PDPA = "pdpa"  # Personal Data Protection Act
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"  # PII/PHI
    FINANCIAL = "financial"

class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NEEDS_REVIEW = "needs_review"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class ComplianceRule:
    """Individual compliance rule"""
    id: str
    name: str
    description: str
    standard: ComplianceStandard
    region: ComplianceRegion
    severity: str  # critical, high, medium, low
    category: str  # data_protection, access_control, audit, etc.
    enabled: bool = True

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    id: str
    rule_id: str
    timestamp: datetime
    severity: str
    description: str
    resource: str
    region: ComplianceRegion
    remediation: str
    status: str = "open"

@dataclass
class LocalizationConfig:
    """Localization configuration"""
    locale: str
    language: str
    country: str
    timezone: str
    currency: str
    date_format: str
    number_format: str
    text_direction: str = "ltr"

class InternationalizationManager:
    """Manages internationalization and localization"""
    
    def __init__(self):
        self.supported_locales = {
            "en_US": LocalizationConfig("en_US", "English", "United States", "America/New_York", "USD", "MM/DD/YYYY", "#,##0.00"),
            "en_GB": LocalizationConfig("en_GB", "English", "United Kingdom", "Europe/London", "GBP", "DD/MM/YYYY", "#,##0.00"),
            "de_DE": LocalizationConfig("de_DE", "German", "Germany", "Europe/Berlin", "EUR", "DD.MM.YYYY", "#.##0,00"),
            "fr_FR": LocalizationConfig("fr_FR", "French", "France", "Europe/Paris", "EUR", "DD/MM/YYYY", "# ##0,00"),
            "ja_JP": LocalizationConfig("ja_JP", "Japanese", "Japan", "Asia/Tokyo", "JPY", "YYYY/MM/DD", "#,##0"),
            "zh_CN": LocalizationConfig("zh_CN", "Chinese (Simplified)", "China", "Asia/Shanghai", "CNY", "YYYY/MM/DD", "#,##0.00"),
            "es_ES": LocalizationConfig("es_ES", "Spanish", "Spain", "Europe/Madrid", "EUR", "DD/MM/YYYY", "#.##0,00"),
            "pt_BR": LocalizationConfig("pt_BR", "Portuguese", "Brazil", "America/Sao_Paulo", "BRL", "DD/MM/YYYY", "#.##0,00"),
        }
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files"""
        # Default English translations
        self.translations["en_US"] = {
            "app.name": "Terragon SDLC Manager",
            "app.description": "Autonomous Software Development Lifecycle Management System",
            "status.healthy": "Healthy",
            "status.degraded": "Degraded", 
            "status.unhealthy": "Unhealthy",
            "action.start": "Start",
            "action.stop": "Stop",
            "action.restart": "Restart",
            "error.generic": "An error occurred",
            "error.network": "Network error",
            "error.permission": "Permission denied",
            "compliance.gdpr": "GDPR Compliance",
            "compliance.ccpa": "CCPA Compliance",
            "compliance.status.compliant": "Compliant",
            "compliance.status.non_compliant": "Non-Compliant",
        }
        
        # German translations
        self.translations["de_DE"] = {
            "app.name": "Terragon SDLC Manager",
            "app.description": "Autonomes System für Software-Entwicklungslebenszyklus-Management",
            "status.healthy": "Gesund",
            "status.degraded": "Beeinträchtigt",
            "status.unhealthy": "Ungesund",
            "action.start": "Starten",
            "action.stop": "Stoppen",
            "action.restart": "Neu starten",
            "error.generic": "Ein Fehler ist aufgetreten",
            "error.network": "Netzwerkfehler",
            "error.permission": "Berechtigung verweigert",
            "compliance.gdpr": "DSGVO-Konformität",
            "compliance.ccpa": "CCPA-Konformität",
            "compliance.status.compliant": "Konform",
            "compliance.status.non_compliant": "Nicht konform",
        }
        
        # French translations
        self.translations["fr_FR"] = {
            "app.name": "Gestionnaire SDLC Terragon",
            "app.description": "Système Autonome de Gestion du Cycle de Vie du Développement Logiciel",
            "status.healthy": "Sain",
            "status.degraded": "Dégradé",
            "status.unhealthy": "Malsain",
            "action.start": "Démarrer",
            "action.stop": "Arrêter",
            "action.restart": "Redémarrer",
            "error.generic": "Une erreur s'est produite",
            "error.network": "Erreur réseau",
            "error.permission": "Permission refusée",
            "compliance.gdpr": "Conformité RGPD",
            "compliance.ccpa": "Conformité CCPA",
            "compliance.status.compliant": "Conforme",
            "compliance.status.non_compliant": "Non conforme",
        }
        
        # Japanese translations
        self.translations["ja_JP"] = {
            "app.name": "Terragon SDLC マネージャー",
            "app.description": "自律型ソフトウェア開発ライフサイクル管理システム",
            "status.healthy": "正常",
            "status.degraded": "劣化",
            "status.unhealthy": "異常",
            "action.start": "開始",
            "action.stop": "停止",
            "action.restart": "再起動",
            "error.generic": "エラーが発生しました",
            "error.network": "ネットワークエラー",
            "error.permission": "アクセス拒否",
            "compliance.gdpr": "GDPR準拠",
            "compliance.ccpa": "CCPA準拠",
            "compliance.status.compliant": "準拠",
            "compliance.status.non_compliant": "非準拠",
        }
        
        # Chinese (Simplified) translations
        self.translations["zh_CN"] = {
            "app.name": "Terragon SDLC 管理器",
            "app.description": "自主软件开发生命周期管理系统",
            "status.healthy": "健康",
            "status.degraded": "降级",
            "status.unhealthy": "不健康",
            "action.start": "开始",
            "action.stop": "停止", 
            "action.restart": "重启",
            "error.generic": "发生错误",
            "error.network": "网络错误",
            "error.permission": "权限被拒绝",
            "compliance.gdpr": "GDPR合规",
            "compliance.ccpa": "CCPA合规",
            "compliance.status.compliant": "合规",
            "compliance.status.non_compliant": "不合规",
        }
    
    def get_translation(self, key: str, locale: str = "en_US") -> str:
        """Get translation for a key in the specified locale"""
        translations = self.translations.get(locale, self.translations["en_US"])
        return translations.get(key, key)  # Return key if translation not found
    
    def get_locale_config(self, locale: str) -> Optional[LocalizationConfig]:
        """Get localization configuration for a locale"""
        return self.supported_locales.get(locale)
    
    def format_currency(self, amount: float, locale: str = "en_US") -> str:
        """Format currency amount for locale"""
        config = self.get_locale_config(locale)
        if not config:
            return f"${amount:.2f}"
        
        # Simplified currency formatting
        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "BRL": "R$"
        }
        symbol = currency_symbols.get(config.currency, config.currency)
        
        if locale in ["de_DE", "fr_FR", "es_ES"]:
            return f"{amount:,.2f} {symbol}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{symbol}{amount:,.2f}"
    
    def format_date(self, date: datetime, locale: str = "en_US") -> str:
        """Format date for locale"""
        config = self.get_locale_config(locale)
        if not config:
            return date.strftime("%Y-%m-%d")
        
        format_map = {
            "MM/DD/YYYY": "%m/%d/%Y",
            "DD/MM/YYYY": "%d/%m/%Y", 
            "DD.MM.YYYY": "%d.%m.%Y",
            "YYYY/MM/DD": "%Y/%m/%d",
        }
        
        return date.strftime(format_map.get(config.date_format, "%Y-%m-%d"))

class ComplianceRuleEngine:
    """Engine for managing and evaluating compliance rules"""
    
    def __init__(self):
        self.rules: List[ComplianceRule] = []
        self.violations: List[ComplianceViolation] = []
        self._initialize_standard_rules()
    
    def _initialize_standard_rules(self):
        """Initialize standard compliance rules"""
        
        # GDPR Rules
        self.rules.extend([
            ComplianceRule(
                "gdpr_data_encryption",
                "Data Encryption at Rest",
                "Personal data must be encrypted when stored",
                ComplianceStandard.GDPR,
                ComplianceRegion.EU,
                "critical",
                "data_protection"
            ),
            ComplianceRule(
                "gdpr_data_retention", 
                "Data Retention Policy",
                "Personal data must not be retained longer than necessary",
                ComplianceStandard.GDPR,
                ComplianceRegion.EU,
                "high",
                "data_protection"
            ),
            ComplianceRule(
                "gdpr_access_logging",
                "Access Logging",
                "All access to personal data must be logged",
                ComplianceStandard.GDPR,
                ComplianceRegion.EU,
                "medium",
                "audit"
            ),
            ComplianceRule(
                "gdpr_consent_management",
                "Consent Management",
                "Explicit consent must be obtained for data processing",
                ComplianceStandard.GDPR,
                ComplianceRegion.EU,
                "critical",
                "data_protection"
            ),
        ])
        
        # CCPA Rules
        self.rules.extend([
            ComplianceRule(
                "ccpa_data_disclosure",
                "Data Disclosure Requirements",
                "Consumers must be informed about data collection",
                ComplianceStandard.CCPA,
                ComplianceRegion.US,
                "high",
                "data_protection"
            ),
            ComplianceRule(
                "ccpa_opt_out",
                "Opt-Out Mechanism",
                "Consumers must have option to opt out of data sale",
                ComplianceStandard.CCPA,
                ComplianceRegion.US,
                "high",
                "data_protection"
            ),
        ])
        
        # SOC2 Rules
        self.rules.extend([
            ComplianceRule(
                "soc2_security_monitoring",
                "Security Monitoring",
                "Security events must be monitored and logged",
                ComplianceStandard.SOC2,
                ComplianceRegion.GLOBAL,
                "high",
                "security"
            ),
            ComplianceRule(
                "soc2_access_control",
                "Access Control",
                "Access to systems must be controlled and reviewed",
                ComplianceStandard.SOC2,
                ComplianceRegion.GLOBAL,
                "critical",
                "access_control"
            ),
        ])
        
        # ISO 27001 Rules
        self.rules.extend([
            ComplianceRule(
                "iso27001_risk_assessment",
                "Risk Assessment",
                "Regular risk assessments must be conducted",
                ComplianceStandard.ISO27001,
                ComplianceRegion.GLOBAL,
                "high",
                "risk_management"
            ),
            ComplianceRule(
                "iso27001_incident_response",
                "Incident Response",
                "Incident response procedures must be defined and tested",
                ComplianceStandard.ISO27001,
                ComplianceRegion.GLOBAL,
                "high",
                "incident_management"
            ),
        ])
    
    async def evaluate_compliance(self, system_state: Dict[str, Any], region: ComplianceRegion) -> Dict[str, Any]:
        """Evaluate system compliance against rules"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "region": region.value,
            "rules_evaluated": 0,
            "compliant": 0,
            "non_compliant": 0,
            "partially_compliant": 0,
            "not_applicable": 0,
            "violations": [],
            "recommendations": []
        }
        
        applicable_rules = [r for r in self.rules if r.region in [region, ComplianceRegion.GLOBAL] and r.enabled]
        
        for rule in applicable_rules:
            results["rules_evaluated"] += 1
            status = await self._evaluate_rule(rule, system_state)
            
            if status == ComplianceStatus.COMPLIANT:
                results["compliant"] += 1
            elif status == ComplianceStatus.NON_COMPLIANT:
                results["non_compliant"] += 1
                violation = ComplianceViolation(
                    id=f"violation_{rule.id}_{int(datetime.now().timestamp())}",
                    rule_id=rule.id,
                    timestamp=datetime.now(timezone.utc),
                    severity=rule.severity,
                    description=f"Non-compliance with {rule.name}: {rule.description}",
                    resource="system",
                    region=region,
                    remediation=self._get_remediation_advice(rule)
                )
                self.violations.append(violation)
                results["violations"].append(asdict(violation))
            elif status == ComplianceStatus.PARTIALLY_COMPLIANT:
                results["partially_compliant"] += 1
            else:
                results["not_applicable"] += 1
        
        # Generate recommendations
        results["recommendations"] = self._generate_compliance_recommendations(results)
        
        return results
    
    async def _evaluate_rule(self, rule: ComplianceRule, system_state: Dict[str, Any]) -> ComplianceStatus:
        """Evaluate a specific compliance rule"""
        # This is a simplified evaluation - in practice, each rule would have
        # specific evaluation logic based on system configuration and data
        
        if rule.id == "gdpr_data_encryption":
            # Check if encryption is enabled
            encryption_enabled = system_state.get("encryption", {}).get("enabled", False)
            return ComplianceStatus.COMPLIANT if encryption_enabled else ComplianceStatus.NON_COMPLIANT
        
        elif rule.id == "gdpr_access_logging":
            # Check if access logging is enabled
            logging_enabled = system_state.get("logging", {}).get("access_logs", False)
            return ComplianceStatus.COMPLIANT if logging_enabled else ComplianceStatus.NON_COMPLIANT
        
        elif rule.id == "soc2_security_monitoring":
            # Check if security monitoring is active
            monitoring_enabled = system_state.get("monitoring", {}).get("security", False)
            return ComplianceStatus.COMPLIANT if monitoring_enabled else ComplianceStatus.NON_COMPLIANT
        
        elif rule.id == "soc2_access_control":
            # Check if access controls are implemented
            access_control = system_state.get("access_control", {}).get("enabled", False)
            return ComplianceStatus.COMPLIANT if access_control else ComplianceStatus.NON_COMPLIANT
        
        # Default to needs review for unimplemented rules
        return ComplianceStatus.NEEDS_REVIEW
    
    def _get_remediation_advice(self, rule: ComplianceRule) -> str:
        """Get remediation advice for a rule violation"""
        remediation_map = {
            "gdpr_data_encryption": "Enable encryption for data at rest and in transit",
            "gdpr_access_logging": "Implement comprehensive access logging for all personal data access",
            "gdpr_consent_management": "Implement explicit consent mechanisms for data processing",
            "ccpa_data_disclosure": "Update privacy policy to include required data collection disclosures",
            "soc2_security_monitoring": "Enable security event monitoring and alerting",
            "soc2_access_control": "Implement role-based access controls and regular access reviews",
            "iso27001_risk_assessment": "Conduct regular risk assessments and document findings",
            "iso27001_incident_response": "Define and test incident response procedures",
        }
        
        return remediation_map.get(rule.id, "Review and address compliance requirements")
    
    def _generate_compliance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []
        
        if results["non_compliant"] > 0:
            recommendations.append("Address non-compliant items immediately to reduce regulatory risk")
        
        if results["partially_compliant"] > 0:
            recommendations.append("Complete partially compliant items to achieve full compliance")
        
        violation_count = len(results["violations"])
        if violation_count > 0:
            recommendations.append(f"Remediate {violation_count} compliance violations")
        
        # Severity-based recommendations
        critical_violations = [v for v in results["violations"] if v["severity"] == "critical"]
        if critical_violations:
            recommendations.append(f"URGENT: Address {len(critical_violations)} critical compliance violations")
        
        return recommendations

class MultiRegionManager:
    """Manages multi-region deployment and compliance"""
    
    def __init__(self):
        self.regions: Dict[str, Dict[str, Any]] = {
            ComplianceRegion.EU.value: {
                "name": "European Union",
                "primary_datacenter": "eu-west-1",
                "backup_datacenter": "eu-central-1",
                "compliance_standards": [ComplianceStandard.GDPR.value],
                "data_residency_required": True,
                "encryption_required": True,
                "supported_locales": ["en_GB", "de_DE", "fr_FR", "es_ES"]
            },
            ComplianceRegion.US.value: {
                "name": "United States",
                "primary_datacenter": "us-east-1",
                "backup_datacenter": "us-west-2",
                "compliance_standards": [ComplianceStandard.CCPA.value, ComplianceStandard.SOX.value],
                "data_residency_required": False,
                "encryption_required": True,
                "supported_locales": ["en_US"]
            },
            ComplianceRegion.APAC.value: {
                "name": "Asia-Pacific",
                "primary_datacenter": "ap-southeast-1",
                "backup_datacenter": "ap-northeast-1",
                "compliance_standards": [ComplianceStandard.PDPA.value],
                "data_residency_required": True,
                "encryption_required": True,
                "supported_locales": ["ja_JP", "zh_CN"]
            }
        }
    
    def get_region_config(self, region: ComplianceRegion) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific region"""
        return self.regions.get(region.value)
    
    def get_applicable_standards(self, region: ComplianceRegion) -> List[ComplianceStandard]:
        """Get applicable compliance standards for a region"""
        config = self.get_region_config(region)
        if not config:
            return []
        
        standards = []
        for standard_name in config.get("compliance_standards", []):
            try:
                standards.append(ComplianceStandard(standard_name))
            except ValueError:
                continue
        
        return standards
    
    def should_enforce_data_residency(self, region: ComplianceRegion) -> bool:
        """Check if data residency is required for a region"""
        config = self.get_region_config(region)
        return config.get("data_residency_required", False) if config else False

class GlobalComplianceEngine:
    """Main global compliance orchestrator"""
    
    def __init__(self):
        self.i18n_manager = InternationalizationManager()
        self.rule_engine = ComplianceRuleEngine()
        self.region_manager = MultiRegionManager()
        
        # Current system configuration
        self.system_config = {
            "encryption": {"enabled": True, "algorithm": "AES-256"},
            "logging": {"access_logs": True, "audit_logs": True, "retention_days": 90},
            "monitoring": {"security": True, "compliance": True, "alerts": True},
            "access_control": {"enabled": True, "rbac": True, "mfa": False},
            "data_protection": {"anonymization": True, "pseudonymization": True},
            "backup": {"enabled": True, "encrypted": True, "cross_region": True}
        }
    
    async def evaluate_global_compliance(self) -> Dict[str, Any]:
        """Evaluate compliance across all regions"""
        logger.info("Starting global compliance evaluation")
        
        global_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "compliant",
            "regions": {},
            "summary": {
                "total_rules": 0,
                "compliant": 0,
                "non_compliant": 0,
                "violations": 0,
                "critical_violations": 0
            },
            "recommendations": []
        }
        
        # Evaluate each region
        for region in [ComplianceRegion.EU, ComplianceRegion.US, ComplianceRegion.APAC]:
            region_results = await self.rule_engine.evaluate_compliance(self.system_config, region)
            global_results["regions"][region.value] = region_results
            
            # Update summary
            global_results["summary"]["total_rules"] += region_results["rules_evaluated"]
            global_results["summary"]["compliant"] += region_results["compliant"]
            global_results["summary"]["non_compliant"] += region_results["non_compliant"]
            global_results["summary"]["violations"] += len(region_results["violations"])
            
            # Count critical violations
            critical_violations = len([v for v in region_results["violations"] if v["severity"] == "critical"])
            global_results["summary"]["critical_violations"] += critical_violations
        
        # Determine overall status
        if global_results["summary"]["critical_violations"] > 0:
            global_results["overall_status"] = "critical_issues"
        elif global_results["summary"]["non_compliant"] > 0:
            global_results["overall_status"] = "non_compliant"
        elif global_results["summary"]["violations"] > 0:
            global_results["overall_status"] = "needs_attention"
        
        # Generate global recommendations
        global_results["recommendations"] = self._generate_global_recommendations(global_results)
        
        logger.info(f"Global compliance evaluation completed: {global_results['overall_status']}")
        
        return global_results
    
    def _generate_global_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate global compliance recommendations"""
        recommendations = []
        
        summary = results["summary"]
        
        if summary["critical_violations"] > 0:
            recommendations.append(f"URGENT: Address {summary['critical_violations']} critical compliance violations immediately")
        
        if summary["non_compliant"] > 0:
            recommendations.append(f"Fix {summary['non_compliant']} non-compliant items to meet regulatory requirements")
        
        # Region-specific recommendations
        for region_name, region_data in results["regions"].items():
            if region_data["non_compliant"] > 0:
                recommendations.append(f"Focus on {region_name} region compliance ({region_data['non_compliant']} issues)")
        
        # General recommendations
        recommendations.extend([
            "Implement automated compliance monitoring",
            "Schedule regular compliance audits",
            "Ensure staff training on compliance requirements",
            "Document compliance procedures and controls"
        ])
        
        return recommendations
    
    def get_localized_dashboard(self, locale: str = "en_US") -> Dict[str, Any]:
        """Get compliance dashboard localized for specific locale"""
        config = self.i18n_manager.get_locale_config(locale)
        if not config:
            locale = "en_US"
            config = self.i18n_manager.get_locale_config(locale)
        
        return {
            "locale": locale,
            "app_name": self.i18n_manager.get_translation("app.name", locale),
            "app_description": self.i18n_manager.get_translation("app.description", locale),
            "status_labels": {
                "compliant": self.i18n_manager.get_translation("compliance.status.compliant", locale),
                "non_compliant": self.i18n_manager.get_translation("compliance.status.non_compliant", locale)
            },
            "current_time": self.i18n_manager.format_date(datetime.now(), locale),
            "timezone": config.timezone,
            "currency": config.currency,
            "supported_regions": list(self.region_manager.regions.keys())
        }
    
    async def generate_compliance_report(self, region: Optional[ComplianceRegion] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if region:
            results = await self.rule_engine.evaluate_compliance(self.system_config, region)
            return {
                "region": region.value,
                "report_type": "regional",
                "results": results
            }
        else:
            results = await self.evaluate_global_compliance()
            return {
                "report_type": "global",
                "results": results
            }

# Example usage and testing
async def test_global_compliance():
    """Test the global compliance engine"""
    print("Global Compliance Engine Test")
    print("=" * 40)
    
    engine = GlobalComplianceEngine()
    
    # Test localization
    print("\nTesting Internationalization:")
    for locale in ["en_US", "de_DE", "fr_FR", "ja_JP", "zh_CN"]:
        dashboard = engine.get_localized_dashboard(locale)
        print(f"{locale}: {dashboard['app_name']}")
    
    # Test compliance evaluation
    print(f"\nEvaluating Global Compliance...")
    results = await engine.evaluate_global_compliance()
    
    print(f"Overall Status: {results['overall_status']}")
    print(f"Total Rules Evaluated: {results['summary']['total_rules']}")
    print(f"Compliant: {results['summary']['compliant']}")
    print(f"Non-Compliant: {results['summary']['non_compliant']}")
    print(f"Violations: {results['summary']['violations']}")
    print(f"Critical Violations: {results['summary']['critical_violations']}")
    
    # Show regional breakdown
    print(f"\nRegional Compliance Status:")
    for region, region_data in results["regions"].items():
        status = "✅" if region_data["non_compliant"] == 0 else "❌"
        print(f"{status} {region.upper()}: {region_data['compliant']}/{region_data['rules_evaluated']} rules compliant")
    
    # Show recommendations
    if results["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"{i}. {rec}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_global_compliance())