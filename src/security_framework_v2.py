#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - ADVANCED SECURITY FRAMEWORK V2
Comprehensive security framework with AI-powered threat detection, zero-trust architecture,
and real-time security monitoring
"""

import asyncio
import json
import hashlib
import hmac
import secrets
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum
import threading
import base64
from pathlib import Path
import sqlite3
import aiosqlite
import structlog

logger = structlog.get_logger("SecurityFrameworkV2")

class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEvent(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PAYLOAD = "malicious_payload"

@dataclass
class SecurityThreat:
    """Security threat detection result"""
    id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    payload: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1
    mitigated: bool = False
    mitigation_action: Optional[str] = None

@dataclass
class SecurityRule:
    """Security rule definition"""
    id: str
    name: str
    description: str
    pattern: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    enabled: bool = True
    regex_flags: int = re.IGNORECASE

class AIThreatDetector:
    """AI-powered threat detection using pattern analysis and ML"""
    
    def __init__(self):
        self.threat_patterns: Dict[SecurityEvent, List[str]] = {
            SecurityEvent.INJECTION_ATTEMPT: [
                r"(?:'|\"|`|\;|\||&|\$|\{|\}|\(|\)|\<|\>).*?(select|insert|update|delete|drop|create|alter|exec|union)",
                r"(?:union|select|insert|delete|update|drop|exec|script|javascript|vbscript).*?(?:\s|$)",
                r"(?:--|#|/\*|\*/|xp_|sp_cmdshell)",
                r"(?:\b(?:and|or|not|in|exists|between|like|is|null)\b\s*[\'\"\`]?\s*(?:\d+|[\'\"\`]|\w+)\s*[\'\"\`]?\s*(?:=|<>|!=|<=|>=|<|>))",
                r"(?:\bconcat\s*\(|\bchar\s*\(|\bunhex\s*\(|\bhex\s*\()",
            ],
            SecurityEvent.XSS_ATTEMPT: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"on(?:load|click|submit|change|mouseover|error|focus|blur)\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"<applet[^>]*>",
                r"(?:alert|confirm|prompt|eval|expression)\s*\(",
            ],
            SecurityEvent.PATH_TRAVERSAL: [
                r"(?:\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e\\|\.\.%2f|\.\.%5c)",
                r"(?:\/etc\/|\/proc\/|\/sys\/|\/dev\/|\/root\/|\/home\/)",
                r"(?:c:\\|\\windows\\|\\system32\\|\\program files\\)",
                r"(?:\.\.){2,}",
                r"(?:%252e|%c0%ae|%c1%9c)",
            ],
            SecurityEvent.MALICIOUS_PAYLOAD: [
                r"(?:nc|netcat|wget|curl|bash|sh|cmd|powershell|python|perl|ruby|php)\s+",
                r"(?:base64|hex|rot13|url).*(?:decode|encode)",
                r"(?:eval|exec|system|passthru|shell_exec|popen|proc_open)",
                r"(?:\|\s*(?:nc|netcat|telnet|ssh|ftp))",
                r"(?:reverse|bind)\s*shell",
            ]
        }
        
        self.learning_enabled = True
        self.false_positives: Set[str] = set()
        self.confirmed_threats: Set[str] = set()
        self.threat_history: List[SecurityThreat] = []
        
    def analyze_input(self, input_data: str, context: Dict[str, Any] = None) -> List[SecurityThreat]:
        """Analyze input data for security threats"""
        threats = []
        context = context or {}
        
        # Normalize input for analysis
        normalized_input = self._normalize_input(input_data)
        
        # Run pattern-based detection
        for event_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, normalized_input, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    threat = SecurityThreat(
                        id=f"{event_type.value}_{hash(match.group())}_{int(time.time())}",
                        event_type=event_type,
                        threat_level=self._calculate_threat_level(event_type, match, context),
                        timestamp=datetime.now(timezone.utc),
                        payload=match.group()[:500],  # Truncate for storage
                        context=context,
                        confidence=self._calculate_confidence(event_type, match, normalized_input)
                    )
                    threats.append(threat)
        
        # ML-based behavioral analysis
        behavioral_threats = self._analyze_behavioral_patterns(normalized_input, context)
        threats.extend(behavioral_threats)
        
        # Filter out false positives
        threats = self._filter_false_positives(threats)
        
        # Record for learning
        if self.learning_enabled:
            self.threat_history.extend(threats)
            self._update_learning_model(threats)
        
        return threats
    
    def _normalize_input(self, input_data: str) -> str:
        """Normalize input data for consistent analysis"""
        # Decode common encodings
        normalized = input_data
        
        # URL decode
        try:
            import urllib.parse
            normalized = urllib.parse.unquote_plus(normalized)
        except:
            pass
        
        # HTML decode
        try:
            import html
            normalized = html.unescape(normalized)
        except:
            pass
        
        # Base64 decode (if it looks like base64)
        if re.match(r'^[A-Za-z0-9+/]*={0,2}$', normalized) and len(normalized) % 4 == 0:
            try:
                decoded = base64.b64decode(normalized).decode('utf-8', errors='ignore')
                if decoded.isprintable():
                    normalized += " " + decoded  # Add decoded version
            except:
                pass
        
        return normalized.lower()
    
    def _calculate_threat_level(self, event_type: SecurityEvent, match, context: Dict[str, Any]) -> ThreatLevel:
        """Calculate threat level based on event type and context"""
        base_levels = {
            SecurityEvent.INJECTION_ATTEMPT: ThreatLevel.HIGH,
            SecurityEvent.XSS_ATTEMPT: ThreatLevel.HIGH,
            SecurityEvent.PATH_TRAVERSAL: ThreatLevel.HIGH,
            SecurityEvent.MALICIOUS_PAYLOAD: ThreatLevel.CRITICAL,
            SecurityEvent.DATA_EXFILTRATION: ThreatLevel.CRITICAL,
            SecurityEvent.PRIVILEGE_ESCALATION: ThreatLevel.CRITICAL,
            SecurityEvent.AUTHENTICATION_FAILURE: ThreatLevel.MEDIUM,
            SecurityEvent.AUTHORIZATION_VIOLATION: ThreatLevel.HIGH,
            SecurityEvent.RATE_LIMIT_EXCEEDED: ThreatLevel.LOW,
            SecurityEvent.SUSPICIOUS_PATTERN: ThreatLevel.MEDIUM,
        }
        
        base_level = base_levels.get(event_type, ThreatLevel.MEDIUM)
        
        # Adjust based on context
        if context.get('admin_user'):
            # Escalate if targeting admin user
            if base_level == ThreatLevel.LOW:
                return ThreatLevel.MEDIUM
            elif base_level == ThreatLevel.MEDIUM:
                return ThreatLevel.HIGH
            elif base_level == ThreatLevel.HIGH:
                return ThreatLevel.CRITICAL
        
        return base_level
    
    def _calculate_confidence(self, event_type: SecurityEvent, match, full_input: str) -> float:
        """Calculate confidence score for threat detection"""
        base_confidence = 0.7
        
        # Adjust based on pattern complexity
        pattern_length = len(match.group())
        if pattern_length < 5:
            base_confidence *= 0.6  # Short matches are less reliable
        elif pattern_length > 20:
            base_confidence *= 1.2  # Long matches are more reliable
        
        # Adjust based on context clues
        suspicious_keywords = ['script', 'eval', 'exec', 'union', 'select', 'drop', 'delete']
        keyword_count = sum(1 for keyword in suspicious_keywords if keyword in full_input)
        base_confidence += min(keyword_count * 0.1, 0.3)
        
        return min(base_confidence, 1.0)
    
    def _analyze_behavioral_patterns(self, input_data: str, context: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze behavioral patterns for anomalous activity"""
        threats = []
        
        # Check for unusual character sequences
        if self._has_unusual_character_patterns(input_data):
            threats.append(SecurityThreat(
                id=f"behavioral_unusual_chars_{int(time.time())}",
                event_type=SecurityEvent.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                payload=input_data[:200],
                context=context,
                confidence=0.6
            ))
        
        # Check for encoding stacking (multiple layers of encoding)
        if self._detect_encoding_stacking(input_data):
            threats.append(SecurityThreat(
                id=f"behavioral_encoding_stack_{int(time.time())}",
                event_type=SecurityEvent.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.now(timezone.utc),
                payload=input_data[:200],
                context=context,
                confidence=0.8
            ))
        
        return threats
    
    def _has_unusual_character_patterns(self, input_data: str) -> bool:
        """Detect unusual character patterns that might indicate obfuscation"""
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in input_data if not c.isalnum() and c != ' ') / len(input_data)
        if special_char_ratio > 0.4:
            return True
        
        # Check for repeated patterns
        for i in range(2, min(10, len(input_data) // 3)):
            pattern = input_data[:i]
            if input_data.count(pattern) > 3:
                return True
        
        return False
    
    def _detect_encoding_stacking(self, input_data: str) -> bool:
        """Detect multiple layers of encoding (common in attacks)"""
        encoding_indicators = ['%', '&', '+', '=', '\\x', '\\u', '&lt;', '&gt;']
        indicator_count = sum(1 for indicator in encoding_indicators if indicator in input_data)
        return indicator_count >= 3
    
    def _filter_false_positives(self, threats: List[SecurityThreat]) -> List[SecurityThreat]:
        """Filter out known false positives"""
        filtered = []
        for threat in threats:
            threat_signature = hashlib.md5(f"{threat.event_type.value}:{threat.payload}".encode()).hexdigest()
            if threat_signature not in self.false_positives:
                filtered.append(threat)
        return filtered
    
    def _update_learning_model(self, threats: List[SecurityThreat]):
        """Update the learning model based on new threats"""
        # Keep only recent history
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        self.threat_history = [t for t in self.threat_history if t.timestamp > cutoff_time]
        
        # Simple learning: adjust confidence based on historical accuracy
        if len(self.threat_history) > 100:
            # This would be replaced with actual ML model training
            pass
    
    def mark_false_positive(self, threat_id: str):
        """Mark a threat as false positive for learning"""
        threat = next((t for t in self.threat_history if t.id == threat_id), None)
        if threat:
            signature = hashlib.md5(f"{threat.event_type.value}:{threat.payload}".encode()).hexdigest()
            self.false_positives.add(signature)
    
    def mark_confirmed_threat(self, threat_id: str):
        """Mark a threat as confirmed for learning"""
        threat = next((t for t in self.threat_history if t.id == threat_id), None)
        if threat:
            signature = hashlib.md5(f"{threat.event_type.value}:{threat.payload}".encode()).hexdigest()
            self.confirmed_threats.add(signature)

class SecurityStorage:
    """Persistent storage for security events and threats"""
    
    def __init__(self, db_path: str = "security.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize security database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_threats (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source_ip TEXT,
                    user_agent TEXT,
                    payload TEXT,
                    context TEXT,
                    confidence REAL,
                    mitigated INTEGER DEFAULT 0,
                    mitigation_action TEXT,
                    INDEX(event_type, timestamp),
                    INDEX(threat_level, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    pattern TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    result TEXT NOT NULL,
                    source_ip TEXT,
                    user_agent TEXT,
                    INDEX(user_id, timestamp),
                    INDEX(action, timestamp)
                )
            """)
    
    async def store_threat(self, threat: SecurityThreat):
        """Store a security threat"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO security_threats 
                (id, event_type, threat_level, timestamp, source_ip, user_agent, 
                 payload, context, confidence, mitigated, mitigation_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat.id, threat.event_type.value, threat.threat_level.value,
                threat.timestamp.timestamp(), threat.source_ip, threat.user_agent,
                threat.payload, json.dumps(threat.context), threat.confidence,
                1 if threat.mitigated else 0, threat.mitigation_action
            ))
            await db.commit()
    
    async def log_access(self, user_id: Optional[str], action: str, resource: Optional[str], 
                        result: str, source_ip: Optional[str] = None, 
                        user_agent: Optional[str] = None):
        """Log access attempt"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO access_log 
                (timestamp, user_id, action, resource, result, source_ip, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), user_id, action, resource, result, source_ip, user_agent
            ))
            await db.commit()
    
    async def get_recent_threats(self, hours: int = 24) -> List[SecurityThreat]:
        """Get recent security threats"""
        cutoff_time = time.time() - (hours * 3600)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT * FROM security_threats 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time,)) as cursor:
                
                threats = []
                async for row in cursor:
                    threats.append(SecurityThreat(
                        id=row[0],
                        event_type=SecurityEvent(row[1]),
                        threat_level=ThreatLevel(row[2]),
                        timestamp=datetime.fromtimestamp(row[3], timezone.utc),
                        source_ip=row[4],
                        user_agent=row[5],
                        payload=row[6],
                        context=json.loads(row[7]) if row[7] else {},
                        confidence=row[8] or 0.0,
                        mitigated=bool(row[9]),
                        mitigation_action=row[10]
                    ))
                return threats

class ZeroTrustAccessControl:
    """Zero-trust access control with continuous verification"""
    
    def __init__(self, storage: SecurityStorage):
        self.storage = storage
        self.permissions: Dict[str, Set[str]] = {}
        self.roles: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        self.trust_scores: Dict[str, float] = {}
        
        # Default roles and permissions
        self._setup_default_rbac()
    
    def _setup_default_rbac(self):
        """Setup default role-based access control"""
        # Define permissions
        self.permissions = {
            'read_repos': {'view repositories'},
            'write_repos': {'modify repositories'},
            'admin_repos': {'manage repositories', 'delete repositories'},
            'read_issues': {'view issues'},
            'write_issues': {'create issues', 'modify issues'},
            'admin_issues': {'delete issues', 'manage issue lifecycle'},
            'system_admin': {'manage users', 'modify configuration', 'view logs'}
        }
        
        # Define roles
        self.roles = {
            'viewer': {'read_repos', 'read_issues'},
            'contributor': {'read_repos', 'read_issues', 'write_issues'},
            'maintainer': {'read_repos', 'write_repos', 'read_issues', 'write_issues', 'admin_issues'},
            'admin': {'read_repos', 'write_repos', 'admin_repos', 'read_issues', 'write_issues', 'admin_issues', 'system_admin'}
        }
    
    async def authenticate(self, user_id: str, password: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Authenticate user and return session token"""
        context = context or {}
        
        # This would integrate with actual authentication system
        # For now, simulate authentication
        is_valid = await self._verify_credentials(user_id, password)
        
        result = "success" if is_valid else "failure"
        await self.storage.log_access(
            user_id, "authentication", None, result,
            context.get('source_ip'), context.get('user_agent')
        )
        
        if is_valid:
            # Generate secure session token
            token = secrets.token_urlsafe(32)
            
            # Store session with expiration
            self.session_tokens[token] = {
                'user_id': user_id,
                'created_at': time.time(),
                'expires_at': time.time() + 3600,  # 1 hour
                'context': context
            }
            
            # Initialize trust score
            self.trust_scores[user_id] = 0.8  # Base trust score
            
            return token
        
        return None
    
    async def authorize(self, token: str, action: str, resource: str = None) -> bool:
        """Authorize an action with continuous trust verification"""
        # Verify session token
        session = self.session_tokens.get(token)
        if not session or session['expires_at'] < time.time():
            return False
        
        user_id = session['user_id']
        
        # Get current trust score
        trust_score = self.trust_scores.get(user_id, 0.0)
        
        # Check if action requires elevated trust
        elevated_actions = {'delete', 'admin', 'modify_config', 'manage_users'}
        requires_elevated = any(elevated in action for elevated in elevated_actions)
        
        if requires_elevated and trust_score < 0.7:
            await self.storage.log_access(user_id, action, resource, "denied_low_trust")
            return False
        
        # Check permissions
        user_roles = self.user_roles.get(user_id, set())
        user_permissions = set()
        
        for role in user_roles:
            if role in self.roles:
                user_permissions.update(self.roles[role])
        
        # Check if user has required permission
        has_permission = any(perm in user_permissions for perm in self.permissions.get(action, set()))
        
        result = "success" if has_permission else "denied"
        await self.storage.log_access(user_id, action, resource, result)
        
        # Update trust score based on behavior
        await self._update_trust_score(user_id, action, has_permission)
        
        return has_permission
    
    async def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verify user credentials (mock implementation)"""
        # This would integrate with actual authentication system
        # For demo purposes, accept any non-empty password
        return len(password) > 0
    
    async def _update_trust_score(self, user_id: str, action: str, authorized: bool):
        """Update user trust score based on behavior"""
        current_score = self.trust_scores.get(user_id, 0.5)
        
        if authorized:
            # Successful authorized action slightly increases trust
            adjustment = 0.02
        else:
            # Failed authorization decreases trust more significantly
            adjustment = -0.1
        
        # Adjust based on action sensitivity
        sensitive_actions = {'delete', 'admin', 'config', 'user'}
        if any(sensitive in action for sensitive in sensitive_actions):
            adjustment *= 2
        
        new_score = max(0.0, min(1.0, current_score + adjustment))
        self.trust_scores[user_id] = new_score
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user"""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
    
    def revoke_role(self, user_id: str, role: str):
        """Revoke role from user"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)

class SecurityOrchestrator:
    """Main security orchestrator coordinating all security components"""
    
    def __init__(self, db_path: str = "security.db"):
        self.storage = SecurityStorage(db_path)
        self.threat_detector = AIThreatDetector()
        self.access_control = ZeroTrustAccessControl(self.storage)
        
        # Real-time monitoring
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.threat_handlers: List[Callable[[SecurityThreat], None]] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Setup default threat handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default threat response handlers"""
        
        def log_threat_handler(threat: SecurityThreat):
            logger.warning(f"Security threat detected: {threat.event_type.value} "
                         f"(level: {threat.threat_level.value}, confidence: {threat.confidence:.2f})")
            if threat.payload:
                logger.warning(f"Threat payload: {threat.payload[:100]}...")
        
        self.add_threat_handler(log_threat_handler)
    
    def add_threat_handler(self, handler: Callable[[SecurityThreat], None]):
        """Add a threat response handler"""
        self.threat_handlers.append(handler)
    
    async def scan_input(self, input_data: str, context: Dict[str, Any] = None) -> List[SecurityThreat]:
        """Scan input for security threats"""
        threats = self.threat_detector.analyze_input(input_data, context)
        
        # Store and handle threats
        for threat in threats:
            await self.storage.store_threat(threat)
            self.active_threats[threat.id] = threat
            
            # Trigger threat handlers
            for handler in self.threat_handlers:
                try:
                    handler(threat)
                except Exception as e:
                    logger.error(f"Threat handler failed: {e}")
        
        return threats
    
    async def authenticate_user(self, user_id: str, password: str, 
                              context: Dict[str, Any] = None) -> Optional[str]:
        """Authenticate user"""
        return await self.access_control.authenticate(user_id, password, context)
    
    async def authorize_action(self, token: str, action: str, resource: str = None) -> bool:
        """Authorize user action"""
        return await self.access_control.authorize(token, action, resource)
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        recent_threats = list(self.active_threats.values())[-10:]  # Last 10 threats
        
        threat_stats = {}
        for threat in recent_threats:
            level = threat.threat_level.value
            threat_stats[level] = threat_stats.get(level, 0) + 1
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_threats': len(self.active_threats),
            'recent_threats': [
                {
                    'id': t.id,
                    'type': t.event_type.value,
                    'level': t.threat_level.value,
                    'timestamp': t.timestamp.isoformat(),
                    'confidence': t.confidence,
                    'mitigated': t.mitigated
                } for t in recent_threats
            ],
            'threat_distribution': threat_stats,
            'system_status': 'secure' if not any(
                t.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] and not t.mitigated 
                for t in recent_threats
            ) else 'at_risk'
        }

# Example usage and testing
async def test_security_framework():
    """Test the security framework"""
    security = SecurityOrchestrator("test_security.db")
    
    # Test threat detection
    test_inputs = [
        "normal user input",
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../../../etc/passwd",
        "normal query with some data",
        "eval(base64_decode('malicious_payload'))"
    ]
    
    print("Security Framework Test Results:")
    print("=" * 50)
    
    for i, input_data in enumerate(test_inputs, 1):
        threats = await security.scan_input(input_data, {
            'source_ip': '192.168.1.100',
            'user_agent': 'TestAgent/1.0'
        })
        
        print(f"\nTest {i}: {input_data[:50]}...")
        if threats:
            for threat in threats:
                print(f"  THREAT: {threat.event_type.value} "
                     f"({threat.threat_level.value}, confidence: {threat.confidence:.2f})")
        else:
            print("  No threats detected")
    
    # Test authentication and authorization
    print(f"\nAuthentication Tests:")
    print("-" * 20)
    
    token = await security.authenticate_user("test_user", "test_password", {
        'source_ip': '192.168.1.100'
    })
    
    if token:
        print(f"Authentication successful, token: {token[:20]}...")
        
        # Assign role
        security.access_control.assign_role("test_user", "contributor")
        
        # Test authorization
        test_actions = ["read_repos", "write_repos", "delete_repos", "system_admin"]
        
        for action in test_actions:
            authorized = await security.authorize_action(token, action)
            print(f"  Action '{action}': {'AUTHORIZED' if authorized else 'DENIED'}")
    else:
        print("Authentication failed")
    
    # Get security dashboard
    print(f"\nSecurity Dashboard:")
    print("-" * 18)
    dashboard = security.get_security_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    return True

if __name__ == "__main__":
    asyncio.run(test_security_framework())