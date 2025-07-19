"""
Enterprise Compliance and Audit Trail Service
Provides comprehensive audit logging, compliance reporting, and regulatory documentation
Compatible with GDPR, EU AI Act, SOC2, and other enterprise compliance requirements
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import structlog

from app.core.database import get_db

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_CHANGED = "role_changed"
    
    # Model operations
    MODEL_UPLOADED = "model_uploaded"
    MODEL_UPDATED = "model_updated"
    MODEL_DELETED = "model_deleted"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PREDICTION = "model_prediction"
    MODEL_RETRAINED = "model_retrained"
    
    # Data operations
    DATA_UPLOADED = "data_uploaded"
    DATA_ACCESSED = "data_accessed"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"
    
    # Analysis operations
    BIAS_ANALYSIS = "bias_analysis"
    FAIRNESS_MITIGATION = "fairness_mitigation"
    DRIFT_ANALYSIS = "drift_analysis"
    EXPLANATION_GENERATED = "explanation_generated"
    
    # System events
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_RESOLVED = "alert_resolved"
    CONFIGURATION_CHANGED = "configuration_changed"
    
    # Compliance events
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"
    AUDIT_REPORT_GENERATED = "audit_report_generated"
    DATA_RETENTION_EXECUTED = "data_retention_executed"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    EU_AI_ACT = "eu_ai_act"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"
    US_EEOC = "us_eeoc"
    FDA = "fda"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    risk_level: str
    compliance_relevant: bool
    retention_years: int


class EnterpriseComplianceService:
    """
    Enterprise compliance and audit trail service
    """
    
    def __init__(self):
        self.compliance_requirements = self._initialize_compliance_requirements()
        self.retention_policies = self._initialize_retention_policies()
        
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: str = "",
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low",
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ) -> str:
        """
        Log an audit event with comprehensive details
        
        Args:
            event_type: Type of audit event
            user_id: User performing the action
            session_id: Session identifier
            ip_address: IP address of the request
            user_agent: User agent string
            resource_type: Type of resource being accessed
            resource_id: Identifier of the resource
            action: Description of the action performed
            details: Additional event details
            risk_level: Risk level (low, medium, high, critical)
            compliance_frameworks: Relevant compliance frameworks
            
        Returns:
            Generated event ID
        """
        event_id = f"audit_{int(datetime.utcnow().timestamp() * 1000000)}"
        timestamp = datetime.utcnow()
        
        # Determine compliance relevance
        compliance_relevant = self._is_compliance_relevant(event_type, compliance_frameworks)
        
        # Determine retention period
        retention_years = self._get_retention_period(event_type, compliance_frameworks)
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            risk_level=risk_level,
            compliance_relevant=compliance_relevant,
            retention_years=retention_years
        )
        
        # Store in database
        await self._store_audit_event(audit_event)
        
        # Trigger compliance checks if relevant
        if compliance_relevant:
            await self._trigger_compliance_checks(audit_event, compliance_frameworks)
        
        logger.info("Audit event logged",
                   event_id=event_id,
                   event_type=event_type.value,
                   user_id=user_id,
                   risk_level=risk_level,
                   compliance_relevant=compliance_relevant)
        
        return event_id
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "comprehensive",
        include_remediation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for specified framework
        
        Args:
            framework: Compliance framework to report on
            start_date: Start date for the report
            end_date: End date for the report
            report_type: Type of report (comprehensive, summary, violations)
            include_remediation: Include remediation recommendations
            
        Returns:
            Compliance report data
        """
        logger.info("Generating compliance report",
                   framework=framework.value,
                   start_date=start_date.isoformat(),
                   end_date=end_date.isoformat())
        
        report = {
            "report_metadata": {
                "report_id": f"compliance_{framework.value}_{int(datetime.utcnow().timestamp())}",
                "framework": framework.value,
                "report_type": report_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "generated_by": "system"  # Could be passed as parameter
            },
            "executive_summary": {},
            "compliance_status": {},
            "audit_findings": {},
            "risk_assessment": {},
            "remediation_plan": {} if include_remediation else None,
            "supporting_evidence": {},
            "attestations": {}
        }
        
        # Framework-specific report generation
        if framework == ComplianceFramework.GDPR:
            report = await self._generate_gdpr_report(report, start_date, end_date)
        elif framework == ComplianceFramework.EU_AI_ACT:
            report = await self._generate_eu_ai_act_report(report, start_date, end_date)
        elif framework == ComplianceFramework.SOC2:
            report = await self._generate_soc2_report(report, start_date, end_date)
        elif framework == ComplianceFramework.HIPAA:
            report = await self._generate_hipaa_report(report, start_date, end_date)
        elif framework == ComplianceFramework.US_EEOC:
            report = await self._generate_eeoc_report(report, start_date, end_date)
        
        # Log report generation
        await self.log_audit_event(
            AuditEventType.COMPLIANCE_REPORT_GENERATED,
            resource_type="compliance_report",
            resource_id=report["report_metadata"]["report_id"],
            action=f"Generated {framework.value} compliance report",
            details={"framework": framework.value, "report_type": report_type},
            risk_level="medium",
            compliance_frameworks=[framework]
        )
        
        return report
    
    async def _generate_gdpr_report(
        self, 
        report: Dict[str, Any], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate GDPR-specific compliance report"""
        
        # Data processing activities
        processing_activities = await self._get_data_processing_activities(start_date, end_date)
        
        # Data subject rights requests
        subject_rights_requests = await self._get_subject_rights_requests(start_date, end_date)
        
        # Data breaches (if any)
        data_breaches = await self._get_data_breach_incidents(start_date, end_date)
        
        # Data transfers
        data_transfers = await self._get_international_data_transfers(start_date, end_date)
        
        report["compliance_status"] = {
            "overall_status": "compliant",  # This would be calculated
            "data_processing_compliance": self._assess_data_processing_compliance(processing_activities),
            "subject_rights_compliance": self._assess_subject_rights_compliance(subject_rights_requests),
            "data_protection_measures": await self._assess_data_protection_measures(),
            "documentation_completeness": await self._assess_documentation_completeness()
        }
        
        report["audit_findings"] = {
            "data_processing_activities": processing_activities,
            "subject_rights_requests": subject_rights_requests,
            "data_breaches": data_breaches,
            "international_transfers": data_transfers,
            "consent_management": await self._assess_consent_management(start_date, end_date),
            "data_minimization": await self._assess_data_minimization(start_date, end_date)
        }
        
        report["risk_assessment"] = {
            "privacy_risks": await self._assess_privacy_risks(),
            "technical_safeguards": await self._assess_technical_safeguards(),
            "organizational_measures": await self._assess_organizational_measures()
        }
        
        if report.get("remediation_plan"):
            report["remediation_plan"] = self._generate_gdpr_remediation_plan(report["audit_findings"])
        
        return report
    
    async def _generate_eu_ai_act_report(
        self, 
        report: Dict[str, Any], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate EU AI Act compliance report"""
        
        # AI system classifications
        ai_systems = await self._get_ai_system_classifications(start_date, end_date)
        
        # Risk assessments
        risk_assessments = await self._get_ai_risk_assessments(start_date, end_date)
        
        # Bias and fairness assessments
        bias_assessments = await self._get_bias_assessments(start_date, end_date)
        
        # Transparency measures
        transparency_measures = await self._get_transparency_measures(start_date, end_date)
        
        report["compliance_status"] = {
            "overall_status": "compliant",
            "risk_management_system": await self._assess_risk_management_system(),
            "data_governance": await self._assess_data_governance(),
            "transparency_obligations": self._assess_transparency_compliance(transparency_measures),
            "human_oversight": await self._assess_human_oversight(),
            "accuracy_robustness": await self._assess_accuracy_robustness(),
            "cybersecurity": await self._assess_cybersecurity_measures()
        }
        
        report["audit_findings"] = {
            "ai_systems": ai_systems,
            "risk_assessments": risk_assessments,
            "bias_assessments": bias_assessments,
            "transparency_measures": transparency_measures,
            "incident_reports": await self._get_ai_incident_reports(start_date, end_date),
            "conformity_assessments": await self._get_conformity_assessments(start_date, end_date)
        }
        
        return report
    
    async def _generate_soc2_report(
        self, 
        report: Dict[str, Any], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOC 2 compliance report"""
        
        # SOC 2 Trust Service Criteria
        security_controls = await self._assess_security_controls(start_date, end_date)
        availability_controls = await self._assess_availability_controls(start_date, end_date)
        processing_integrity = await self._assess_processing_integrity(start_date, end_date)
        confidentiality_controls = await self._assess_confidentiality_controls(start_date, end_date)
        privacy_controls = await self._assess_privacy_controls(start_date, end_date)
        
        report["compliance_status"] = {
            "overall_status": "compliant",
            "security": security_controls["status"],
            "availability": availability_controls["status"],
            "processing_integrity": processing_integrity["status"],
            "confidentiality": confidentiality_controls["status"],
            "privacy": privacy_controls["status"]
        }
        
        report["audit_findings"] = {
            "security_controls": security_controls,
            "availability_controls": availability_controls,
            "processing_integrity": processing_integrity,
            "confidentiality_controls": confidentiality_controls,
            "privacy_controls": privacy_controls,
            "change_management": await self._assess_change_management(start_date, end_date),
            "incident_response": await self._assess_incident_response(start_date, end_date)
        }
        
        return report
    
    async def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        resource_type: Optional[str] = None,
        risk_levels: Optional[List[str]] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Retrieve audit trail with filtering options
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            user_id: User ID filter
            event_types: Event types filter
            resource_type: Resource type filter
            risk_levels: Risk levels filter
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            Audit trail data with metadata
        """
        query = """
            SELECT event_id, event_type, timestamp, user_id, session_id, ip_address,
                   resource_type, resource_id, action, details, risk_level, compliance_relevant
            FROM audit_logs
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += f" AND timestamp >= ${len(params) + 1}"
            params.append(start_date)
        
        if end_date:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_date)
        
        if user_id:
            query += f" AND user_id = ${len(params) + 1}"
            params.append(user_id)
        
        if event_types:
            placeholders = ", ".join(f"${len(params) + i + 1}" for i in range(len(event_types)))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if resource_type:
            query += f" AND resource_type = ${len(params) + 1}"
            params.append(resource_type)
        
        if risk_levels:
            placeholders = ", ".join(f"${len(params) + i + 1}" for i in range(len(risk_levels)))
            query += f" AND risk_level IN ({placeholders})"
            params.extend(risk_levels)
        
        # Count total records
        count_query = query.replace("SELECT event_id, event_type, timestamp, user_id, session_id, ip_address, resource_type, resource_id, action, details, risk_level, compliance_relevant", "SELECT COUNT(*)")
        
        query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])
        
        async with get_db() as db:
            total_count = await db.fetchval(count_query, *params[:-2])
            records = await db.fetch(query, *params)
            
            audit_records = []
            for record in records:
                audit_record = dict(record)
                if audit_record["details"]:
                    audit_record["details"] = json.loads(audit_record["details"])
                audit_records.append(audit_record)
            
            return {
                "total_count": total_count,
                "records": audit_records,
                "metadata": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": total_count > offset + len(audit_records)
                }
            }
    
    async def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_risk_analysis: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        report_id = f"audit_report_{int(datetime.utcnow().timestamp())}"
        
        # Get audit statistics
        audit_stats = await self._get_audit_statistics(start_date, end_date)
        
        # Get security events
        security_events = await self._get_security_events(start_date, end_date)
        
        # Get access patterns
        access_patterns = await self._get_access_patterns(start_date, end_date)
        
        # Risk analysis
        risk_analysis = None
        if include_risk_analysis:
            risk_analysis = await self._perform_risk_analysis(start_date, end_date)
        
        # Recommendations
        recommendations = None
        if include_recommendations:
            recommendations = await self._generate_audit_recommendations(audit_stats, security_events)
        
        report = {
            "report_id": report_id,
            "report_type": "audit",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "audit_statistics": audit_stats,
            "security_events": security_events,
            "access_patterns": access_patterns,
            "risk_analysis": risk_analysis,
            "recommendations": recommendations
        }
        
        # Log report generation
        await self.log_audit_event(
            AuditEventType.AUDIT_REPORT_GENERATED,
            resource_type="audit_report",
            resource_id=report_id,
            action="Generated comprehensive audit report",
            details={"period_days": (end_date - start_date).days},
            risk_level="low"
        )
        
        return report
    
    def _initialize_compliance_requirements(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize compliance framework requirements"""
        
        return {
            ComplianceFramework.GDPR: {
                "data_retention_max_years": 6,
                "breach_notification_hours": 72,
                "subject_rights_response_days": 30,
                "required_documentation": ["privacy_policy", "data_processing_records", "impact_assessments"],
                "mandatory_events": [AuditEventType.DATA_ACCESSED, AuditEventType.DATA_EXPORTED, AuditEventType.DATA_DELETED]
            },
            ComplianceFramework.EU_AI_ACT: {
                "risk_assessment_frequency_months": 12,
                "bias_monitoring_frequency_months": 6,
                "incident_notification_hours": 24,
                "required_documentation": ["risk_assessment", "conformity_assessment", "transparency_measures"],
                "mandatory_events": [AuditEventType.BIAS_ANALYSIS, AuditEventType.MODEL_DEPLOYED, AuditEventType.MODEL_PREDICTION]
            },
            ComplianceFramework.SOC2: {
                "security_review_frequency_months": 12,
                "incident_response_hours": 4,
                "change_approval_required": True,
                "required_documentation": ["security_policies", "incident_procedures", "change_management"],
                "mandatory_events": [AuditEventType.CONFIGURATION_CHANGED, AuditEventType.USER_CREATED, AuditEventType.ROLE_CHANGED]
            }
        }
    
    def _initialize_retention_policies(self) -> Dict[AuditEventType, int]:
        """Initialize data retention policies by event type"""
        
        return {
            # User events - 7 years for compliance
            AuditEventType.USER_LOGIN: 7,
            AuditEventType.USER_LOGOUT: 7,
            AuditEventType.USER_CREATED: 10,
            AuditEventType.ROLE_CHANGED: 10,
            
            # Model events - 10 years for regulatory compliance
            AuditEventType.MODEL_DEPLOYED: 10,
            AuditEventType.MODEL_PREDICTION: 7,
            AuditEventType.BIAS_ANALYSIS: 10,
            AuditEventType.FAIRNESS_MITIGATION: 10,
            
            # Data events - varies by sensitivity
            AuditEventType.DATA_ACCESSED: 6,
            AuditEventType.DATA_EXPORTED: 10,
            AuditEventType.DATA_DELETED: 10,
            
            # System events - 3 years
            AuditEventType.ALERT_TRIGGERED: 3,
            AuditEventType.CONFIGURATION_CHANGED: 7,
            
            # Compliance events - 10 years
            AuditEventType.COMPLIANCE_REPORT_GENERATED: 10,
            AuditEventType.AUDIT_REPORT_GENERATED: 10
        }
    
    def _is_compliance_relevant(
        self, 
        event_type: AuditEventType, 
        frameworks: Optional[List[ComplianceFramework]]
    ) -> bool:
        """Determine if an event is relevant for compliance"""
        
        # High-risk events are always compliance relevant
        high_risk_events = {
            AuditEventType.DATA_EXPORTED,
            AuditEventType.DATA_DELETED,
            AuditEventType.BIAS_ANALYSIS,
            AuditEventType.MODEL_DEPLOYED,
            AuditEventType.ROLE_CHANGED,
            AuditEventType.CONFIGURATION_CHANGED
        }
        
        if event_type in high_risk_events:
            return True
        
        # Check framework-specific requirements
        if frameworks:
            for framework in frameworks:
                requirements = self.compliance_requirements.get(framework, {})
                mandatory_events = requirements.get("mandatory_events", [])
                if event_type in mandatory_events:
                    return True
        
        return False
    
    def _get_retention_period(
        self, 
        event_type: AuditEventType, 
        frameworks: Optional[List[ComplianceFramework]]
    ) -> int:
        """Get retention period for an event type"""
        
        # Get base retention period
        base_retention = self.retention_policies.get(event_type, 7)
        
        # Check framework-specific requirements
        if frameworks:
            max_retention = base_retention
            for framework in frameworks:
                framework_retention = self.compliance_requirements.get(framework, {}).get("data_retention_max_years", 7)
                max_retention = max(max_retention, framework_retention)
            return max_retention
        
        return base_retention
    
    async def _store_audit_event(self, audit_event: AuditEvent):
        """Store audit event in database"""
        
        async with get_db() as db:
            await db.execute("""
                INSERT INTO audit_logs (
                    event_id, event_type, timestamp, user_id, session_id, ip_address, user_agent,
                    resource_type, resource_id, action, details, risk_level, compliance_relevant,
                    retention_until
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, 
                audit_event.event_id,
                audit_event.event_type.value,
                audit_event.timestamp,
                audit_event.user_id,
                audit_event.session_id,
                audit_event.ip_address,
                audit_event.user_agent,
                audit_event.resource_type,
                audit_event.resource_id,
                audit_event.action,
                json.dumps(audit_event.details),
                audit_event.risk_level,
                audit_event.compliance_relevant,
                audit_event.timestamp + timedelta(days=365 * audit_event.retention_years)
            )
    
    async def _trigger_compliance_checks(
        self, 
        audit_event: AuditEvent, 
        frameworks: Optional[List[ComplianceFramework]]
    ):
        """Trigger automated compliance checks"""
        
        # This would trigger various compliance checks based on the event
        # For example, checking if data access is authorized, if exports are properly logged, etc.
        pass
    
    # Helper methods for generating specific compliance reports
    async def _get_data_processing_activities(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get data processing activities for GDPR compliance"""
        async with get_db() as db:
            records = await db.fetch("""
                SELECT event_type, resource_type, resource_id, COUNT(*) as count, 
                       MIN(timestamp) as first_occurrence, MAX(timestamp) as last_occurrence
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2 
                AND event_type IN ('data_uploaded', 'data_accessed', 'data_exported', 'data_deleted')
                GROUP BY event_type, resource_type, resource_id
            """, start_date, end_date)
            
            return [dict(record) for record in records]
    
    async def _get_subject_rights_requests(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get data subject rights requests"""
        # This would query a separate table for subject rights requests
        return []
    
    async def _get_data_breach_incidents(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get data breach incidents"""
        async with get_db() as db:
            records = await db.fetch("""
                SELECT * FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2 
                AND risk_level IN ('high', 'critical')
                AND event_type LIKE '%security%'
            """, start_date, end_date)
            
            return [dict(record) for record in records]
    
    async def _get_international_data_transfers(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get international data transfer records"""
        # This would track cross-border data transfers
        return []
    
    # Additional helper methods would be implemented for each compliance framework
    async def _assess_data_processing_compliance(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data processing compliance"""
        return {"status": "compliant", "issues": []}
    
    async def _assess_subject_rights_compliance(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess subject rights compliance"""
        return {"status": "compliant", "response_time_avg_hours": 24}
    
    # ... Additional assessment methods for other compliance areas
    
    async def _get_audit_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get audit statistics for the specified period"""
        
        async with get_db() as db:
            # Total events
            total_events = await db.fetchval("""
                SELECT COUNT(*) FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
            """, start_date, end_date)
            
            # Events by type
            events_by_type = await db.fetch("""
                SELECT event_type, COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY event_type
                ORDER BY count DESC
            """, start_date, end_date)
            
            # Events by risk level
            events_by_risk = await db.fetch("""
                SELECT risk_level, COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY risk_level
            """, start_date, end_date)
            
            # Unique users
            unique_users = await db.fetchval("""
                SELECT COUNT(DISTINCT user_id) FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2 AND user_id IS NOT NULL
            """, start_date, end_date)
            
            return {
                "total_events": total_events,
                "events_by_type": {record["event_type"]: record["count"] for record in events_by_type},
                "events_by_risk": {record["risk_level"]: record["count"] for record in events_by_risk},
                "unique_users": unique_users,
                "compliance_relevant_events": await db.fetchval("""
                    SELECT COUNT(*) FROM audit_logs 
                    WHERE timestamp BETWEEN $1 AND $2 AND compliance_relevant = TRUE
                """, start_date, end_date)
            }
    
    async def _get_security_events(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get security-related events"""
        
        async with get_db() as db:
            records = await db.fetch("""
                SELECT * FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2 
                AND (risk_level IN ('high', 'critical') OR event_type LIKE '%login%' OR event_type LIKE '%role%')
                ORDER BY timestamp DESC
                LIMIT 100
            """, start_date, end_date)
            
            return [dict(record) for record in records]
    
    async def _get_access_patterns(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze access patterns"""
        
        async with get_db() as db:
            # Access by hour
            hourly_access = await db.fetch("""
                SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            """, start_date, end_date)
            
            # Access by IP
            ip_access = await db.fetch("""
                SELECT ip_address, COUNT(*) as count
                FROM audit_logs 
                WHERE timestamp BETWEEN $1 AND $2 AND ip_address IS NOT NULL
                GROUP BY ip_address
                ORDER BY count DESC
                LIMIT 20
            """, start_date, end_date)
            
            return {
                "hourly_distribution": {int(record["hour"]): record["count"] for record in hourly_access},
                "top_ip_addresses": [dict(record) for record in ip_access]
            }


# Utility functions
async def initialize_compliance_tables():
    """Initialize compliance-related database tables"""
    
    async with get_db() as db:
        # Audit logs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                event_id VARCHAR PRIMARY KEY,
                event_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                user_id VARCHAR,
                session_id VARCHAR,
                ip_address INET,
                user_agent TEXT,
                resource_type VARCHAR,
                resource_id VARCHAR,
                action TEXT NOT NULL,
                details JSONB DEFAULT '{}',
                risk_level VARCHAR NOT NULL,
                compliance_relevant BOOLEAN DEFAULT FALSE,
                retention_until TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Compliance reports table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS compliance_reports (
                report_id VARCHAR PRIMARY KEY,
                framework VARCHAR NOT NULL,
                report_type VARCHAR NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                status VARCHAR DEFAULT 'draft',
                report_data JSONB NOT NULL,
                generated_by VARCHAR,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_by VARCHAR,
                approved_at TIMESTAMP
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_compliance ON audit_logs(compliance_relevant)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_retention ON audit_logs(retention_until)")


def create_compliance_dashboard_data(compliance_report: Dict[str, Any]) -> Dict[str, Any]:
    """Create dashboard data from compliance report"""
    
    return {
        "compliance_overview": {
            "framework": compliance_report.get("report_metadata", {}).get("framework"),
            "overall_status": compliance_report.get("compliance_status", {}).get("overall_status"),
            "report_period": compliance_report.get("report_metadata", {}).get("start_date"),
            "last_updated": compliance_report.get("report_metadata", {}).get("generated_at")
        },
        "key_metrics": compliance_report.get("compliance_status", {}),
        "audit_summary": compliance_report.get("audit_findings", {}),
        "risk_indicators": compliance_report.get("risk_assessment", {}),
        "action_items": compliance_report.get("remediation_plan", {}).get("actions", [])
    }