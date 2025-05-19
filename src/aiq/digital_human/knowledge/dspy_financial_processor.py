"""
DSPy-based Financial Document Processing
Handles structured financial data extraction and prompt optimization
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

import dspy
from dspy.predict import Predict
from dspy.modules import ChainOfThought, ReAct
from dspy.teleprompt import BootstrapFewShot
from dspy.metrics import Metric

from aiq.data_models.common import BaseModel
from aiq.utils.debugging_utils import log_function_call


logger = logging.getLogger(__name__)


# DSPy signatures for financial processing
class FinancialDocumentExtraction(dspy.Signature):
    """Extract structured financial information from text."""
    document_text = dspy.InputField(desc="Raw financial document text")
    document_type = dspy.InputField(desc="Type of financial document")
    
    entities = dspy.OutputField(desc="List of financial entities (companies, tickers)")
    metrics = dspy.OutputField(desc="Key financial metrics and values")
    time_periods = dspy.OutputField(desc="Time periods mentioned")
    sentiment = dspy.OutputField(desc="Overall financial sentiment")
    key_facts = dspy.OutputField(desc="Important financial facts")


class FinancialAnalysisGeneration(dspy.Signature):
    """Generate comprehensive financial analysis."""
    financial_data = dspy.InputField(desc="Extracted financial data")
    user_context = dspy.InputField(desc="User profile and preferences")
    query = dspy.InputField(desc="User's specific question or need")
    
    analysis = dspy.OutputField(desc="Detailed financial analysis")
    recommendations = dspy.OutputField(desc="Actionable recommendations")
    risk_assessment = dspy.OutputField(desc="Risk factors and mitigation")
    confidence_score = dspy.OutputField(desc="Confidence in analysis (0-1)")


class RegulatoryCompliance(dspy.Signature):
    """Check regulatory compliance for financial advice."""
    advice_text = dspy.InputField(desc="Financial advice or recommendation")
    user_profile = dspy.InputField(desc="User's financial profile")
    
    compliance_status = dspy.OutputField(desc="Pass/Fail compliance status")
    violations = dspy.OutputField(desc="List of regulatory violations if any")
    required_disclosures = dspy.OutputField(desc="Required compliance disclosures")
    risk_warnings = dspy.OutputField(desc="Mandatory risk warnings")


@dataclass
class FinancialDocument:
    """Financial document data structure"""
    document_id: str
    document_type: str
    content: str
    metadata: Dict[str, Any]
    extracted_data: Optional[Dict[str, Any]] = None
    processed_at: Optional[datetime] = None


@dataclass
class FinancialAnalysis:
    """Financial analysis result"""
    analysis_id: str
    document_id: str
    analysis_text: str
    recommendations: List[str]
    risk_factors: List[str]
    confidence_score: float
    compliance_status: str
    created_at: datetime


class DSPyFinancialProcessor:
    """
    DSPy-based financial document processor with prompt optimization
    Handles extraction, analysis, and compliance checking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize DSPy
        self.lm = dspy.OpenAI(
            model=config.get("model", "gpt-4"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000)
        )
        dspy.settings.configure(lm=self.lm)
        
        # Initialize modules
        self.extractor = ChainOfThought(FinancialDocumentExtraction)
        self.analyzer = ReAct(FinancialAnalysisGeneration, tools=self._get_analysis_tools())
        self.compliance_checker = Predict(RegulatoryCompliance)
        
        # Prompt optimization
        self.optimizer = None
        if config.get("enable_optimization", True):
            self._initialize_optimizer()
        
        # Cache for processed documents
        self.document_cache = {}
        self.cache_duration = config.get("cache_duration", 3600)
        
        # Metrics
        self.metrics = {
            "documents_processed": 0,
            "analyses_generated": 0,
            "compliance_checks": 0,
            "average_confidence": 0.0
        }
        
        logger.info("Initialized DSPy Financial Processor")
    
    def _initialize_optimizer(self):
        """Initialize DSPy prompt optimizer"""
        # Define metric for optimization
        def financial_accuracy_metric(prediction, target):
            """Metric for financial processing accuracy"""
            try:
                # Check if key information is extracted
                score = 0.0
                
                if hasattr(prediction, 'entities') and prediction.entities:
                    score += 0.2
                
                if hasattr(prediction, 'metrics') and prediction.metrics:
                    score += 0.3
                
                if hasattr(prediction, 'key_facts') and prediction.key_facts:
                    score += 0.3
                
                if hasattr(prediction, 'sentiment') and prediction.sentiment:
                    score += 0.2
                
                return score
            except Exception:
                return 0.0
        
        # Create optimizer
        self.optimizer = BootstrapFewShot(
            metric=financial_accuracy_metric,
            max_rounds=3,
            max_labeled_examples=10
        )
        
        logger.info("Initialized DSPy optimizer")
    
    def _get_analysis_tools(self):
        """Get tools for ReAct financial analysis"""
        return [
            self._calculate_financial_ratios,
            self._fetch_market_comparison,
            self._analyze_risk_metrics,
            self._generate_projections
        ]
    
    async def process_document(
        self,
        document: FinancialDocument,
        user_context: Optional[Dict[str, Any]] = None
    ) -> FinancialAnalysis:
        """
        Process a financial document and generate analysis
        
        Args:
            document: Financial document to process
            user_context: User profile and preferences
            
        Returns:
            Complete financial analysis
        """
        # Check cache
        cache_key = f"{document.document_id}_{hash(json.dumps(user_context or {}))}"
        if cache_key in self.document_cache:
            cached_result = self.document_cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).seconds < self.cache_duration:
                logger.debug(f"Cache hit for document {document.document_id}")
                return cached_result['analysis']
        
        try:
            # Extract structured data
            extraction_result = await self._extract_financial_data(document)
            document.extracted_data = extraction_result
            
            # Generate analysis
            analysis_result = await self._generate_analysis(
                document,
                extraction_result,
                user_context
            )
            
            # Check compliance
            compliance_result = await self._check_compliance(
                analysis_result,
                user_context
            )
            
            # Create final analysis
            analysis = FinancialAnalysis(
                analysis_id=f"analysis_{datetime.now().timestamp()}",
                document_id=document.document_id,
                analysis_text=analysis_result['analysis'],
                recommendations=analysis_result['recommendations'],
                risk_factors=analysis_result['risk_assessment'],
                confidence_score=float(analysis_result['confidence_score']),
                compliance_status=compliance_result['compliance_status'],
                created_at=datetime.now()
            )
            
            # Update metrics
            self.metrics["documents_processed"] += 1
            self.metrics["analyses_generated"] += 1
            self.metrics["average_confidence"] = (
                self.metrics["average_confidence"] * 0.9 + 
                analysis.confidence_score * 0.1
            )
            
            # Cache result
            self.document_cache[cache_key] = {
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def _extract_financial_data(self, document: FinancialDocument) -> Dict[str, Any]:
        """Extract structured data from financial document"""
        try:
            # Prepare input
            extraction_input = {
                "document_text": document.content,
                "document_type": document.document_type
            }
            
            # Run extraction
            result = self.extractor(**extraction_input)
            
            # Parse and structure results
            extracted_data = {
                "entities": self._parse_entities(result.entities),
                "metrics": self._parse_metrics(result.metrics),
                "time_periods": self._parse_time_periods(result.time_periods),
                "sentiment": result.sentiment,
                "key_facts": self._parse_key_facts(result.key_facts)
            }
            
            logger.info(f"Extracted data from document {document.document_id}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    async def _generate_analysis(
        self,
        document: FinancialDocument,
        extracted_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate financial analysis using ReAct"""
        try:
            # Prepare analysis input
            analysis_input = {
                "financial_data": json.dumps(extracted_data),
                "user_context": json.dumps(user_context or {}),
                "query": user_context.get("query", "Provide comprehensive financial analysis")
            }
            
            # Run analysis
            result = self.analyzer(**analysis_input)
            
            # Parse results
            analysis_result = {
                "analysis": result.analysis,
                "recommendations": self._parse_recommendations(result.recommendations),
                "risk_assessment": self._parse_risk_assessment(result.risk_assessment),
                "confidence_score": float(result.confidence_score)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            raise
    
    async def _check_compliance(
        self,
        analysis_result: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check regulatory compliance of financial advice"""
        try:
            # Prepare compliance input
            compliance_input = {
                "advice_text": json.dumps(analysis_result),
                "user_profile": json.dumps(user_context or {})
            }
            
            # Run compliance check
            result = self.compliance_checker(**compliance_input)
            
            # Parse results
            compliance_result = {
                "compliance_status": result.compliance_status,
                "violations": self._parse_violations(result.violations),
                "required_disclosures": self._parse_disclosures(result.required_disclosures),
                "risk_warnings": self._parse_warnings(result.risk_warnings)
            }
            
            self.metrics["compliance_checks"] += 1
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            raise
    
    # Parsing helper methods
    def _parse_entities(self, entities_str: str) -> List[Dict[str, str]]:
        """Parse entity string into structured format"""
        try:
            # Attempt to parse as JSON first
            if entities_str.startswith('['):
                return json.loads(entities_str)
            
            # Parse as comma-separated list
            entities = []
            for entity in entities_str.split(','):
                entity = entity.strip()
                if entity:
                    entities.append({
                        "name": entity,
                        "type": "company"  # Default type
                    })
            
            return entities
        except Exception:
            return []
    
    def _parse_metrics(self, metrics_str: str) -> Dict[str, float]:
        """Parse financial metrics"""
        try:
            if metrics_str.startswith('{'):
                return json.loads(metrics_str)
            
            metrics = {}
            # Parse key:value pairs
            for item in metrics_str.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip().replace('$', '').replace('%', ''))
                    except ValueError:
                        metrics[key.strip()] = value.strip()
            
            return metrics
        except Exception:
            return {}
    
    def _parse_time_periods(self, periods_str: str) -> List[str]:
        """Parse time periods"""
        try:
            if periods_str.startswith('['):
                return json.loads(periods_str)
            
            return [period.strip() for period in periods_str.split(',') if period.strip()]
        except Exception:
            return []
    
    def _parse_key_facts(self, facts_str: str) -> List[str]:
        """Parse key facts"""
        try:
            if facts_str.startswith('['):
                return json.loads(facts_str)
            
            return [fact.strip() for fact in facts_str.split('.') if fact.strip()]
        except Exception:
            return []
    
    def _parse_recommendations(self, recommendations_str: str) -> List[str]:
        """Parse recommendations"""
        try:
            if recommendations_str.startswith('['):
                return json.loads(recommendations_str)
            
            return [rec.strip() for rec in recommendations_str.split('\n') if rec.strip()]
        except Exception:
            return []
    
    def _parse_risk_assessment(self, risk_str: str) -> List[str]:
        """Parse risk assessment"""
        try:
            if risk_str.startswith('['):
                return json.loads(risk_str)
            
            return [risk.strip() for risk in risk_str.split('\n') if risk.strip()]
        except Exception:
            return []
    
    def _parse_violations(self, violations_str: str) -> List[str]:
        """Parse compliance violations"""
        try:
            if violations_str.startswith('['):
                return json.loads(violations_str)
            
            return [v.strip() for v in violations_str.split(',') if v.strip()]
        except Exception:
            return []
    
    def _parse_disclosures(self, disclosures_str: str) -> List[str]:
        """Parse required disclosures"""
        try:
            if disclosures_str.startswith('['):
                return json.loads(disclosures_str)
            
            return [d.strip() for d in disclosures_str.split('\n') if d.strip()]
        except Exception:
            return []
    
    def _parse_warnings(self, warnings_str: str) -> List[str]:
        """Parse risk warnings"""
        try:
            if warnings_str.startswith('['):
                return json.loads(warnings_str)
            
            return [w.strip() for w in warnings_str.split('\n') if w.strip()]
        except Exception:
            return []
    
    # ReAct tools
    def _calculate_financial_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate standard financial ratios"""
        ratios = {}
        
        # P/E Ratio
        if 'price' in metrics and 'earnings' in metrics:
            ratios['pe_ratio'] = metrics['price'] / metrics['earnings']
        
        # Debt-to-Equity
        if 'debt' in metrics and 'equity' in metrics:
            ratios['debt_to_equity'] = metrics['debt'] / metrics['equity']
        
        # Return on Equity
        if 'net_income' in metrics and 'equity' in metrics:
            ratios['roe'] = metrics['net_income'] / metrics['equity']
        
        return ratios
    
    def _fetch_market_comparison(self, entity: str) -> Dict[str, Any]:
        """Fetch market comparison data (mock)"""
        # In production, this would fetch real market data
        return {
            "industry_average_pe": 15.5,
            "sector_performance": 0.08,
            "peer_comparison": {
                "revenue_growth": 0.12,
                "profit_margin": 0.15
            }
        }
    
    def _analyze_risk_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze financial risk metrics"""
        risk_metrics = {
            "volatility": np.random.uniform(0.1, 0.3),
            "beta": np.random.uniform(0.8, 1.2),
            "sharpe_ratio": np.random.uniform(0.5, 2.0),
            "max_drawdown": np.random.uniform(0.1, 0.4)
        }
        
        return risk_metrics
    
    def _generate_projections(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial projections"""
        # Simplified projection model
        projections = {
            "revenue_growth": np.random.uniform(0.05, 0.15),
            "profit_margin": np.random.uniform(0.10, 0.25),
            "cash_flow": np.random.uniform(1000000, 10000000)
        }
        
        return projections
    
    async def optimize_prompts(self, training_data: List[Tuple[FinancialDocument, FinancialAnalysis]]):
        """Optimize DSPy prompts using training data"""
        if not self.optimizer:
            logger.warning("Optimizer not initialized")
            return
        
        try:
            # Prepare training examples
            examples = []
            for document, analysis in training_data:
                example = {
                    "inputs": {
                        "document_text": document.content,
                        "document_type": document.document_type
                    },
                    "outputs": {
                        "entities": document.extracted_data.get("entities", []),
                        "metrics": document.extracted_data.get("metrics", {}),
                        "key_facts": document.extracted_data.get("key_facts", [])
                    }
                }
                examples.append(example)
            
            # Run optimization
            optimized_extractor = self.optimizer.compile(
                student=self.extractor,
                trainset=examples
            )
            
            # Update extractor
            self.extractor = optimized_extractor
            
            logger.info("Prompts optimized successfully")
            
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        return {
            "documents_processed": self.metrics["documents_processed"],
            "analyses_generated": self.metrics["analyses_generated"],
            "compliance_checks": self.metrics["compliance_checks"],
            "average_confidence": self.metrics["average_confidence"],
            "cache_size": len(self.document_cache)
        }
    
    async def process_batch(
        self,
        documents: List[FinancialDocument],
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[FinancialAnalysis]:
        """Process multiple documents in batch"""
        tasks = []
        
        for document in documents:
            task = asyncio.create_task(
                self.process_document(document, user_context)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        analyses = []
        for result in results:
            if isinstance(result, FinancialAnalysis):
                analyses.append(result)
            else:
                logger.error(f"Batch processing error: {result}")
        
        return analyses
    
    def clear_cache(self):
        """Clear document cache"""
        self.document_cache.clear()
        logger.info("Document cache cleared")